import os
import pickle
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand
from pyro.poutine.util import prune_subsample_sites
import pyro
import pyro.distributions as dist
from pyro.poutine import trace, replay, block, condition
from tqdm import trange
import mlflow
import mlflow.pytorch

from neural.modules import (
    SetEquivariantDesignNetwork,
    BatchDesignBaseline,
    LazyDelta,
)

from oed.primitives import observation_sample, compute_design
from experiment_tools.pyro_tools import auto_seed

from oed.design import OED
from contrastive.mi import (
    PriorContrastiveEstimationScoreGradient,
    PriorContrastiveEstimationDiscreteObsTotalEnum,
)

from extra_distributions.truncated_normal import LowerTruncatedNormal

class EncoderNetwork(nn.Module):
    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim, n_hidden_layers=2, activation=nn.Softplus):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.activation_layer = activation()
        self.input_layer = nn.Linear(design_dim + observation_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, **kwargs):
        # Print the shapes for debugging
        print(f"xi shape: {xi.shape}")
        print(f"y shape: {y.shape}")

        # Ensure xi and y have the same number of dimensions
        if xi.dim() > y.dim():
            # Expand y to match the dimensions of xi
            for _ in range(xi.dim() - y.dim()):
                y = y.unsqueeze(1)
        elif y.dim() > xi.dim():
            # Expand xi to match the dimensions of y
            for _ in range(y.dim() - xi.dim()):
                xi = xi.unsqueeze(1)

        # Now xi and y should have the same number of dimensions
        # Broadcast the smaller dimension (if one is 1 and the other is > 1) to match the larger size
        y = y.expand_as(xi)

        # Now concatenate along the last dimension
        inputs = torch.cat([xi, y], dim=-1)

        x = self.input_layer(inputs)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x



class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=2, activation=nn.Softplus):
        super().__init__()
        self.activation_layer = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation())
                    for _ in range(n_hidden_layers - 1)
                ]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        x = self.input_layer(r)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


class SEIRSDEModel(nn.Module):
    # (Constructor remains the same)

    def model(self):
        pyro.module("design_net", self.design_net)

        # Initial conditions
        S = pyro.sample("S_0", dist.Delta(torch.tensor(self.S0).expand(self.num_outer_samples, self.num_inner_samples)))
        E = pyro.sample("E_0", dist.Delta(torch.tensor(self.E0).expand(self.num_outer_samples, self.num_inner_samples)))
        I = pyro.sample("I_0", dist.Delta(torch.tensor(self.I0).expand(self.num_outer_samples, self.num_inner_samples)))
        R = pyro.sample("R_0", dist.Delta(torch.tensor(self.R0).expand(self.num_outer_samples, self.num_inner_samples)))

        y_outcomes = []
        xi_designs = []

        for t in range(self.T):
            plate_name_outer = f"outer_plate_{t}"
            plate_name_inner = f"inner_plate_{t}"

            with pyro.plate(plate_name_outer, size=self.num_outer_samples, dim=-3):
                with pyro.plate(plate_name_inner, size=self.num_inner_samples, dim=-4):
                    # Design computation
                    xi = compute_design(f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes)))
                    xi = torch.relu(xi)  # Ensure design is positive

                    # Expand the dimensions of xi if necessary
                    if xi.dim() == 2:  # (batch_size, design_dim)
                        xi = xi.unsqueeze(1).unsqueeze(2)  # Add dimensions to (1, 1, batch_size, design_dim)
                    elif xi.dim() == 3:  # (batch_size, time, design_dim)
                        xi = xi.unsqueeze(1)  # Add an additional dimension (1, batch_size, time, design_dim)

                    # Deterministic components of SEIR model
                    dS = (-self.beta * S * I / self.N) * self.dt
                    dE = (self.beta * S * I / self.N - self.sigma * E) * self.dt
                    dI = (self.sigma * E - self.gamma * I) * self.dt
                    dR = (self.gamma * I) * self.dt

                    # Ensure the shapes are consistent
                    dS = dS.unsqueeze(-1)
                    dE = dE.unsqueeze(-1)
                    dI = dI.unsqueeze(-1)
                    dR = dR.unsqueeze(-1)

                    # Batch size handling
                    batch_size = S.shape[0]

                    # Initialize the diffusion terms for each component
                    g_S = torch.zeros(batch_size, 4, device=S.device)
                    g_S[:, 0] = -torch.sqrt(self.beta * S * I / self.N).squeeze(-1)

                    g_E = torch.zeros(batch_size, 4, device=S.device)
                    g_E[:, 0] = torch.sqrt(self.beta * S * I / self.N).squeeze(-1)
                    g_E[:, 1] = -torch.sqrt(self.sigma * E).squeeze(-1)

                    g_I = torch.zeros(batch_size, 4, device=S.device)
                    g_I[:, 1] = torch.sqrt(self.sigma * E).squeeze(-1)
                    g_I[:, 2] = -torch.sqrt(self.gamma * I).squeeze(-1)

                    g_R = torch.zeros(batch_size, 4, device=S.device)
                    g_R[:, 2] = torch.sqrt(self.gamma * I).squeeze(-1)

                    # Sample Wiener process increments (4, 1)
                    dW = torch.randn(4, 1, device=S.device) * torch.sqrt(torch.tensor(self.dt, device=S.device))

                    # Apply the stochastic updates
                    S = S + dS + (g_S @ dW).squeeze(-1)
                    E = E + dE + (g_E @ dW).squeeze(-1)
                    I = I + dI + (g_I @ dW).squeeze(-1)
                    R = R + dR + (g_R @ dW).squeeze(-1)

                    # Generate y_outcome with correct dimensions
                    y_outcome = torch.stack([S, E, I, R], dim=-1)  # (batch_size, 4)
                    y_outcome = y_outcome.unsqueeze(0)  # Convert to (1, batch_size, 4)

                    # Observation
                    pyro.sample(f"y{t + 1}", dist.Delta(y_outcome).to_event(2), obs=y_outcome)

                    # Store outcomes
                    y_outcomes.append(y_outcome)
                    xi_designs.append(xi)

        return y_outcomes


def single_run(
    seed,
    num_steps,
    num_inner_samples,
    num_outer_samples,
    lr,
    gamma_val,
    T,
    N,
    device,
    hidden_dim,
    encoding_dim,
    num_layers,
    arch,
    complete_enum,
    mlflow_experiment_name,
    beta,
    sigma,
    gamma,
    initial_values,
):

    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)
    mlflow.set_experiment(mlflow_experiment_name)

    mlflow.log_param("seed", seed)
    mlflow.log_param("num_experiments", T)
    mlflow.log_param("lr", lr)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("encoding_dim", encoding_dim)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("gamma_val", gamma_val)
    mlflow.log_param("complete_enum", complete_enum)
    mlflow.log_param("arch", arch)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)

    # Set up model based on experiment type (arch)
    if arch == "static":
        design_net = BatchDesignBaseline(T, 4).to(device)
    elif arch == "sum":
        encoder = EncoderNetwork(4, 4, hidden_dim, encoding_dim, n_hidden_layers=num_layers)
        emitter = EmitterNetwork(encoding_dim, hidden_dim, 4, n_hidden_layers=num_layers)
        design_net = SetEquivariantDesignNetwork(encoder, emitter, empty_value=torch.ones(4)).to(device)
    elif arch == "variational":
        design_net = LazyDelta(T, 4).to(device)
    else:
        raise ValueError(f"Unexpected architecture specification: '{arch}'.")

    seir_process = SEIRSDEModel(
        design_net=design_net,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        N=N,
        dt=0.1,  # example time increment
        T=T,
        num_inner_samples=num_inner_samples,
        num_outer_samples=num_outer_samples,
        initial_values=initial_values,
    )

    # Annealed LR optimizer
    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": lr, "betas": [0.9, 0.999], "weight_decay": 0,},
            "gamma": gamma_val,
        }
    )
    if complete_enum:
        pce_loss = PriorContrastiveEstimationDiscreteObsTotalEnum(num_outer_samples, num_inner_samples)
    else:
        pce_loss = PriorContrastiveEstimationScoreGradient(num_outer_samples, num_inner_samples)

    # Corrected OED initialization
    oed = OED(scheduler, pce_loss)

    # Optimize
    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    for i in t:
        loss = oed.step(seir_process.model)
        loss = torch_item(loss)
        t.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 50 == 0:
            mlflow.log_metric("loss", oed.evaluate_loss(seir_process.model))
        if i % 1000 == 0:
            scheduler.step()

    mlflow.log_metric("loss_diff50", np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1)

    # Evaluate and store results
    runs_output = seir_process.eval()
    results = {
        "design_network": design_net.cpu(),
        "seed": seed,
        "loss_history": loss_history,
        "runs_output": runs_output,
    }

    # Log model
    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(seir_process.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model stored in {model_loc}.")

    print(f"Run completed {mlflow.active_run().info.artifact_uri}.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Adaptive Design example: SEIR SDE Model.")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=100, type=int)
    parser.add_argument("--num-inner-samples", default=50, type=int)
    parser.add_argument("--num-outer-samples", default=50, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma_val", default=0.95, type=float)
    parser.add_argument("--num-experiments", default=4, type=int)  # == T
    parser.add_argument("--num-people", default=1000, type=int)  # == N
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--complete-enum", default=False, type=bool)
    parser.add_argument("--num-layers", default=2, type=int, help="Number of hidden layers.")
    parser.add_argument("--arch", default="sum", type=str, help="Architecture", choices=["static", "sum", "variational"])
    parser.add_argument("--mlflow-experiment-name", default="Default", type=str)
    parser.add_argument("--beta", default=0.3, type=float, help="Transmission rate")
    parser.add_argument("--sigma", default=0.1, type=float, help="Rate from E to I")
    parser.add_argument("--gamma", default=0.1, type=float, help="Recovery rate")
    parser.add_argument("--initial_values", nargs=4, type=int, default=[999, 0, 1, 0], help="Initial values for S, E, I, R")

    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma_val=args.gamma_val,
        device=args.device,
        T=args.num_experiments,
        N=args.num_people,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        num_layers=args.num_layers,
        arch=args.arch,
        complete_enum=args.complete_enum,
        mlflow_experiment_name=args.mlflow_experiment_name,
        beta=args.beta,
        sigma=args.sigma,
        gamma=args.gamma,
        initial_values=args.initial_values,
    )
