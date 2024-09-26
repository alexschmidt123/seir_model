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
        self.middle = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation()) for _ in range(n_hidden_layers - 1)]) if n_hidden_layers > 1 else nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y):
        inputs = torch.cat([xi.expand_as(y), y], dim=-1)
        x = self.activation_layer(self.input_layer(inputs))
        x = self.middle(x)
        return self.output_layer(x)

class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=2, activation=nn.Softplus):
        super().__init__()
        self.activation_layer = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.middle = nn.Sequential(*[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation()) for _ in range(n_hidden_layers - 1)]) if n_hidden_layers > 1 else nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, r):
        x = self.activation_layer(self.input_layer(r))
        x = self.middle(x)
        return self.output_layer(x)

class SEIR_SDE_Model(nn.Module):
    """Model class for SEIR SDE experiment."""

    def __init__(self, design_net, beta=None, sigma=None, gamma=None, N=50, T=2):
        super().__init__()
        # SEIR model parameters
        self.design_net = design_net
        self.beta = beta if beta is not None else torch.tensor(0.3)  # Default transmission rate
        self.sigma = sigma if sigma is not None else torch.tensor(0.1)  # Default exposed to infected rate
        self.gamma = gamma if gamma is not None else torch.tensor(0.1)  # Default recovery rate
        self.N = N  # Number of people
        self.T = T  # Total time steps
        self.softplus = nn.Softplus()

    def seir_simulation(self, initial_values, dt):
        # Calculate the number of steps based on total time T and time step dt
        num_steps = int(self.T / dt)

        # Unpack initial values
        S, E, I, R = initial_values

        # Initialize tensors for S, E, I, R
        S = torch.tensor([S], dtype=torch.float32)
        E = torch.tensor([E], dtype=torch.float32)
        I = torch.tensor([I], dtype=torch.float32)
        R = torch.tensor([R], dtype=torch.float32)

        # Store the results over time
        S_values, E_values, I_values, R_values = [], [], [], []

        for _ in range(num_steps):
            # Deterministic components of SEIR model
            dS = (-self.beta * S * I / self.N) * dt
            dE = (self.beta * S * I / self.N - self.sigma * E) * dt
            dI = (self.sigma * E - self.gamma * I) * dt
            dR = (self.gamma * I) * dt

            # Ensure the shapes are consistent
            dS = dS.unsqueeze(-1)
            dE = dE.unsqueeze(-1)
            dI = dI.unsqueeze(-1)
            dR = dR.unsqueeze(-1)

            # Batch size handling (single batch in this case)
            batch_size = S.shape[0]

            # Initialize the diffusion terms for each component
            g_S = torch.zeros(batch_size, 4)
            g_S[:, 0] = -torch.sqrt(self.beta * S * I / self.N).squeeze(-1)  # g_SS component

            g_E = torch.zeros(batch_size, 4)
            g_E[:, 0] = torch.sqrt(self.beta * S * I / self.N).squeeze(-1)  # g_ES component
            g_E[:, 1] = -torch.sqrt(self.sigma * E).squeeze(-1)        # g_EE component

            g_I = torch.zeros(batch_size, 4)
            g_I[:, 1] = torch.sqrt(self.sigma * E).squeeze(-1)         # g_IE component
            g_I[:, 2] = -torch.sqrt(self.gamma * I).squeeze(-1)        # g_II component

            g_R = torch.zeros(batch_size, 4)
            g_R[:, 2] = torch.sqrt(self.gamma * I).squeeze(-1)         # g_RI component

            # Sample Wiener process increments (4, 1)
            dW = torch.randn(4, 1) * torch.sqrt(torch.tensor(dt))

            # Apply the stochastic updates according to the SDEs
            S = S + dS + (g_S @ dW).squeeze(-1)
            E = E + dE + (g_E @ dW).squeeze(-1)
            I = I + dI + (g_I @ dW).squeeze(-1)
            R = R + dR + (g_R @ dW).squeeze(-1)

            # Store results
            S_values.append(S.item())
            E_values.append(E.item())
            I_values.append(I.item())
            R_values.append(R.item())

        return S_values, E_values, I_values, R_values

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        # Initial values for S, E, I, R
        initial_values = (self.N - 1, 0, 1, 0)  # Start with 1 infected, rest susceptible
        dt = 0.1  # Set a default time step for simulation
        
        # Call the SEIR simulation method
        y_outcomes = self.seir_simulation(initial_values, dt)

        return y_outcomes

    def eval(self, n_trace=2):
        self.design_net.eval()
        output = []
        with torch.no_grad():
            for i in range(n_trace):
                trace = pyro.poutine.trace(self.model).get_trace()

                # Extract the results from the trace
                S = trace.nodes["S"]["value"].item()
                E = trace.nodes["E"]["value"].item()
                I = trace.nodes["I"]["value"].item()
                R = trace.nodes["R"]["value"].item()

                run_df = pd.DataFrame({
                    "S": S,
                    "E": E,
                    "I": I,
                    "R": R,
                    "run_id": i + 1
                })
                output.append(run_df)

        return pd.concat(output)

    def rollout(self, n_rollout, grid):
        self.design_net.eval()
        
        grid_size = grid.shape[0]

        def vectorized_model():
            with pyro.plate("vectorization", n_rollout):
                return self.model()

        with torch.no_grad():
            trace = pyro.poutine.trace(vectorized_model).get_trace()
            # Using a fixed value for theta as an example; adjust based on your model logic
            trace.nodes["theta"]["value"] = torch.tensor([1.50], device=trace.nodes["theta"]["value"].device)
            trace = pyro.poutine.prune(trace)
            trace.compute_log_prob()

            data = {
                name: node["value"].expand(grid_size) 
                for name, node in trace.nodes.items()
                if node.get("subtype") in ["observation_sample", "design_sample"]
            }
            data["theta"] = grid.expand(n_rollout, -1)  # Expand grid for theta

            def conditional_model():
                with pyro.plate_stack("vectorization", (grid_size, n_rollout)):
                    pyro.condition(self.model, data=data)()

            condition_trace = pyro.poutine.trace(conditional_model).get_trace()
            condition_trace = pyro.poutine.prune(condition_trace)
            condition_trace.compute_log_prob()

        return condition_trace

def single_run(
    seed, num_steps, num_inner_samples, num_outer_samples, lr, gamma_val, T, N, device, hidden_dim, encoding_dim, num_layers, arch, complete_enum, mlflow_experiment_name, beta, sigma, gamma, initial_values
):
    pyro.clear_param_store()
    seed = auto_seed(seed)
    pyro.set_rng_seed(seed)
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_param("seed", seed)

    if arch == "static":
        design_net = BatchDesignBaseline(T, 4).to(device)
    elif arch == "sum":
        encoder = EncoderNetwork(4, 4, hidden_dim, encoding_dim, n_hidden_layers=num_layers)
        emitter = EmitterNetwork(encoding_dim, hidden_dim, 4, n_hidden_layers=num_layers)
        design_net = SetEquivariantDesignNetwork(encoder, emitter, empty_value=torch.ones(4)).to(device)
    else:
        raise ValueError(f"Unexpected architecture specification: '{arch}'.")

    seir_process = SEIR_SDE_Model(
        design_net=design_net,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        N=N,
        dt=0.1,
        T=T,
        num_inner_samples=num_inner_samples,
        num_outer_samples=num_outer_samples,
        initial_values=initial_values,
    )

    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({"optimizer": optimizer, "optim_args": {"lr": lr}, "gamma": gamma_val})
    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    for i in t:
        loss = OED.step(seir_process.model)
        loss_history.append(loss)

    return loss_history

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
