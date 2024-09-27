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
from tqdm import trange

import mlflow
import mlflow.pytorch

from neural.modules import (
    SetEquivariantDesignNetwork,
    BatchDesignBaseline,
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
    def __init__(
        self,
        design_dim,
        observation_dim,
        hidden_dim,
        encoding_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
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
        if xi.dim() == 0:
            xi = xi.unsqueeze(0)
    
        # Flatten y (4*(T/dt)) into a 1D tensor
        y = y.view(-1)

        inputs = torch.cat([xi, y], dim=-1)
        x = self.input_layer(inputs)
        x = self.activation_layer(x)
        x = self.middle(x)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_hidden_layers=2,
        activation=nn.Softplus,
    ):
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


def seir_simulation(beta, sigma, gamma, initial_values, N, dt, T):
    num_steps = int(T / dt)
    S, E, I, R = initial_values

    S = torch.tensor([S], dtype=torch.float32)
    E = torch.tensor([E], dtype=torch.float32)
    I = torch.tensor([I], dtype=torch.float32)
    R = torch.tensor([R], dtype=torch.float32)

    S_values, E_values, I_values, R_values = [], [], [], []

    for step in range(num_steps):
        S = S.clamp(min=1e-6)
        E = E.clamp(min=1e-6)
        I = I.clamp(min=1e-6)
        R = R.clamp(min=1e-6)

        dS = (-beta * S * I / N) * dt
        dE = (beta * S * I / N - sigma * E) * dt
        dI = (sigma * E - gamma * I) * dt
        dR = (gamma * I) * dt

        dS = dS.unsqueeze(-1)
        dE = dE.unsqueeze(-1)
        dI = dI.unsqueeze(-1)
        dR = dR.unsqueeze(-1)

        batch_size = S.shape[0]

        g_S = torch.zeros(batch_size, 4)
        g_S[:, 0] = -torch.sqrt(beta * S * I / N).squeeze(-1)

        g_E = torch.zeros(batch_size, 4)
        g_E[:, 0] = torch.sqrt(beta * S * I / N).squeeze(-1)
        g_E[:, 1] = -torch.sqrt(sigma * E).squeeze(-1)

        g_I = torch.zeros(batch_size, 4)
        g_I[:, 1] = torch.sqrt(sigma * E).squeeze(-1)
        g_I[:, 2] = -torch.sqrt(gamma * I).squeeze(-1)

        g_R = torch.zeros(batch_size, 4)
        g_R[:, 2] = torch.sqrt(gamma * I).squeeze(-1)

        dW = torch.randn(4, 1) * torch.sqrt(torch.tensor(dt))

        S = S + dS + (g_S @ dW).squeeze(-1)
        E = E + dE + (g_E @ dW).squeeze(-1)
        I = I + dI + (g_I @ dW).squeeze(-1)
        R = R + dR + (g_R @ dW).squeeze(-1)

        # Debugging output
        if torch.isnan(S).any() or torch.isnan(E).any() or torch.isnan(I).any() or torch.isnan(R).any():
            print(f"Step {step}: S={S.item()}, E={E.item()}, I={I.item()}, R={R.item()}")
            raise ValueError("NaN detected in SEIR simulation values.")

        S_values.append(S.item())
        E_values.append(E.item())
        I_values.append(I.item())
        R_values.append(R.item())

    return S_values, E_values, I_values, R_values


class SEIRSDEModel(nn.Module):

    def __init__(
        self,
        design_net,
        beta,
        sigma,
        gamma,
        initial_values,
        N,
        T,
        dt,
        theta_loc=None,
        theta_scale=None,
        theta_dist="truncated normal",
    ):
        super().__init__()
        self.design_net = design_net
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.initial_values = initial_values
        self.N = N
        self.T = T
        self.dt = dt
        self.theta_loc = theta_loc if theta_loc is not None else torch.tensor([beta, sigma, gamma])
        self.theta_scale = theta_scale if theta_scale is not None else torch.tensor([0.1, 0.1, 0.1])
        self.theta_dist = theta_dist
        if theta_dist == "truncated normal":
            self.theta_prior_dist = LowerTruncatedNormal(
                self.theta_loc, self.theta_scale, 0.0
            )
        elif theta_dist == "lognormal":
            self.theta_prior_dist = dist.LogNormal(self.theta_loc, self.theta_scale)
        else:
            raise ValueError("Invalid option: `theta_dist`=%s." % theta_dist)
        self.softplus = nn.Softplus()

        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        self.y_outcomes = []
        self.xi_designs = []

        theta = pyro.sample("theta", self.theta_prior_dist)
        theta = theta.clamp(min=1e-10, max=1e10)

        beta, sigma, gamma = theta

        for t in range(self.T):
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(self.xi_designs, self.y_outcomes))
            )
            xi = self.softplus(xi.squeeze(-1))

            S_values, E_values, I_values, R_values = seir_simulation(
                beta=beta, sigma=sigma, gamma=gamma,
                initial_values=self.initial_values, N=self.N, dt=self.dt, T=self.T
            )

            # Convert I_values[-1] to a tensor before checking for NaN
            if torch.isnan(torch.tensor(I_values[-1])):
                raise ValueError(f"NaN detected in I_values[-1] at time {t}")

            y = observation_sample(f"y{t + 1}", dist.Normal(I_values[-1], 0.1))
            self.y_outcomes.append(y)
            self.xi_designs.append(xi)

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)

        theta = pyro.sample("theta", self.theta_prior_dist)
        theta = theta.clamp(min=1e-10, max=1e10)

        beta, sigma, gamma = theta

        y_outcomes = []  # To store S, E, I, and optionally R values for each time step
        xi_designs = []

        for t in range(self.T):
            xi = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            xi = self.softplus(xi.squeeze(-1))

            # Simulate the SEIR model
            S_values, E_values, I_values, R_values = seir_simulation(
                beta=beta, sigma=sigma, gamma=gamma,
                initial_values=self.initial_values, N=self.N, dt=self.dt, T=self.T
            )

            # Store the S, E, I, and optionally R values for this time step
            y_outcomes.append((S_values[-1], E_values[-1], I_values[-1], R_values[-1]))

            # Generate observation for I values
            y = observation_sample(f"y{t + 1}", dist.Normal(I_values[-1], 0.1))
            xi_designs.append(xi)

        # Convert y_outcomes to a 4 x (T/dt) matrix
        y_outcomes = torch.stack([torch.tensor(outcome) for outcome in y_outcomes], dim=1)

        return y_outcomes


    def eval(self, n_trace=2, theta=None):
        self.design_net.eval()
        
        output = []
        with torch.no_grad():
            for i in range(n_trace):
                if theta is not None:
                    model = pyro.condition(self.model, data={"theta": theta})
                else:
                    model = self.model
                trace = pyro.poutine.trace(model).get_trace()
                true_theta = trace.nodes["theta"]["value"].numpy()
                run_xis = []
                run_ys = []

                for t in range(self.T):
                    xi = trace.nodes[f"xi{t + 1}"]["value"].item()
                    run_xis.append(xi)

                    y = trace.nodes[f"y{t + 1}"]["value"].item()
                    run_ys.append(y)

                run_df = pd.DataFrame(
                    {
                        "designs": run_xis,
                        "observations": run_ys,
                        "order": list(range(1, self.T + 1)),
                    }
                )
                run_df["run_id"] = i + 1
                run_df["theta"] = true_theta
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
            trace.nodes["theta"]["value"] = torch.tensor([1.50], device=trace.nodes["theta"]["value"].device)
            trace = prune_subsample_sites(trace)
            trace.compute_log_prob()

            data = {
                name: lexpand(node["value"], grid_size)
                for name, node in trace.nodes.items()
                if node.get("subtype") in ["observation_sample", "design_sample"]
            }
            data["theta"] = rexpand(grid, n_rollout)

            def conditional_model():
                with pyro.plate_stack("vectorization", (grid_size, n_rollout)):
                    pyro.condition(self.model, data=data)()

            condition_trace = pyro.poutine.trace(conditional_model).get_trace()
            condition_trace = prune_subsample_sites(condition_trace)
            condition_trace.compute_log_prob()

        return condition_trace

def single_run(
    seed,
    num_steps,
    num_inner_samples,
    num_outer_samples,
    lr,
    gamma_val,  # Optimization parameter for learning rate scheduler
    T,
    N,
    beta,
    sigma,
    gamma,  # SEIR model recovery rate
    initial_values,
    dt,
    device,
    hidden_dim,
    encoding_dim,
    num_layers,
    arch,
    complete_enum,
    mlflow_experiment_name,
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
    mlflow.log_param("gamma_val", gamma_val)  # Logging the optimization parameter
    mlflow.log_param("complete_enum", complete_enum)
    mlflow.log_param("arch", arch)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)

    # Set up the design network based on the architecture choice
    if arch == "static":
        design_net = BatchDesignBaseline(T, 3).to(device)
    elif arch == "sum":
        encoder = EncoderNetwork(
            design_dim=3, observation_dim=1, hidden_dim=hidden_dim, 
            encoding_dim=encoding_dim, n_hidden_layers=num_layers
        )
        emitter = EmitterNetwork(
            input_dim=encoding_dim, hidden_dim=hidden_dim, output_dim=3, 
            n_hidden_layers=num_layers
        )
        design_net = SetEquivariantDesignNetwork(
            encoder, emitter, empty_value=torch.ones(3)
        ).to(device)
    else:
        raise ValueError(f"Unexpected architecture specification: '{arch}'.")

    theta_prior_loc = torch.tensor([beta, sigma, gamma], device=device)
    theta_prior_scale = torch.tensor([0.1, 0.1, 0.1], device=device)
    seir_sde_model = SEIRSDEModel(
        design_net=design_net,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        initial_values=initial_values,
        N=N,
        T=T,
        dt=dt,
        theta_loc=theta_prior_loc,
        theta_scale=theta_prior_scale,
    )

    optimizer = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR(
        {
            "optimizer": optimizer,
            "optim_args": {"lr": lr, "betas": [0.9, 0.999], "weight_decay": 0,},
            "gamma": gamma_val,  # Used for learning rate decay
        }
    )
    
    if complete_enum:
        pce_loss = PriorContrastiveEstimationDiscreteObsTotalEnum(
            num_outer_samples, num_inner_samples
        )
    else:
        pce_loss = PriorContrastiveEstimationScoreGradient(
            num_outer_samples, num_inner_samples
        )
    
    # Ensure correct number of arguments passed to OED
    oed = OED(seir_sde_model.model, scheduler, pce_loss)

    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    for i in t:
        loss = oed.step()
        loss = torch_item(loss)
        t.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 50 == 0:
            mlflow.log_metric("loss", oed.evaluate_loss())
        if i % 1000 == 0:
            scheduler.step()

    mlflow.log_metric(
        "loss_diff50", np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1
    )

    runs_output = seir_sde_model.eval()
    results = {
        "design_network": design_net.cpu(),
        "seed": seed,
        "loss_history": loss_history,
        "runs_output": runs_output,
    }

    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(seir_sde_model.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model stored in {model_loc}.")

    print(f"Run completed {mlflow.active_run().info.artifact_uri}.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Adaptive Design example: SEIR SDE Model."
    )
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num_steps", default=5000, type=int)
    parser.add_argument("--num_inner_samples", default=100, type=int)
    parser.add_argument("--num_outer_samples", default=200, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma_val", default=0.95, type=float)  # Optimization parameter
    parser.add_argument("--num_experiments", default=4, type=int)
    parser.add_argument("--beta", default=0.3, type=float)  # SEIR model parameter
    parser.add_argument("--sigma", default=0.1, type=float)  # SEIR model parameter
    parser.add_argument("--gamma", default=0.1, type=float)  # SEIR model parameter
    parser.add_argument("--num_people", default=100, type=int)
    parser.add_argument("--initial_values", default=[99, 1, 0, 0], nargs=4, type=float)
    parser.add_argument("--dt", default=1.0, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--complete-enum", default=False, type=bool)
    parser.add_argument(
        "--num-layers", default=2, type=int, help="Number of hidden layers."
    )
    parser.add_argument(
        "--arch",
        default="sum",
        type=str,
        help="Architecture",
        choices=["static", "sum"],
    )
    parser.add_argument("--mlflow-experiment-name", default="Default", type=str)
    args = parser.parse_args()

    single_run(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma_val=args.gamma_val,  # Pass as optimization parameter
        device=args.device,
        T=args.num_experiments,
        N=args.num_people,
        beta=args.beta,
        sigma=args.sigma,
        gamma=args.gamma,  # Pass as SEIR model parameter
        initial_values=args.initial_values,
        dt=args.dt,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        num_layers=args.num_layers,
        arch=args.arch,
        complete_enum=args.complete_enum,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
