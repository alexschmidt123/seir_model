import os
import argparse
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer.util import torch_item
from pyro.contrib.util import lexpand, rexpand
from pyro.poutine.util import prune_subsample_sites
from tqdm import trange
import pyro.optim as optim
import mlflow
import mlflow.pytorch

from oed.primitives import observation_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from neural.modules import SetEquivariantDesignNetwork, BatchDesignBaseline
from contrastive.mi import (
    PriorContrastiveEstimationScoreGradient,
    PriorContrastiveEstimationDiscreteObsTotalEnum,
)

class EncoderNetwork(nn.Module):
    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim, n_hidden_layers=2, activation=nn.Softplus):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.activation_layer = activation()
        self.input_layer = nn.Linear(design_dim + observation_dim, hidden_dim)
        if n_hidden_layers > 1:
            self.middle = nn.Sequential(
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation()) for _ in range(n_hidden_layers - 1)]
            )
        else:
            self.middle = nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)

    def forward(self, xi, y, **kwargs):
        while y.dim() < xi.dim():
            y = y.unsqueeze(-1)
        if xi.dim() > y.dim():
            xi = xi.unsqueeze(-1)
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
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation()) for _ in range(n_hidden_layers - 1)]
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

class SEIRDesignModel(nn.Module):
    def __init__(self, design_net, simdata, T=4, lower_bound=1e-2, upper_bound=100.0 - 1e-2):
        super().__init__()
        self.design_net = design_net
        self.simdata = simdata
        self.T = T
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.theta_dim = simdata["prior_samples"].shape[1]

    def simulator(self, xi, theta, batch_data):
        nearest = torch.min(torch.abs(xi.unsqueeze(-1) - batch_data["ts"][1:-1]), axis=1).indices
        y = batch_data["ys"][:, nearest, :].reshape(-1, 4)
        return y[:, 2].reshape(-1, 1)  # Focus on the infected compartment


    def model(self):
        xi_designs = []
        y_outcomes = []
        xi_prev = self.lower_bound

        # Prepare batch_data for use in the model
        batch_data = {
            "ys": self.simdata["ys"],
            "ts": self.simdata["ts"],
            "prior_samples": self.simdata["prior_samples"],
            "dt": self.simdata["dt"],
        }

        batch_size, num_time_points, _ = self.simdata["ys"].shape  # Should be [10000, 10, 4]
        
        # Split the batch_size into smaller batches (e.g., 1000)
        smaller_batch_size = 1000
        num_batches = batch_size // smaller_batch_size

        for t in range(self.T):
            xi_name = f"xi_{t + 1}_design"
            xi_untransformed_tensor = compute_design(xi_name, self.design_net.lazy(*zip(xi_designs, y_outcomes)))

            print(f"xi_untransformed_tensor shape: {xi_untransformed_tensor.shape}")

            # Process in smaller batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * smaller_batch_size
                end_idx = (batch_idx + 1) * smaller_batch_size
                
                # Get the corresponding slice of the batch
                xi = xi_untransformed_tensor.unsqueeze(-1)[start_idx:end_idx]
                xi = xi.repeat(smaller_batch_size // 200, 10, 1)  # Expand to [1000, 10, 1] for this batch

                print(f"xi shape after expansion: {xi.shape} for batch {batch_idx + 1}/{num_batches}")

                # Get the relevant portion of batch_data
                batch_data_slice = {
                    "ys": batch_data["ys"][start_idx:end_idx],
                    "ts": batch_data["ts"],
                    "prior_samples": batch_data["prior_samples"][start_idx:end_idx],
                    "dt": batch_data["dt"],
                }

                # Sample for this smaller batch
                y = observation_sample(f"y_{t + 1}_obs", self.simulator, xi=xi, theta=batch_data_slice["prior_samples"], batch_data=batch_data_slice)

                y_outcomes.append(y)
                xi_designs.append(xi_untransformed_tensor)

            xi_prev = xi

        return y_outcomes


    def eval(self, n_trace=2):
        self.design_net.eval()
        output = []
        for i in range(n_trace):
            trace = pyro.poutine.trace(self.model).get_trace()
            true_theta = trace.nodes["indices"]["value"].item()

            run_xis, run_ys = [], []

            for t in range(self.T):
                xi = trace.nodes[f"xi_{t + 1}_design"]["value"].item()
                run_xis.append(xi)
                y = trace.nodes[f"y_{t + 1}_obs"]["value"].item()
                run_ys.append(y)

            run_df = pd.DataFrame({"designs": run_xis, "observations": run_ys, "order": list(range(1, self.T + 1))})
            run_df["run_id"] = i + 1
            run_df["theta"] = true_theta
            output.append(run_df)

        return pd.concat(output)

    def _get_batch_data(self, indices):
        return {
            "ys": self.simdata["ys"][:, indices, :],
            "prior_samples": self.simdata["prior_samples"][indices, :],
            "ts": self.simdata["ts"],
            "dt": self.simdata["dt"],
        }

    def _transform_designs(self, xi_untransformed, xi_prev):
        xi_prop = nn.Sigmoid()(xi_untransformed)
        xi = xi_prev + xi_prop * (self.upper_bound - xi_prev)
        return xi

def run_experiment(seed, num_steps, num_inner_samples, num_outer_samples, lr, gamma, T, device, hidden_dim, encoding_dim, num_layers, arch, complete_enum, simdata, experiment_type, mlflow_experiment_name):
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
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("complete_enum", complete_enum)
    mlflow.log_param("arch", arch)
    mlflow.log_param("num_steps", num_steps)
    mlflow.log_param("num_inner_samples", num_inner_samples)
    mlflow.log_param("num_outer_samples", num_outer_samples)
    mlflow.log_param("experiment_type", experiment_type)

    if simdata is None:
        simdata = torch.load("data/seir_sde_data.pt", map_location=device)
        if simdata is None:
            raise ValueError("Simulation data (simdata) is not properly loaded or initialized.")

    if arch == "static":
        design_net = BatchDesignBaseline(T, 1).to(device)
    elif arch == "sum":
        encoder = EncoderNetwork(1, 1, hidden_dim, encoding_dim).to(device)
        emitter = EmitterNetwork(encoding_dim, hidden_dim, 1).to(device)
        design_net = SetEquivariantDesignNetwork(encoder, emitter, empty_value=torch.ones(1)).to(device)
    else:
        raise ValueError(f"Unexpected architecture specification: '{arch}'.")

    seir_model = SEIRDesignModel(design_net=design_net, simdata=simdata, T=T)

    pyro_optimizer = pyro.optim.Adam({"lr": lr, "betas": (0.9, 0.999)})

    if experiment_type in ["dad", "fixed", "variational"]:
        loss_fn = PriorContrastiveEstimationScoreGradient(num_outer_samples, num_inner_samples)
    else:
        raise ValueError(f"Unexpected experiment type: '{experiment_type}'.")

    oed = OED(optim=pyro_optimizer, loss=loss_fn)

    loss_history = []
    t = trange(1, num_steps + 1, desc="Loss: 0.000 ")
    for i in t:
        loss = oed.step(seir_model.model)
        loss = torch_item(loss)
        t.set_description("Loss: {:.3f} ".format(loss))
        loss_history.append(loss)
        if i % 50 == 0:
            mlflow.log_metric("loss", oed.evaluate_loss(seir_model.model))
        if i % 1000 == 0:
            pyro_optimizer.step()

    mlflow.log_metric("loss_diff50", np.mean(loss_history[-51:-1]) / np.mean(loss_history[0:50]) - 1)

    runs_output = seir_model.eval()
    results = {
        "design_network": design_net.cpu(),
        "seed": seed,
        "loss_history": loss_history,
        "runs_output": runs_output,
    }

    print("Storing model to MlFlow... ", end="")
    mlflow.pytorch.log_model(seir_model.cpu(), "model")
    ml_info = mlflow.active_run().info
    model_loc = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/model"
    print(f"Model stored in {model_loc}.")

    print(f"Run completed {mlflow.active_run().info.artifact_uri}.")
    print(f"The experiment-id of this run is {ml_info.experiment_id}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Adaptive Design example: SEIR Model.")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--num-inner-samples", default=100, type=int)
    parser.add_argument("--num-outer-samples", default=200, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--num-experiments", default=4, type=int)  # == T
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--encoding-dim", default=16, type=int)
    parser.add_argument("--complete-enum", default=False, type=bool)
    parser.add_argument("--num-layers", default=2, type=int, help="Number of hidden layers.")
    parser.add_argument("--arch", default="sum", type=str, help="Architecture", choices=["static", "sum"])
    parser.add_argument("--simdata-path", default="data/seir_sde_data.pt", type=str)
    parser.add_argument("--mlflow-experiment-name", default="SEIR-Experiment", type=str)
    parser.add_argument("--experiment-type", default="dad", type=str, help="Experiment Type", choices=["dad", "fixed", "variational"])
    args = parser.parse_args()

    simdata = torch.load(args.simdata_path)
    print("Loaded simulation data:")
    print(f"ts shape: {simdata['ts'].shape}")  # Time steps shape
    print(f"ys shape: {simdata['ys'].shape}")  # SEIR compartments shape
    print(f"prior_samples shape: {simdata['prior_samples'].shape}")  # Parameter samples shape
    print(f"Number of samples (num_samples): {simdata['prior_samples'].shape[0]}")

    run_experiment(
        seed=args.seed,
        num_steps=args.num_steps,
        num_inner_samples=args.num_inner_samples,
        num_outer_samples=args.num_outer_samples,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device,
        T=args.num_experiments,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
        num_layers=args.num_layers,
        arch=args.arch,
        complete_enum=args.complete_enum,
        simdata=simdata,
        experiment_type=args.experiment_type,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )
