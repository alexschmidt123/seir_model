import os
import pickle
import argparse
import math

import pandas as pd
from tqdm import trange
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import mlflow
import mlflow.pytorch
from epidemic_SEIR_simulate_data import solve_seir_sdes
from oed.primitives import observation_sample, latent_sample, compute_design
from experiment_tools.pyro_tools import auto_seed
from oed.design import OED
from estimators.bb_mi import InfoNCE, NWJ
from neural.modules import Mlp
from neural.aggregators import (
    PermutationInvariantImplicitDAD,
    LSTMImplicitDAD,
    ConcatImplicitDAD,
)
from neural.baselines import (
    ConstantBatchBaseline,
    BatchDesignBaseline,
    RandomDesignBaseline,
)
from neural.critics import CriticDotProd, CriticBA

mi_estimator_options = {"NWJ": NWJ, "InfoNCE": InfoNCE}


class SEIR_SDE_Simulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, batch_data, device):
        with torch.no_grad():
            nearest = torch.min(
                torch.abs(inputs.reshape(-1, 1) - batch_data["ts"][1:-1]), axis=1
            ).indices
        # Extract number of infected from data
        y = batch_data["ys"][1:-1][nearest, range(nearest.shape[0])].reshape(-1, 1)
        ctx.save_for_backward(inputs)
        ctx.device = device
        ctx.nearest = nearest
        ctx.grads = (batch_data["ys"][2:, :] - batch_data["ys"][:-2, :]) / (
            2 * batch_data["dt"]
        )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        device = ctx.device
        nearest = ctx.nearest
        grads = ctx.grads
        y_grads = grads[nearest, range(nearest.shape[0])].T
        y_grads = y_grads.reshape(-1, 1)
        identity = torch.eye(1, device=device, dtype=torch.float32).reshape(1, 1, 1)
        Jac = torch.mul(identity.repeat(len(y_grads), 1, 1), y_grads[:, None])
        grad_input = Jac.matmul(grad_output[:, :, None]).reshape(-1, 1)
        return grad_input, None, None


class Epidemic(nn.Module):
    def __init__(
        self,
        design_net,
        T,
        design_transform="iid",
        simdata=None,
        lower_bound=torch.tensor(1e-2),
        upper_bound=torch.tensor(100.0 - 1e-2),
    ):
        super(Epidemic, self).__init__()
        self.p = 1  # Only tracking infected compartment 'I'
        self.design_net = design_net
        self.T = T
        self.SIMDATA = simdata
        loc = torch.tensor([0.5, 0.2, 0.1]).log().to(simdata["ys"].device)
        covmat = torch.eye(3).to(simdata["ys"].device) * 0.5 ** 2
        self._prior_on_log_theta = torch.distributions.MultivariateNormal(loc, covmat)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fc_transform = nn.Linear(32, 128)
        if design_transform == "ts":
            self.transform_designs = self._transform_designs_increasing
        elif design_transform == "iid":
            self.transform_designs = self._transform_designs_independent
        else:
            raise ValueError

    def simulator(self, xi, theta, batch_data):
        sim_seir = SEIR_SDE_Simulator.apply
        y = sim_seir(xi, batch_data, theta.device)
        return y

    def _get_batch_data(self, indices):
        if self.SIMDATA is None:
            raise ValueError("SIMDATA is not initialized.")
        batch_data = {
            "ys": self.SIMDATA["ys"][:, indices],  # Correct indexing for 2D infected data
            "prior_samples": self.SIMDATA["prior_samples"][indices, :],
            "ts": self.SIMDATA["ts"],
            "dt": self.SIMDATA["dt"],
        }
        print(f"Batch data 'ys' shape: {batch_data['ys'].shape}")
        print(f"Prior samples shape: {batch_data['prior_samples'].shape}")
        return batch_data

    def _transform_designs_increasing(self, xi_untransformed, xi_prev):
        xi_prop = nn.Sigmoid()(xi_untransformed)
        xi = xi_prev + xi_prop * (self.upper_bound - xi_prev)
        return xi

    def _transform_designs_independent(self, xi_untransformed, xi_prev=None):
        xi_prop = nn.Sigmoid()(xi_untransformed)
        xi = self.lower_bound + xi_prop * (self.upper_bound - self.lower_bound)
        return xi

    def model(self):
        if hasattr(self.design_net, "parameters"):
            pyro.module("design_net", self.design_net)
        device = self.SIMDATA["prior_samples"].device
        prior_on_index = dist.Categorical(
            torch.ones(self.SIMDATA["num_samples"], device=device)
        )
        indices = pyro.sample("indices", prior_on_index)
        batch_data = self._get_batch_data(indices)
        def get_theta():
            return batch_data["prior_samples"]
        theta = latent_sample("theta", get_theta)
        y_outcomes = []
        xi_designs = []
        xi_prev = self.lower_bound
        for t in range(self.T):
            print(f"Sampling at step {t+1}")
            xi_untransformed = compute_design(
                f"xi{t + 1}", self.design_net.lazy(*zip(xi_designs, y_outcomes))
            )
            print(f"xi_untransformed shape: {xi_untransformed.shape}")
            xi = self.transform_designs(
                xi_untransformed=xi_untransformed.squeeze(1), xi_prev=xi_prev,
            )
            print(f"Transformed xi shape: {xi.shape}")
            y = observation_sample(
                f"y{t + 1}", self.simulator, xi=xi, theta=theta, batch_data=batch_data
            )
            y_outcomes.append(y)
            xi_designs.append(xi_untransformed)
            xi_prev = xi
        del batch_data
        return theta, xi_designs, y_outcomes

    def forward(self, indices):
        self.design_net.eval()
        def conditioned_model():
            with pyro.plate_stack("expand_theta_test", [indices.shape[0]]):
                return pyro.condition(self.model, data={"indices": indices})()
        with torch.no_grad():
            theta, designs, observations = conditioned_model()
        return theta, designs, observations


def train_model(
    num_steps,
    batch_size,
    num_negative_samples,
    seed,
    lr,
    lr_critic,
    gamma,
    device,
    T,
    hidden_dim,
    encoding_dim,
    critic_arch,
    mi_estimator,
    mlflow_experiment_name,
    design_arch,
    design_transform,
):
    pyro.clear_param_store()
    mlflow.set_experiment(mlflow_experiment_name)
    seed = auto_seed(seed)
    n = 2 # Only tracking infected compartment 'I'
    design_dim = (n, 1)
    latent_dim = 3
    observation_dim = n
    if lr_critic is None:
        lr_critic = lr
    history_encoder = Mlp(
        input_dim=[*design_dim, observation_dim],
        hidden_dim=[8, 64, hidden_dim],
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="policy_history_encoder",
    )
    design_emitter = Mlp(
        input_dim=encoding_dim * max((T - 1), 1),
        hidden_dim=encoding_dim // 2,
        output_dim=design_dim,
        activation=nn.ReLU(),
        name="policy_design_emitter",
    )
    design_net = LSTMImplicitDAD(
        history_encoder, design_emitter, empty_value=torch.zeros(design_dim, device=device), num_hidden_layers=2
    ).to(device)
    critic_latent_encoder = Mlp(
        input_dim=latent_dim,
        hidden_dim=[8, 64, hidden_dim],
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="critic_latent_encoder",
    )
    critic_design_obs_encoder = Mlp(
        input_dim=[*design_dim, observation_dim],
        hidden_dim=[8, 64, hidden_dim],
        output_dim=encoding_dim,
        name="critic_design_obs_encoder",
    )
    critic_head = Mlp(
        input_dim=encoding_dim * T,
        hidden_dim=encoding_dim // 2,
        output_dim=encoding_dim,
        activation=nn.ReLU(),
        name="critic_head",
    )
    critic_net = CriticDotProd(
        history_encoder_network=LSTMImplicitDAD(
            critic_design_obs_encoder,
            critic_head,
            empty_value=torch.ones(n, latent_dim, device=device),
            num_hidden_layers=2,
        ),
        latent_encoder_network=critic_latent_encoder,
    ).to(device)
    SIMDATA = torch.load("data/seir_sde_data.pt", map_location=device)
    if SIMDATA is None:
        raise ValueError("SIMDATA is not properly initialized or loaded.")
    mlflow.log_param("dt", SIMDATA["dt"].cpu().item())
    test_theta = torch.tensor([[0.60, 0.15, 0.05]], dtype=torch.float, device=device)
    epidemic = Epidemic(
        design_net=design_net, T=T, design_transform=design_transform, simdata=SIMDATA,
    )
    scheduler = pyro.optim.ExponentialLR(
        {"optimizer": torch.optim.Adam, "optim_args": lambda m, _: {"lr": lr_critic if m == "critic_net" else lr}, "gamma": gamma}
    )
    oed = OED(optim=scheduler, loss=mi_estimator_options[mi_estimator](
        model=epidemic.model, critic=critic_net, batch_size=batch_size, num_negative_samples=num_negative_samples
    ))
    outputs_history = []
    for i in trange(1, num_steps + 1, desc="Loss: 0.000 "):
        epidemic.train()
        loss = oed.step()
        if (i - 1) % 200 == 0:
            df, latents = epidemic.eval(theta=test_theta, verbose=False)
            df["step"] = i
            outputs_history.append(df)
        if i % 1000 == 0 and i % 1000 == 0:
            print("Resampling SIMDATA")
            epidemic._remove_data()
            del SIMDATA
            SIMDATA = solve_seir_sdes(num_samples=120000, device=device, grid=10000)
            epidemic.SIMDATA = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in SIMDATA.items()}
    pd.concat(outputs_history).reset_index().to_csv(f"mlflow_outputs/designs_hist.csv")
    mlflow.log_artifact(f"mlflow_outputs/designs_hist.csv", artifact_path="designs")
    mlflow.pytorch.log_model(epidemic.cpu(), "model")
    mlflow.pytorch.log_model(critic_net.cpu(), "critic")
    return epidemic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iDAD: SDE-Based SEIR Model")
    parser.add_argument("--num-steps", default=1000, type=int)
    parser.add_argument("--num-batch-samples", default=512, type=int)
    parser.add_argument("--num-negative-samples", default=511, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--lr-critic", default=None, type=float)
    parser.add_argument("--gamma", default=0.96, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--num-experiments", default=5, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--encoding-dim", default=32, type=int)
    parser.add_argument("--mi-estimator", default="InfoNCE", type=str)
    parser.add_argument("--design-transform", default="ts", type=str, choices=["ts", "iid"])
    parser.add_argument("--critic-arch", default="lstm", type=str, choices=["cat", "sum", "lstm"])
    parser.add_argument("--design-arch", default="lstm", type=str, choices=["sum", "lstm", "static", "equal_interval", "random"])
    parser.add_argument("--mlflow-experiment-name", default="epidemic", type=str)
    args = parser.parse_args()
    train_model(
        num_steps=args.num_steps, batch_size=args.num_batch_samples, num_negative_samples=args.num_negative_samples,
        seed=args.seed, lr=args.lr, lr_critic=args.lr_critic, gamma=args.gamma, device=args.device,
        T=args.num_experiments, hidden_dim=args.hidden_dim, encoding_dim=args.encoding_dim,
        critic_arch=args.critic_arch, mi_estimator=args.mi_estimator, mlflow_experiment_name=args.mlflow_experiment_name,
        design_arch=args.design_arch, design_transform=args.design_transform
    )
ep