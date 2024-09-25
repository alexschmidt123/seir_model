import os
import math
import argparse
import pandas as pd
import torch
import mlflow
from tqdm import tqdm
from oed.design import OED
from estimators.bb_mi import InfoNCE, NWJ
from experiment_tools.output_utils import get_mlflow_meta


def evaluate(
    experiment_id,
    run_id,
    n_rollout,
    num_negative_samples,
    device,
    simdata,
    mi_estimator,
):
    artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
    model_location = f"{artifact_path}/model"
    own_critic_location = f"{artifact_path}/critic"
    mi_estimator_options = {"NWJ": NWJ, "InfoNCE": InfoNCE}

    factor = 16
    n_rollout = n_rollout // factor

    with torch.no_grad():
        try:
            seir_model = mlflow.pytorch.load_model(model_location, map_location=device)
            seir_model.SIMDATA = simdata  # Ensure SIMDATA is assigned
            print(f"Assigned SIMDATA: {seir_model.SIMDATA}")  # Debugging line
            
            critic_net_own = mlflow.pytorch.load_model(own_critic_location, map_location=device)

            # Initialize mutual information estimator
            mi_own = mi_estimator_options[mi_estimator](
                model=seir_model.model,
                critic=critic_net_own,
                batch_size=factor,
                num_negative_samples=num_negative_samples,
            )

            # Compute loss multiple times
            eig_own = torch.tensor([-mi_own.loss() for _ in range(n_rollout)])
            eig_own_mean = eig_own.mean().item()
            eig_own_std = eig_own.std().item() / math.sqrt(n_rollout)

            # Store results
            res = pd.DataFrame(
                {"mean": eig_own_mean, "se": eig_own_std, "bound": "lower"},
                index=[seir_model.T],
            )
            res.to_csv(f"mlflow_outputs/seir_eval.csv")

            # Log results to MLflow
            with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
                mlflow.log_artifact(f"mlflow_outputs/seir_eval.csv", artifact_path="evaluation")
                mlflow.log_metric("eig_own_mean", eig_own_mean)

        except Exception as e:
            print(f"Error loading model or critic: {e}")
            return


def eval_experiment(experiment_id, n_rollout, num_negative_samples, device="cpu"):
    filter_string = "params.status='complete'"
    meta = get_mlflow_meta(experiment_id=experiment_id, filter_string=filter_string)

    # Load SIMDATA
    try:
        SIMDATA = torch.load("data/seir_sde_data_test.pt", map_location=device)
        print("SIMDATA loaded successfully.")
    except Exception as e:
        print(f"Error loading SIMDATA: {e}")
        return

    # Filter for runs that haven't been evaluated
    meta = [m for m in meta if "eig_own_mean" not in m.data.metrics.keys()]
    experiment_run_ids = [run.info.run_id for run in meta]
    
    if not experiment_run_ids:
        print("No runs to evaluate.")
        return

    print(f"Evaluating runs: {experiment_run_ids}")
    for i, run_id in enumerate(experiment_run_ids):
        print(f"Evaluating run {i+1} out of {len(experiment_run_ids)} runs... {run_id}")
        evaluate(
            experiment_id=experiment_id,
            run_id=run_id,
            n_rollout=n_rollout,
            num_negative_samples=num_negative_samples,
            device=device,
            simdata=SIMDATA,
            mi_estimator=meta[i].data.params["mi_estimator"],
        )
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implicit Deep Adaptive Design: evaluate SEIR model")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--n-rollout", default=2048 * 2, type=int)
    parser.add_argument("--num-negative-samples", default=10000, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    # Load SIMDATA
    try:
        SIMDATA = torch.load("data/seir_sde_data.pt", map_location=args.device)
        if SIMDATA is None:
            raise ValueError("SIMDATA is not properly initialized or loaded.")
        print("SIMDATA loaded successfully.")
    except Exception as e:
        print(f"Failed to load SIMDATA: {e}")
        exit(1)

    # Compute validation scores
    eval_experiment(
        experiment_id=args.experiment_id,
        n_rollout=args.n_rollout,
        num_negative_samples=args.num_negative_samples,
        device=args.device,
    )
