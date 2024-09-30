import os
import argparse
import sys
import time
import torch
import torchsde

# needed for torchsde
sys.setrecursionlimit(1500)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SEIR_SDE(torch.nn.Module):
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):
        super().__init__()
        # parameters: (beta, sigma, gamma)
        self.params = params
        self.N = population_size

    def f_and_g(self, t, x):
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        S, E, I = x.T  # Note: We stop tracking R explicitly
        beta, sigma, gamma = self.params.T

        # Drift (f_term)
        f_term = torch.stack([
            -beta * S * I / self.N,                    # dS/dt
            beta * S * I / self.N - sigma * E,         # dE/dt
            sigma * E - gamma * I                      # dI/dt
        ], dim=-1)

        # Diffusion (g_term)
        batch_size = S.shape[0]

        g_S = torch.zeros(batch_size, 3)
        g_S[:, 0] = -torch.sqrt(beta * S * I / self.N).squeeze(-1)

        g_E = torch.zeros(batch_size, 3)
        g_E[:, 0] = torch.sqrt(beta * S * I / self.N).squeeze(-1)
        g_E[:, 1] = -torch.sqrt(sigma * E).squeeze(-1)

        g_I = torch.zeros(batch_size, 3)
        g_I[:, 1] = torch.sqrt(sigma * E).squeeze(-1)
        g_I[:, 2] = -torch.sqrt(gamma * I).squeeze(-1)

        # Stack all components together
        g_term = torch.stack([g_S, g_E, g_I], dim=-1)  # (batch_size, 3, 3)

        return f_term, g_term


def solve_seir_sdes(
    num_samples,
    device,
    grid=10000,
    savegrad=False,
    save=False,
    filename="seir_sde_data.pt",
    theta_loc=None,
    theta_covmat=None,
):
    ####### Change priors here ######
    if theta_loc is None or theta_covmat is None:
        theta_loc = torch.tensor([0.3, 0.1, 0.1], device=device).log()
        theta_covmat = torch.eye(3, device=device) * 0.5 ** 2

    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
    params = prior.sample(torch.Size([num_samples])).exp()
    #################################

    T0, T = 0.0, 100.0  # initial and final time
    GRID = grid  # time-grid

    population_size = 500.0
    initial_exposed = 1.0  # initial number of exposed individuals
    initial_infected = 2.0  # initial number of infected individuals

    ## [S, E, I] (excluding R, as it will be calculated)
    y0 = torch.tensor(
        num_samples * [[population_size - initial_exposed - initial_infected, initial_exposed, initial_infected]],
        device=device,
    )  # starting point
    ts = torch.linspace(T0, T, GRID, device=device)  # time grid

    sde = SEIR_SDE(
        population_size=torch.tensor(population_size, device=device), params=params,
    ).to(device)

    start_time = time.time()
    ys = torchsde.sdeint(sde, y0, ts)  # solved sde
    end_time = time.time()
    print("Simulation Time: %s seconds" % (end_time - start_time))

    save_dict = dict()
    idx_good = torch.where(ys[:, :, 2].mean(0) >= 1)[0]  # Filter based on infected individuals (Index 2 for I)

    save_dict["prior_samples"] = params[idx_good].cpu()
    save_dict["ts"] = ts.cpu()
    save_dict["dt"] = (ts[1] - ts[0]).cpu()  # delta-t (time grid)

    # Save only I (infected) component at index 2
    save_dict["ys"] = ys[:, idx_good, 2].cpu()  # Only infected individuals

    if savegrad:
        # central difference for gradient calculation
        grads = (ys[2:, ...] - ys[:-2, ...]) / (2 * save_dict["dt"])
        save_dict["grads"] = grads[:, idx_good, 2].cpu()  # Gradients only for infected compartment

    # meta data
    save_dict["N"] = population_size
    save_dict["E0"] = initial_exposed
    save_dict["I0"] = initial_infected
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]

    if save:
        print("Saving data.", end=" ")
        torch.save(save_dict, f"data/{filename}")

    print("DONE.")
    return save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epidemic: solve SEIR equations")
    parser.add_argument("--num-samples", default=1000, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    if not os.path.exists("data"):
        os.makedirs("data")

    args = parser.parse_args()

    print("Generating initial training data...")
    solve_seir_sdes(
        num_samples=args.num_samples,
        device=args.device,
        grid=10000,
        save=True,
        savegrad=False,
    )
    print("Generating initial test data...")
    ####### generate a big test dataset
    test_data = []
    for i in range(3):
        dict_i = solve_seir_sdes(
            num_samples=args.num_samples,
            device=args.device,
            grid=10000,
            save=False,
            savegrad=False,
        )
        test_data.append(dict_i)

    save_dict = {
        "prior_samples": torch.cat([d["prior_samples"] for d in test_data]),
        "ys": torch.cat([d["ys"] for d in test_data], dim=1),
        "dt": test_data[0]["dt"],
        "ts": test_data[0]["ts"],
        "N": test_data[0]["N"],
        "E0": test_data[0]["E0"],
        "I0": test_data[0]["I0"],
    }
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]
    torch.save(save_dict, "data/seir_sde_data_test.pt")
