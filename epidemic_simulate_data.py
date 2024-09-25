import os
import argparse
import sys
import time

import torch
import torchsde

# needed for torchsde
sys.setrecursionlimit(1500)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEIR_SDE(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):
        super().__init__()
        # parameters: (beta, sigma, gamma)
        self.params = params
        self.N = population_size

    # Implement drift and diffusion together
    def f_and_g(self, t, x):
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        # Susceptible, Exposed, Infected, Recovered
        S, E, I, R = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # Rates
        beta = self.params[:, 0]
        sigma = self.params[:, 1]
        gamma = self.params[:, 2]

        # Infection pressure
        p_inf = beta * S * I / self.N
        p_inf_sqrt = torch.sqrt(p_inf + 1e-8)  # Prevent sqrt(0)

        # Exposed to Infected rate
        p_exp = sigma * E
        p_exp_sqrt = torch.sqrt(p_exp + 1e-8)  # Prevent sqrt(0)

        # Recovery rate
        p_rec = gamma * I
        p_rec_sqrt = torch.sqrt(p_rec + 1e-8)  # Prevent sqrt(0)
        
        # Debug: Print recovery term for better visibility
        # print("Recovery term sqrt(gamma * I):", p_rec_sqrt.cpu().numpy())

        # Drift term (4 compartments: S, E, I, R)
        f_term = torch.stack([-p_inf, p_inf - p_exp, p_exp - p_rec, p_rec], dim=-1)

        # Diffusion term (should match dimensions for S, E, I, R)
        g_term = torch.zeros(x.size(0), 4, 4, device=x.device)  # Initialize a zero matrix for each sample

        # g_S = [-sqrt(beta * S * I / N), 0, 0, 0]
        g_term[:, 0, 0] = -p_inf_sqrt  # S -> S
        
        # g_E = [sqrt(beta * S * I / N), -sqrt(sigma * E), 0, 0]
        g_term[:, 0, 1] = p_inf_sqrt  # S -> E
        g_term[:, 1, 1] = -p_exp_sqrt  # E -> E
        g_term[:, 1, 2] = p_exp_sqrt  # E -> I

        # g_I = [0, sqrt(sigma * E), -sqrt(gamma * I), 0]
        g_term[:, 2, 2] = -p_rec_sqrt  # I -> I
        g_term[:, 2, 3] = p_rec_sqrt   # I -> R

        # Debug: Print diffusion term for recovered compartment
        # print("g_term for recovered (R):", g_term[:, 2, 3].cpu().numpy())

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
        theta_loc = torch.tensor([0.5, 0.2, 0.1], device=device).log()
        theta_covmat = torch.eye(3, device=device) * 0.5 ** 2

    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
    params = prior.sample(torch.Size([num_samples])).exp()
    #################################

    T0, T = 0.0, 100.0  # initial and final time
    GRID = grid  # time-grid

    population_size = 500.0
    initial_infected = 2.0  # Initial number of infected individuals
    initial_exposed = 1.0  # Initial number of exposed individuals
    initial_recovered = 0.0  # No one recovered at the beginning

    # Add sanity check
    if initial_exposed + initial_infected >= population_size:
        raise ValueError("Sum of initial exposed and infected exceeds or equals population size")

    # Susceptible population
    S_0 = population_size - initial_exposed - initial_infected

    # Initialize the SEIR compartments with distinct initial values
    y0 = torch.tensor(
        num_samples * [[S_0, initial_exposed, initial_infected, initial_recovered]],
        device=device,
    )  # starting point (shape: [num_samples, 4])

    # Debugging: Print the initial values to ensure correctness
    print(f"Initial conditions (S_0, E_0, I_0, R_0): {S_0}, {initial_exposed}, {initial_infected}, {initial_recovered}")

    ts = torch.linspace(T0, T, GRID, device=device)  # time grid

    sde = SEIR_SDE(
        population_size=torch.tensor(population_size, device=device), params=params,
    ).to(device)

    start_time = time.time()
    
    # Solve the SDE
    ys = torchsde.sdeint(sde, y0, ts)  # solved sde
    end_time = time.time()
    print("Simulation Time: %s seconds" % (end_time - start_time))

    # Add your print statement here to check the recovered population over time
    print("Recovered population over time:", ys[:, :, 3].cpu().numpy())

    save_dict = dict()
    idx_good = torch.where(ys[:, :, 2].mean(0) >= 1)[0]  # Infected compartment

    save_dict["prior_samples"] = params[idx_good].cpu()
    save_dict["ts"] = ts.cpu()
    save_dict["dt"] = (ts[1] - ts[0]).cpu()  # delta-t (time grid)
    save_dict["ys"] = ys[:, idx_good, :].cpu()  # Full SEIR compartments

    # grads can be calculated in backward pass (saves space)
    if savegrad:
        grads = (ys[2:, ...] - ys[:-2, ...]) / (2 * save_dict["dt"])
        save_dict["grads"] = grads[:, idx_good, :].cpu()

    # Meta data
    save_dict["N"] = population_size
    save_dict["I0"] = initial_infected
    save_dict["E0"] = initial_exposed
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]

    if save:
        print("Saving data.", end=" ")
        torch.save(save_dict, f"data/{filename}")

    print("DONE.")
    return save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epidemic: solve SEIR equations")
    parser.add_argument("--num-samples", default=10, type=int, help="Number of samples to generate")
    parser.add_argument("--device", default="cpu", type=str, help="Device to run the simulations on (cpu/cuda)")

    # Ensure the output directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    args = parser.parse_args()

    # Generate initial training data
    print("Generating initial training data...")
    solve_seir_sdes(
        num_samples=args.num_samples,
        device=args.device,
        grid=10000,
        save=True,
        savegrad=False,
    )

    # Generate initial test data
    print("Generating initial test data...")
    test_data = []
    for i in range(3):  # Generate data for 3 test datasets
        dict_i = solve_seir_sdes(
            num_samples=args.num_samples,
            device=args.device,
            grid=10000,
            save=False,
            savegrad=False,
        )
        test_data.append(dict_i)

    # Save combined test data
    save_dict = {
        "prior_samples": torch.cat([d["prior_samples"] for d in test_data]),
        "ys": torch.cat([d["ys"] for d in test_data], dim=1),
        "dt": test_data[0]["dt"],
        "ts": test_data[0]["ts"],
        "N": test_data[0]["N"],
        "I0": test_data[0]["I0"],
    }
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]
    torch.save(save_dict, "data/seir_sde_data_test.pt")
    print("Test data saved successfully.")
