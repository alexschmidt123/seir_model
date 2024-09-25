import argparse
import torch
import matplotlib.pyplot as plt

def seir_simulation(beta, sigma, gamma, initial_values, N, dt, T):
    # Calculate the number of steps based on total time T and time step dt
    num_steps = int(T / dt)

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
        dS = (-beta * S * I / N) * dt
        dE = (beta * S * I / N - sigma * E) * dt
        dI = (sigma * E - gamma * I) * dt
        dR = (gamma * I) * dt

        # Ensure the shapes are consistent
        dS = dS.unsqueeze(-1)
        dE = dE.unsqueeze(-1)
        dI = dI.unsqueeze(-1)
        dR = dR.unsqueeze(-1)

        # Batch size handling (single batch in this case)
        batch_size = S.shape[0]

        # Initialize the diffusion terms for each component
        g_S = torch.zeros(batch_size, 4)
        g_S[:, 0] = -torch.sqrt(beta * S * I / N).squeeze(-1)  # g_SS component

        g_E = torch.zeros(batch_size, 4)
        g_E[:, 0] = torch.sqrt(beta * S * I / N).squeeze(-1)  # g_ES component
        g_E[:, 1] = -torch.sqrt(sigma * E).squeeze(-1)        # g_EE component

        g_I = torch.zeros(batch_size, 4)
        g_I[:, 1] = torch.sqrt(sigma * E).squeeze(-1)         # g_IE component
        g_I[:, 2] = -torch.sqrt(gamma * I).squeeze(-1)        # g_II component

        g_R = torch.zeros(batch_size, 4)
        g_R[:, 2] = torch.sqrt(gamma * I).squeeze(-1)         # g_RI component

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

def plot_seir(S_values, E_values, I_values, R_values, dt):
    time_points = [i * dt for i in range(len(S_values))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, S_values, label="Susceptible (S)", color="blue")
    plt.plot(time_points, E_values, label="Exposed (E)", color="orange")
    plt.plot(time_points, I_values, label="Infected (I)", color="red")
    plt.plot(time_points, R_values, label="Recovered (R)", color="green")
    
    plt.title("SEIR Model Simulation with Stochastic Elements")
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEIR Model Simulation")
    parser.add_argument("--beta", default=0.3, type=float, help="Transmission rate")
    parser.add_argument("--sigma", default=0.1, type=float, help="Rate from E to I")
    parser.add_argument("--gamma", default=0.1, type=float, help="Recovery rate")
    parser.add_argument("--initial_values", nargs=4, type=int, default=[999, 0, 1, 0], help="Initial values for S, E, I, R")
    parser.add_argument("--N", default=1000, type=int, help="Total population size")
    parser.add_argument("--dt", default=1.0, type=float, help="Time step size in days")
    parser.add_argument("--T", default=100.0, type=float, help="Total time of the simulation in days")

    args = parser.parse_args()

    S_values, E_values, I_values, R_values = seir_simulation(
        beta=args.beta,
        sigma=args.sigma,
        gamma=args.gamma,
        initial_values=args.initial_values,
        N=args.N,
        dt=args.dt,
        T=args.T
    )

    # Plot the results
    plot_seir(S_values, E_values, I_values, R_values, dt=args.dt)
