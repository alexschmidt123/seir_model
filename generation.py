import sys
import torch
import torchsde
import matplotlib.pyplot as plt

# Increase recursion limit
sys.setrecursionlimit(2000)

class SEIR_SDE(torch.nn.Module):
    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):
        super().__init__()
        self.params = params
        self.N = population_size

    def f_and_g(self, t, x):
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        S, E, I = x[:, 0], x[:, 1], x[:, 2]
        beta, alpha, gamma = self.params[:, 0], self.params[:, 1], self.params[:, 2]

        # Drift term
        p_inf = beta * S * I / self.N
        p_exp = alpha * E
        p_rec = gamma * I

        f_term = torch.stack([-p_inf, p_inf - p_exp, p_exp - p_rec], dim=-1)

        # Diffusion term
        p_inf_sqrt = torch.sqrt(p_inf)
        p_exp_sqrt = torch.sqrt(p_exp)
        p_rec_sqrt = torch.sqrt(p_rec)

        g_term = torch.stack(
            [
                torch.stack([-p_inf_sqrt, torch.zeros_like(p_inf_sqrt), torch.zeros_like(p_inf_sqrt)], dim=-1),
                torch.stack([p_inf_sqrt, -p_exp_sqrt, torch.zeros_like(p_exp_sqrt)], dim=-1),
                torch.stack([torch.zeros_like(p_rec_sqrt), p_exp_sqrt, -p_rec_sqrt], dim=-1),
            ],
            dim=-1,
        )

        return f_term, g_term

def solve_seir_sdes(num_samples=3, grid=500, device='cpu'):
    # Define priors for (beta, alpha, gamma)
    theta_loc = torch.tensor([0.5, 0.2, 0.1], device=device).log()
    theta_covmat = torch.eye(3, device=device) * 0.5 ** 2

    prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
    params = prior.sample(torch.Size([num_samples])).exp()

    T0, T = 0.0, 100.0
    GRID = grid
    population_size = 1000.0
    initial_exposed = 10.0
    initial_infected = 5.0

    # Initial conditions for S, E, I
    y0 = torch.tensor(
        num_samples * [[population_size - initial_exposed - initial_infected, initial_exposed, initial_infected]],
        device=device,
    )

    ts = torch.linspace(T0, T, GRID, device=device)
    sde = SEIR_SDE(params=params, population_size=population_size).to(device)

    ys = torchsde.sdeint(sde, y0, ts)
    return ts.cpu(), ys.cpu()

# Generate the SEIR model data
ts, ys = solve_seir_sdes(num_samples=3, grid=500, device='cpu')

# Save the data to a .pt file
output_file = 'seir_sde_data.pt'
torch.save({'time': ts, 'data': ys}, output_file)
print(f"Data saved to {output_file}")

# Load and check data for visualization
time_points = ts.numpy()
seir_data = ys.numpy()

print(f"Shape of time_points: {time_points.shape}")  # Should be (500,)
print(f"Shape of seir_data: {seir_data.shape}")  # Should be (500, 3, 3)


