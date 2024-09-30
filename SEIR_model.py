import torch
import argparse
import mlflow
import mlflow.pytorch
import pyro
import pyro.distributions as dist
import torch.nn as nn
from oed.design import OED
from oed.primitives import compute_design, observation_sample
from estimators.bb_mi import InfoNCE
from tqdm import trange

# Encoder Network: Processes the input designs and observations
class EncoderNetwork(nn.Module):
    def __init__(self, design_dim, observation_dim, hidden_dim, encoding_dim, num_layers=2):
        super(EncoderNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(design_dim + observation_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, encoding_dim)
        )

    def forward(self, xi, y):
        inputs = torch.cat([xi, y], dim=-1)
        return self.layers(inputs)

# Emitter Network: Generates the designs based on encoded input
class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(EmitterNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, r):
        return self.layers(r)

# SEIR Model Simulation with I compartment
class SEIRSDEModel(nn.Module):
    def __init__(self, encoder, emitter, beta, sigma, gamma, initial_values, N, T, dt):
        super(SEIRSDEModel, self).__init__()
        self.encoder = encoder
        self.emitter = emitter
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.initial_values = initial_values
        self.N = N
        self.T = T
        self.dt = dt

    def model(self):
        theta = pyro.sample("theta", dist.Normal(torch.tensor([self.beta, self.sigma, self.gamma]), 0.1).to_event(1))
        beta, sigma, gamma = theta
        xi_designs, y_outcomes = [], []

        for t in range(self.T):
            xi = compute_design(f"xi{t + 1}", self.emitter(self.encoder.lazy(*zip(xi_designs, y_outcomes))))
            y = self.simulate_seir(beta, sigma, gamma, xi)
            xi_designs.append(xi)
            y_outcomes.append(y)
        return theta, xi_designs, y_outcomes

    def simulate_seir(self, beta, sigma, gamma, xi):
        S, E, I, R = self.initial_values
        for _ in range(int(self.T / self.dt)):
            dS = (-beta * S * I / self.N) * self.dt
            dE = (beta * S * I / self.N - sigma * E) * self.dt
            dI = (sigma * E - gamma * I) * self.dt
            dR = (gamma * I) * self.dt
            S, E, I, R = S + dS, E + dE, I + dI, R + dR
        return torch.tensor(I)

# Function to load data from .pt files
# Function to load data from .pt files and move to the specified device
def load_data(file_path, device):
    data = torch.load(file_path)
    # Ensure all tensor values in the dictionary are moved to the correct device
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    return data


# Train function for DAD model
def train_dad_model(model, data, optimizer, num_epochs, device):
    pyro.clear_param_store()
    optimizer.zero_grad()
    for epoch in trange(num_epochs, desc="Training DAD Model"):
        loss = 0
        for batch in data:
            theta, xi_designs, y_outcomes = model()
            loss = ((y_outcomes - batch) ** 2).sum()
            loss.backward()
        optimizer.step()

# Main experiment with DAD and Fixed Design
def experiment(file_path, num_steps, batch_size, learning_rate, T, dt, device):
    # Load SEIR data and move tensors to the specified device
    data = load_data(file_path, device)
    
    # Encoder and Emitter Networks
    encoder = EncoderNetwork(design_dim=1, observation_dim=1, hidden_dim=32, encoding_dim=16).to(device)
    emitter = EmitterNetwork(input_dim=16, hidden_dim=32, output_dim=1).to(device)

    # Define SEIR model with the Encoder and Emitter
    seir_model_dad = SEIRSDEModel(encoder=encoder, emitter=emitter, beta=0.3, sigma=0.1, gamma=0.1, 
                                  initial_values=(99, 1, 0, 0), N=100, T=T, dt=dt).to(device)
    
    # Optimizer
    optimizer_dad = torch.optim.Adam(seir_model_dad.parameters(), lr=learning_rate)

    # Train the DAD model
    print("Training DAD Model...")
    train_dad_model(seir_model_dad, data, optimizer_dad, num_epochs=num_steps, device=device)

    # Fixed Design: Use same encoder/emitter but without training
    print("Running Fixed Design Experiment...")
    with torch.no_grad():
        seir_model_fixed = SEIRSDEModel(encoder=encoder, emitter=emitter, beta=0.3, sigma=0.1, gamma=0.1, 
                                        initial_values=(99, 1, 0, 0), N=100, T=T, dt=dt).to(device)
        seir_model_fixed.model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAD and Fixed Design Experiment on SEIR Model")
    parser.add_argument("--file-path", type=str, default="data/seir_sde_data.pt")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--T", type=int, default=4, help="Number of experiments")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the experiment on")

    args = parser.parse_args()
    
    # Run Experiment
    experiment(file_path=args.file_path, num_steps=args.num_steps, batch_size=args.batch_size, 
               learning_rate=args.learning_rate, T=args.T, dt=args.dt, device=args.device)
