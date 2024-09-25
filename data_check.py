import torch
import matplotlib.pyplot as plt
import os

# Load the saved data
data = torch.load('data/seir_sde_data.pt')

# Extract relevant data
ts = data['ts']  # Time grid
ys = data['ys']  # SEIR compartments for all samples (shape: [time, samples, compartments])

# Get the total number of samples
num_samples = ys.shape[1]

# Create the "Generated_plots" directory if it doesn't exist
output_dir = 'Generated_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over all samples and save individual plots
for i in range(num_samples):
    # Extract SEIR compartments for the current sample
    S = ys[:, i, 0]  # Susceptible compartment
    E = ys[:, i, 1]  # Exposed compartment
    I = ys[:, i, 2]  # Infected compartment
    R = ys[:, i, 3]  # Recovered compartment
    
    # Create a new plot for the current sample
    plt.figure(figsize=(8, 6))
    plt.plot(ts, S, label='Susceptible', color='darkgoldenrod')  # Dark yellow for Susceptible
    plt.plot(ts, E, label='Exposed', color='blue')               # Blue for Exposed
    plt.plot(ts, I, label='Infected', color='red')               # Red for Infected
    plt.plot(ts, R, label='Recovered', color='green')            # Green for Recovered
    
    # Add title, labels, legend, and grid
    plt.title(f'Sample {i + 1}')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the plot as a .jpg file in the "Generated_plots" folder
    output_path = os.path.join(output_dir, f'Sample_{i + 1}.jpg')
    plt.savefig(output_path)
    
    # Close the plot to free up memory
    plt.close()

# Print the number of plots saved
print(f"{num_samples} plots have been saved in the '{output_dir}' folder.")
