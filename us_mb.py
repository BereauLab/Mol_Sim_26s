import numpy as np
import matplotlib.pyplot as plt
import numba

# Parameters for the MB potential
A_par     = np.array([-200.0, -100.0, -170.0,  15.0])
alpha_par = np.array([  -1.0,   -1.0,   -6.5,   0.7])
beta_par  = np.array([   0.0,    0.0,   11.0,   0.6])
gamma_par = np.array([ -10.0,  -10.0,   -6.5,   0.7])
a_par     = np.array([   1.0,    0.0,   -0.5,  -1.0])
b_par     = np.array([   0.0,    0.5,    1.5,   1.0])

def potential(x, y):
    """Müller-Brown potential"""
    dx = x - a_par
    dy = y - b_par
    return np.sum(A_par * np.exp(alpha_par*dx**2 + beta_par*dx*dy + gamma_par*dy**2))

def gradient(x, y):
    """Force F = -dU/dr."""
    dx = x - a_par
    dy = y - b_par
    uk = A_par * np.exp(alpha_par*dx**2 + beta_par*dx*dy + gamma_par*dy**2)
    dU_dx = np.sum(uk * (2*alpha_par*dx + beta_par*dy))
    dU_dy = np.sum(uk * (beta_par*dx + 2*gamma_par*dy))
    return np.array([dU_dx, dU_dy])


@numba.njit
def langevin(
    numberOfCycles: int,
    temperature: float = 1.0,
    center: np.ndarray = None,
    k_spring_x: float = 1000.0,
    k_spring_y: float = 1000.0,
    bias_x: bool = False,
    bias_y: bool = False,
):
    """
    Langevin dynamics with conditional biasing potential for a single center.

    Parameters:
    - numberOfCycles (int): Number of steps.
    - temperature (float): Temperature.
    - center (numpy.ndarray): Biasing center with shape (2,).
    - k_spring_x (float): Spring constant for the bias along the x-axis.
    - k_spring_y (float): Spring constant for the bias along the y-axis.
    - bias_x (bool): Whether to apply bias along the x-axis.
    - bias_y (bool): Whether to apply bias along the y-axis.

    Returns:
    - positions (numpy.ndarray): Array of shape (n, 2) with positions.
    - bias_energies (numpy.ndarray): Array of shape (n, num_biased_dims) with biasing energies.
    """
    if center is None:
        center = np.zeros(2)  # Default center if none provided
    if center.shape != (2,):
        raise ValueError("Center must have shape (2,) to represent a 2D system.")

    # Determine the number of biased dimensions
    num_biased_dims = int(bias_x) + int(bias_y)

    # Initialize arrays
    positions = np.zeros((numberOfCycles, 2))  # Always 2D
    bias_energies = np.zeros((numberOfCycles, num_biased_dims))  # Dimension matches number of biases
    bias_dx, bias_dy = 0.0, 0.0

    dt = 5e-4
    gamma = 1.0
    theta = np.exp(-gamma * dt)
    sigma = np.sqrt((1 - theta**2) * temperature)

    # Initialize position and velocity
    positions[0] = center
    vel = np.random.randn(2) * np.sqrt(temperature)
    potentialEnergy = potential(positions[0])
    dudx, dudy = gradient(positions[0])

    for cycle in numba.prange(1, numberOfCycles):

        # Update velocities (half-step)
        force = -np.array([dudx, dudy]) - np.array([bias_dx if bias_x else 0.0, bias_dy if bias_y else 0.0])
        vel += 0.5 * force * dt
        vel = theta * vel + sigma * np.random.randn(2)

        # Update positions
        positions[cycle] = positions[cycle - 1] + vel * dt

        # Update gradients
        potentialEnergy = potential(postions[cycle])
        dudx, dudy = gradient(positions[cycle])

        # Compute bias potential energies
        energy_idx = 0
        if bias_x:
            # start 
            bias_dx = None
            bias_energies[cycle, energy_idx] = None
            # end refactor
            energy_idx += 1
        if bias_y:
            # start refactor
            bias_dy = None
            bias_energies[cycle, energy_idx] = None
            # end refactor

        # Update velocities (half-step)
        force = -np.array([dudx, dudy]) - np.array([bias_dx if bias_x else 0.0, bias_dy if bias_y else 0.0])
        vel += 0.5 * force * dt

    return positions, bias_energies
    
    
### Compute the umbrellas

n_windows = 10
numberOfCycles = int(1e6)
temperature = 1
k_spring_x = 400
k_spring_y = 400
# Bias boolean flags
bias_x = True
bias_y = True

# Define umbrella centers

x_centers = None  # Centers along x-axis with shape (n_windows,)
y_centers = None  # Centers along y-axis with shape (n_windows,)
centers = np.column_stack((x_centers, y_centers))  # Shape: (n_windows, 2)


# Run umbrella sampling for each window
trajectories = []
bias_energies = []

for center in centers:
    print(f"Running simulation for center: {center}")
    traj, bias_energy = langevin(
        numberOfCycles=numberOfCycles,
        temperature=temperature,
        center=center,
        k_spring_x=k_spring_x,
        k_spring_y=k_spring_y,
        bias_x=bias_x,
        bias_y=bias_y,
    )
    trajectories.append(traj)
    bias_energies.append(bias_energy)

# Aggregate results
trajectories = np.array(trajectories)  # Shape: (num_windows, numberOfCycles, 2)
bias_energies = np.array(bias_energies)  # Shape: (num_windows, numberOfCycles)

# Plotting the results
fig, ax = plt.subplots()
# Here plot the MB potential

colors = plt.cm.viridis(np.linspace(0, 1, n_windows))

# Plot all trajectories in the same heatmap
for w in range(n_windows):
    positions = trajectories[w]
    # Skip the first 10,000 steps to avoid initial relaxation
    ax.scatter(positions[10000:, 0], positions[10000:, 1], s=0.1, color=colors[w], label=f"Window {w+1}")
ax.set_title("Umbrella Sampling Trajectories")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

