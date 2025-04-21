import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange
import multiprocessing

# Parameters
n = 60  # nxn subdivision - increased for better visualization
cRed = 1.0  # Rate Constants
cBlue = 1.0  # Rate Constants
dx = 1.0/n
dy = 1.0/n
dt = 0.1*dx*dy  # Timestep based on grid size

# Satisfaction thresholds
pBlueSatisfied = 0.45  # 50% Satisfaction Level
pRedSatisfied = 0.45  # 50% Satisfaction Level

# Randomize initial conditions flag
R = False  # False means Sinusoidal initial conditions

#Initial conditiosn
@njit
def b0(x, y):
    x_coord = x * dx
    y_coord = y * dy
    if R:
        return np.random.uniform(0, 0.4)
    return 0.2*np.sin(10.9*x_coord)*np.sin(9*y_coord) + 0.2

@njit
def r0(x, y):
    x_coord = x * dx
    y_coord = y * dy
    if R:
        return np.random.uniform(0, 0.4)
    return 0.2*np.cos(10*x_coord)*np.cos(15*y_coord) + 0.2

#Calculate percentage of red and blue in the neighborhood
@njit
def calculate_percentage(red_values, blue_values):
    total = red_values + blue_values
    if total > 0:
        return red_values / total, blue_values / total
    return 0.0, 0.0

#Calculate sum of red and blue in the neighborhood in a 3x3 square around the cell considered. 
@njit
def get_neighborhood_sums(grid, i, j, n):
    total = 0.0
    for di in range(-1, 2):
        ni = i + di
        if ni <= 0 or ni >= n:
            continue
        for dj in range(-1, 2):
            nj = j + dj
            if nj <= 0 or nj >= n:
                continue
            total += grid[ni, nj]
    return total

# Calculate percentage grids for red and blue using the above two functions. 
@njit
def calculate_percentage_grids(RedGrid, BlueGrid):
    BluePercentGrid = np.zeros_like(BlueGrid)
    RedPercentGrid = np.zeros_like(RedGrid)
    
    for i in range(1, n):
        for j in range(1, n):
            red_sum = get_neighborhood_sums(RedGrid, i, j, n)
            blue_sum = get_neighborhood_sums(BlueGrid, i, j, n)
            total = red_sum + blue_sum
            
            if total > 0:
                RedPercentGrid[i, j] = red_sum / total
                BluePercentGrid[i, j] = blue_sum / total
    
    # Apply Neumann boundary conditions
    RedPercentGrid[0, :] = RedPercentGrid[1, :]
    RedPercentGrid[n, :] = RedPercentGrid[n-1, :]
    RedPercentGrid[:, 0] = RedPercentGrid[:, 1]
    RedPercentGrid[:, n] = RedPercentGrid[:, n-1]
    
    BluePercentGrid[0, :] = BluePercentGrid[1, :]
    BluePercentGrid[n, :] = BluePercentGrid[n-1, :]
    BluePercentGrid[:, 0] = BluePercentGrid[:, 1]
    BluePercentGrid[:, n] = BluePercentGrid[:, n-1]
    
    return RedPercentGrid, BluePercentGrid

#Heaviside functions for red and blue
@njit
def HRed(p):
    return 1.0 if p < pRedSatisfied else 0.0

@njit
def HBlue(p):
    return 1.0 if p < pBlueSatisfied else 0.0


# Compute diffusion step for red and blue grids using all above functions and methods desribed in code
@njit(parallel=True)
def compute_diffusion_step(RedGrid, BlueGrid, RedPercentGrid, BluePercentGrid):
    R_new = np.copy(RedGrid)
    B_new = np.copy(BlueGrid)
    
    for i in prange(1, n):
        for j in range(1, n):
            # Red diffusion
            if HRed(RedPercentGrid[i, j]) > 0:
                laplacian_r = (
                    4 * (RedPercentGrid[i+1, j] + RedPercentGrid[i-1, j] +
                         RedPercentGrid[i, j+1] + RedPercentGrid[i, j-1]) +
                    (RedPercentGrid[i+1, j+1] + RedPercentGrid[i+1, j-1] +
                     RedPercentGrid[i-1, j+1] + RedPercentGrid[i-1, j-1]) -
                    20 * RedPercentGrid[i, j]
                ) / (6 * dx**2)
                
                R_new[i, j] = RedGrid[i, j] - RedGrid[i, j] * dt * cRed * laplacian_r

            # Blue diffusion
            if HBlue(BluePercentGrid[i, j]) > 0:
                laplacian_b = (
                    4 * (BluePercentGrid[i+1, j] + BluePercentGrid[i-1, j] +
                         BluePercentGrid[i, j+1] + BluePercentGrid[i, j-1]) +
                    (BluePercentGrid[i+1, j+1] + BluePercentGrid[i+1, j-1] +
                     BluePercentGrid[i-1, j+1] + BluePercentGrid[i-1, j-1]) -
                    20 * BluePercentGrid[i, j]
                ) / (6 * dx**2)
                
                B_new[i, j] = BlueGrid[i, j] - BlueGrid[i, j] * dt * cBlue * laplacian_b

            R_new[i, j] = R_new[i, j]
            B_new[i, j] = B_new[i, j]

    return R_new, B_new

# Apply Neumann boundary conditions to a grid
@njit
def apply_boundary_conditions(grid):
    grid[0, :] = grid[1, :]
    grid[n, :] = grid[n-1, :]
    grid[:, 0] = grid[:, 1]
    grid[:, n] = grid[:, n-1]
    return grid

def run_simulation(max_time):
    # Initialize grids
    BlueGrid = np.zeros((n+1, n+1))
    RedGrid = np.zeros((n+1, n+1))
    
    # Apply initial conditions
    for i in range(n+1):
        for j in range(n+1):
            BlueGrid[i, j] = b0(i, j)
            RedGrid[i, j] = r0(i, j)
    
    # Apply boundary conditions
    BlueGrid = apply_boundary_conditions(BlueGrid)
    RedGrid = apply_boundary_conditions(RedGrid)
    
    # Set up visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pcm_red = ax1.pcolormesh(RedGrid, cmap=plt.cm.Reds, vmin=0, vmax=0.5)
    pcm_blue = ax2.pcolormesh(BlueGrid, cmap=plt.cm.Blues, vmin=0, vmax=0.5)
    
    #Color mapping
    plt.colorbar(pcm_red, ax=ax1)
    plt.colorbar(pcm_blue, ax=ax2)
    
    ax1.set_title("Red Distribution")
    ax2.set_title("Blue Distribution")
    
    # Run simulation
    counter = 0.0
    while counter < max_time:
        # Calculate percentage grids
        RedPercentGrid, BluePercentGrid = calculate_percentage_grids(RedGrid, BlueGrid)
        
        # Compute diffusion step
        RedGrid, BlueGrid = compute_diffusion_step(RedGrid, BlueGrid, RedPercentGrid, BluePercentGrid)
        
        # Apply boundary conditions
        RedGrid = apply_boundary_conditions(RedGrid)
        BlueGrid = apply_boundary_conditions(BlueGrid)
        if np.any(RedGrid < 0) or np.any(BlueGrid < 0):
            raise ValueError("Negative concentration encountered!")
        # Update counter
        counter += dt
        
        # Update visualization every 100*n iterations - can change this to speed up or slow down the entire simulation (slowest step, it is rate determining)
        if int(counter / dt) % (100*n) == 0:
            pcm_red.set_array(RedGrid.ravel())
            pcm_blue.set_array(BlueGrid.ravel())
            fig.suptitle(f"Simulation at t: {counter:.3f} [s]")
            ax1.set_title(f"Red Distribution - t: {counter:.3f}")
            ax2.set_title(f"Blue Distribution - t: {counter:.3f}")
            plt.pause(0.0000000000000001)
    
    plt.show()

if __name__ == "__main__":
    # Set Numba threading layer
    import os
    os.environ["NUMBA_NUM_THREADS"] = str(multiprocessing.cpu_count())
    print(f"Running with {multiprocessing.cpu_count()} threads")
    
    # Run simulation
    run_simulation(max_time=1.0)  
