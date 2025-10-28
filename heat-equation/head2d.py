"""
Distributed 2D Heat Equation Solver using Dask Array
----------------------------------------------------
Solves ∂T/∂t = α∇²T on a 2D grid with parallel updates.
"""

from dask.distributed import Client
import dask.array as da
import numpy as np
import time
import matplotlib.pyplot as plt

# === Step 1. Cluster Connection ===
MASTER_IP = "<master_ip>:8786"  # Replace with your master IP
client = Client(MASTER_IP)
print("Connected to cluster:", client)

# === Step 2. Simulation Parameters ===
nx, ny = 400, 400         # grid size
dx = dy = 1.0
alpha = 0.1
dt = 0.25 * dx * dy / alpha
steps = 300
chunk = (100, 100)

# === Step 3. Initialize Temperature Field ===
T = da.zeros((nx, ny), chunks=chunk)
# Hot spot in center
T = T.map_blocks(lambda x: np.pad(x, ((0,0),(0,0)), mode="constant"))
T = T.compute_chunk_sizes()
T = T.persist()

cx, cy = nx // 2, ny // 2
T = T.map_blocks(lambda x: x)
T = T + da.from_array(np.exp(-(((np.arange(nx)[:, None]-cx)**2 + (np.arange(ny)-cy)**2)/200.0)), chunks=chunk)

def heat_step(T):
    """Single time step update with Neumann boundary conditions."""
    lap = (
        T[:-2,1:-1] + T[2:,1:-1] +
        T[1:-1,:-2] + T[1:-1,2:] - 4*T[1:-1,1:-1]
    )
    T_next = T[1:-1,1:-1] + alpha * lap
    return da.pad(T_next, 1, mode="edge")

# === Step 4. Time Evolution ===
print(f"Simulating {steps} time steps on {nx}x{ny} grid...")
t0 = time.time()

for i in range(steps):
    T = T.map_overlap(heat_step, depth=1, boundary="reflect")
    if i % 50 == 0:
        print(f"Step {i}/{steps}")

T_final = T.compute()
t1 = time.time()
print(f"✅ Simulation done in {t1 - t0:.2f} s")

# === Step 5. Save Result ===
np.save("heat_final.npy", T_final)
plt.imshow(T_final, cmap='hot')
plt.title("Final Temperature Distribution")
plt.colorbar()
plt.savefig("heat_final.png")
print("Saved final heat map to heat_final.png")
