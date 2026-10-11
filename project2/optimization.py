import math
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.scipy.latin_hypercube import LatinHypercubeGenerator

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()


def equation(x, y):
  phi1 = x * (x**2 + y**2)
  phi2 = y**3 - x**2 * y + 0.5 * x
  return 0.5 * (np.sin(10.0 * phi1) ** 2 + np.sin(10.0 * phi2) ** 2)


def evaluate(inputs):
  return {"f": float(equation(inputs["x"], inputs["y"]))}


def select_maxmin_dispersion(candidates, k=20):
  """Select k candidates that are maximally spread apart."""
  if len(candidates) <= k:
    return candidates

  dist_matrix = cdist(candidates, candidates)
  selected_indices = [0]
  min_dists = dist_matrix[:, 0]

  for _ in range(1, k):
    next_idx = np.argmax(min_dists)
    selected_indices.append(next_idx)
    min_dists = np.minimum(min_dists, dist_matrix[:, next_idx])
  return candidates[selected_indices]


def split_into_grid(z):
  """Split the grid into even sections, one section per processor"""
  n = math.isqrt(z)
  if n * n != z:
    raise ValueError("Number of processors must be a perfect square.")

  edges = np.linspace(-1, 1, n + 1)
  sections = []
  for i in range(n):
    for j in range(n):
      section = {
          "x": (edges[j], edges[j + 1]),
          "y": (edges[i], edges[i + 1]),
      }
      sections.append(section)

  return sections


if rank == 0:
  grid_sections = split_into_grid(size)
else:
  grid_sections = None

local_section = COMM.scatter(grid_sections, root=0)

evaluator = Evaluator(function=evaluate)

x1, x2, y1, y2 = (
    local_section["x"][0],
    local_section["x"][1],
    local_section["y"][0],
    local_section["y"][1],
)
vocs = VOCS(
    variables={"x": [x1, x2], "y": [y1, y2]}, objectives={"f": "EXPLORE"}
)

generator = LatinHypercubeGenerator(vocs=vocs, batch_size=2048)

# Use the Latin Hypercube for sampling the subdomain
X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator)

for _ in range(2048):
  X.step()

vocs_opt = VOCS(
    variables={"x": [x1, x2], "y": [y1, y2]}, objectives={"f": "MINIMIZE"}
)

bo_generator = UpperConfidenceBoundGenerator(vocs=vocs_opt, beta=2.0)

X_opt = Xopt(
    vocs=vocs_opt, generator=bo_generator, evaluator=evaluator, data=X.data
)

# Training the model
for i in range(6):
  X_opt.generator.train_model()
  X_opt.step()

gathered_data_list = COMM.gather(X_opt.data, root=0)

COMM.Barrier()

if rank == 0:
  df = pd.concat(gathered_data_list, ignore_index=True)

  threshold = 0.001
  candidates_df = df[df["f"] < threshold]
  candidate_zeros = candidates_df[["x", "y"]].to_numpy()

  if len(candidate_zeros) < 20:
    print(
        f"Warning: Only found {len(candidate_zeros)} candidates under threshold"
        " in Xopt data. Consider adjusting threshold."
    )
    # If not enough points make the threshold take the lowest 50 points overall
    candidate_zeros = (
        df.nsmallest(50, "f")[["x", "y"]].to_numpy()
        if len(candidate_zeros) == 0
        else candidate_zeros
    )

  # 3. Select 20 zeros that are the most spread apart
  spread_zeros = select_maxmin_dispersion(candidate_zeros, k=20)

  # Plotting results
  x_grid = np.linspace(-1, 1, 250)
  y_grid = np.linspace(-1, 1, 250)
  X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
  Z = equation(X_mesh, Y_mesh)
  fig, ax = plt.subplots(figsize=(9, 7))
  contour = ax.contourf(X_mesh, Y_mesh, Z, levels=50, cmap="viridis")
  plt.colorbar(contour, label="f(x, y)")

  n = math.isqrt(size)
  edges = np.linspace(-1, 1, n + 1)

  for edge in edges:
    ax.axhline(edge, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(edge, color="white", linestyle="--", linewidth=1.2, alpha=0.8)

  ax.scatter(
      spread_zeros[:, 0],
      spread_zeros[:, 1],
      c="red",
      marker="*",
      s=120,
      edgecolors="black",
      linewidths=0.5,
  )

  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title("20 Maximally Spread Zeros")
  plt.savefig("plot_4.png", dpi=300, bbox_inches="tight")
  plt.close()
