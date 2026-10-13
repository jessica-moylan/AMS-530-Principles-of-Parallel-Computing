import math
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator, BayesianExplorationGenerator
from xopt.generators.scipy.latin_hypercube import LatinHypercubeGenerator
from xopt.generators.ga.cnsga import CNSGAGenerator

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

def space_filter_data(df, min_dist=0.05):
    """Filters data so no two points are closer than min_dist in parameter space."""
    kept_rows = []
    points = df[["x", "y"]].to_numpy()

    for i, pt in enumerate(points):
        if not kept_rows:
            kept_rows.append(i)
            continue

        # Distance to previously accepted points
        dists = np.linalg.norm(points[kept_rows] - pt, axis=1)
        if np.all(dists >= min_dist):
            kept_rows.append(i)

    return df.iloc[kept_rows]

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
    variables={"x": [x1, x2], "y": [y1, y2]},
    objectives={"f": "EXPLORE"},
    constraints={"f": ["LESS_THAN", 0.01]}
)

ga_generator = BayesianExplorationGenerator(
    vocs=vocs_opt
)

# Set parameters as attributes afterwards
ga_generator.n_candidates = 10

filtered_data = space_filter_data(X.data, min_dist=0.05)

X_opt = Xopt(
    vocs=vocs_opt, generator=ga_generator, evaluator=evaluator, data=filtered_data
)

# Run 10 iterations of the model training
for i in range(10):
    print(f"Iteration {i+1}")
    X_opt.generator.train_model()
    X_opt.step()

gathered_data_list = COMM.gather(X_opt.data, root=0)

COMM.Barrier()

if rank == 0:
  df = pd.concat(gathered_data_list, ignore_index=True)

  threshold = 0.005
  candidate_zeros = df[df["f"] < threshold][["x", "y"]].to_numpy()

  if len(candidate_zeros) < 30:
     candidate_zeros = df.nsmallest(100, "f")[["x", "y"]].to_numpy()

# Pick 20 widely spread zeros
  spread_20_zeros = select_maxmin_dispersion(candidate_zeros, k=20)


# --- Plotting ---
  fig, ax = plt.subplots(figsize=(8, 6))

# Background contour plot
  x_grid = np.linspace(-1, 1, 250)
  y_grid = np.linspace(-1, 1, 250)
  XX, YY = np.meshgrid(x_grid, y_grid)
  ZZ = equation(XX, YY)

  contour = ax.contourf(XX, YY, ZZ, levels=50, cmap="viridis")
  plt.colorbar(contour, label="f(x, y)")

  n = math.isqrt(size)
  edges = np.linspace(-1, 1, n + 1)

  for edge in edges:
    ax.axhline(edge, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(edge, color="white", linestyle="--", linewidth=1.2, alpha=0.8)

# Plot all sampled points as small translucent dots
  ax.scatter(df["x"], df["y"], c="white", alpha=0.2, s=8, label="Sampled Points")

# Plot the 20 maximally separated zeros
  ax.scatter(
     spread_20_zeros[:, 0],
     spread_20_zeros[:, 1],
     c="red",
     marker="*",
     s=120,
     edgecolors="black",
     linewidths=0.5,
     label="20 Spread Zeros"
  )

  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title("20 Maximally Spread Zeros")
  plt.savefig("plot_4.png", dpi=300, bbox_inches="tight")
  plt.close()
