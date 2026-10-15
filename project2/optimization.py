import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpi4py import MPI
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import BayesianExplorationGenerator
from xopt.generators.scipy.latin_hypercube import LatinHypercubeGenerator

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

# Constants
BOUNDS = (-1.0, 1.0)
NUM_ZEROS = 20


def equation(x, y):
  phi1 = x * (x**2 + y**2)
  phi2 = y**3 - x**2 * y + 0.5 * x
  return 0.5 * (np.sin(10.0 * phi1) ** 2 + np.sin(10.0 * phi2) ** 2)


def evaluate(inputs):
  return {"f": float(equation(inputs["x"], inputs["y"]))}


def split_into_grid(z):
  """Split the domain into an n x n grid of sections, one per processor."""
  n = math.isqrt(z)
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

def select_maxmin_dispersion(candidates, k=NUM_ZEROS):
  """Greedy farthest-point selection of k candidates."""
  if len(candidates) <= k:
    return candidates

  selected = [0]
  min_dists = np.linalg.norm(candidates - candidates[0], axis=1)
  for _ in range(1, k):
    nxt = int(np.argmax(min_dists))
    selected.append(nxt)
    min_dists = np.minimum(
        min_dists, np.linalg.norm(candidates - candidates[nxt], axis=1)
    )
  return candidates[selected]


def space_filter_data(df):
  """Remove points so that no two are closer than 0.02 apart"""
  points = df[["x", "y"]].to_numpy()
  kept = []
  for i, pt in enumerate(points):
    if not kept or np.all(np.linalg.norm(points[kept] - pt, axis=1) >= 0.02):
      kept.append(i)
  return df.iloc[kept]


def reduce_seed_data(df, rng):
  """Thin and cap the seed data: half the best points, half random."""
  df = space_filter_data(df)
  if len(df) <= 200:
    return df

  # Bias towards points that are close to 0 and than select 100 additional, regardless of closeness to 0.
  best = df.nsmallest(100, "f")
  rest = df.drop(best.index)
  extra = rest.iloc[rng.choice(len(rest), 100, replace=False)]
  return pd.concat([best, extra])

# Presets for the  evaluation function
evaluator = Evaluator(function=evaluate)
  
# Split the sample space into even spaces based on the numbe of processors  
regions = split_into_grid(size)

if rank == 0:
  # Constrain x and y withing (-1,1) than encourage the exploration of the sample space for the LatinHypercubeGenerator random point generator
  global_vocs = VOCS(
    variables={
        "x": list(BOUNDS),
        "y": list(BOUNDS),
    },
    objectives={"f": "EXPLORE"},
  )

  lhs_generator = LatinHypercubeGenerator(vocs=global_vocs, batch_size=16384)

  X_lhs = Xopt(vocs=global_vocs, generator=lhs_generator, evaluator=evaluator)

  for _ in range(16384):
    X_lhs.step()

  pts_array = X_lhs.data[["x", "y", "f"]].copy().to_numpy()

  global_data = pd.DataFrame(pts_array, columns=["x", "y", "f"])
  chunks = []
  for section in regions:
    x1, x2 = section["x"]
    y1, y2 = section["y"]
    x_mask = (pts_array[:, 0] >= x1) & (
        (pts_array[:, 0] < x2) if x2 < BOUNDS[1] else (pts_array[:, 0] <= x2)
    )
    y_mask = (pts_array[:, 1] >= y1) & (
        (pts_array[:, 1] < y2) if y2 < BOUNDS[1] else (pts_array[:, 1] <= y2)
    )
    chunks.append(pts_array[x_mask & y_mask])
else:
  global_data = None
  chunks = None

local_block = COMM.scatter(chunks, root=0)
region_df = pd.DataFrame(local_block, columns=["x", "y", "f"])

local_section = regions[rank]
x1, x2 = local_section["x"]
y1, y2 = local_section["y"]

print(
    f"[rank {rank}] region x=[{x1:.3f}, {x2:.3f}] y=[{y1:.3f}, {y2:.3f}] "
    f"with {len(region_df)} global LHS points",
    flush=True,
)

rng = np.random.default_rng(rank)
local_new = pd.DataFrame(columns=["x", "y", "f"])

seed_df = reduce_seed_data(region_df[["x", "y", "f"]], rng)

vocs = VOCS(
    variables={"x": [x1, x2], "y": [y1, y2]},
    objectives={"f": "EXPLORE"},
    constraints={"f": ["LESS_THAN", 0.01]},
)
generator = BayesianExplorationGenerator(vocs=vocs)
X = Xopt(vocs=vocs, generator=generator, evaluator=evaluator, data=seed_df)

for i in range(40):
    X.step()

# keep only what BO added; the seed points already live in global_data
local_new = X.data.iloc[len(seed_df):][["x", "y", "f"]]

gathered = COMM.gather(local_new, root=0)

# Wait until all processors are done with 
COMM.Barrier()

if rank == 0:
  bo_df = pd.concat(gathered, ignore_index=True)
  df = pd.concat([global_data[["x", "y", "f"]], bo_df], ignore_index=True)

  candidate_zeros = df[df["f"] < 0.005][["x", "y"]].to_numpy()
  if len(candidate_zeros) < 30:
    candidate_zeros = df.nsmallest(100, "f")[["x", "y"]].to_numpy()

  spread = select_maxmin_dispersion(candidate_zeros, k=NUM_ZEROS)
  print(
      f"Total points: {len(df)} (LHS {len(global_data)}, BO {len(bo_df)}); "
      f"zeros found: {len(candidate_zeros)}"
  )

  fig, ax = plt.subplots(figsize=(8, 6))
  xg = np.linspace(*BOUNDS, 250)
  XX, YY = np.meshgrid(xg, xg)
  contour = ax.contourf(XX, YY, equation(XX, YY), levels=50, cmap="viridis")
  plt.colorbar(contour, label="f(x, y)")

  # outline each processor's region
  for sec in regions:
    rx1, rx2 = sec["x"]
    ry1, ry2 = sec["y"]
    ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False,
                           edgecolor="white", linestyle="--", linewidth=1.2,
                           alpha=0.8))

  ax.scatter(global_data["x"], global_data["y"], c="white", alpha=0.1, s=3, label="Global LHS points")

  ax.scatter(bo_df["x"], bo_df["y"], c="orange", alpha=0.5, s=8, label="Bayesian exploration points")

  ax.scatter(spread[:, 0], spread[:, 1], c="red", marker="*", s=120, edgecolor="black", linewidths=0.5, label=f"{NUM_ZEROS} spread zeros")

  ax.set_xlim(*BOUNDS)
  ax.set_ylim(*BOUNDS)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title(f"{NUM_ZEROS} Maximally Spread Zeros ({size} processors)")
  ax.legend(loc="upper right", fontsize=8)
  plt.savefig(f"plot_{size}.png", dpi=300, bbox_inches="tight")
  plt.close()
