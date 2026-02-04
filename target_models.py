# TARGET ORIENTED DIFFUSION MODELS - Refactored
import os
import gzip
import csv
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from models_target import MODELS

# ===============================
# CREATE RESULTS DIRECTORY
# ===============================
os.makedirs("results", exist_ok=True)

# ===============================
# PARAMETERS
# ===============================
T_SNAPSHOTS = 10
K_VALUES = [8, 12, 16, 20]
MAX_STEPS = 10
MC_RUNS = 10  # Monte Carlo runs for greedy seed selection
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ===============================
# DATA LOADING
# ===============================
def load_snap_txt_gz(path):
    """Load SNAP temporal network from .txt.gz file"""
    edges = []
    with gzip.open(path, "rt") as f:
        for l in f:
            if l.startswith('#'):
                continue
            u, v, t = l.split()[:3]
            edges.append((int(u), int(v), int(float(t))))
    return edges


def load_snap_csv_gz(path):
    """Load SNAP temporal network from .csv.gz file with ratings"""
    edges = []
    with gzip.open(path, "rt") as f:
        r = csv.reader(f)
        for row in r:
            if row[0].startswith('#'):
                continue
            u, v, rating, t = row[:4]
            if int(rating) >= 1:
                edges.append((int(u), int(v), int(float(t))))
    return edges


def build_snapshots(edges):
    """Build temporal snapshots from edge list"""
    nodes = set(u for u, v, _ in edges) | set(v for u, v, _ in edges)
    times = [t for _, _, t in edges]
    tmin, tmax = min(times), max(times) + 1
    interval = (tmax - tmin) / T_SNAPSHOTS
    snaps = []
    for i in range(T_SNAPSHOTS):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for u, v, t in edges:
            if tmin + i * interval <= t < tmin + (i + 1) * interval:
                G.add_edge(u, v)
        snaps.append(G)
    return snaps


# ===============================
# LOAD DATASETS
# ===============================
print("Loading datasets...")
datasets = {
    "CollegeMsg": build_snapshots(load_snap_txt_gz("Datasets/CollegeMsg.txt.gz")),
    "Email-Eu-Core": build_snapshots(load_snap_txt_gz("Datasets/email-Eu-core-temporal.txt.gz")),
    "Bitcoin-OTC": build_snapshots(load_snap_csv_gz("Datasets/soc-sign-bitcoinotc.csv.gz")),
    "Bitcoin-Alpha": build_snapshots(load_snap_csv_gz("Datasets/soc-sign-bitcoinalpha.csv.gz"))
}
print("Datasets loaded successfully.")

# ===============================
# MODEL-SPECIFIC GREEDY WITH MONTE CARLO
# ===============================
def greedy_per_model(G, k, model_func, mc_runs=MC_RUNS):
    """
    Greedy algorithm that selects seeds specific to each model.
    Uses Monte Carlo simulations for spread estimation.
    
    Args:
        G: NetworkX DiGraph
        k: Number of seeds to select
        model_func: The diffusion model function to use
        mc_runs: Number of Monte Carlo runs for spread estimation
    
    Returns:
        Set of k selected seed nodes
    """
    S = set()
    for i in range(k):
        print(f"    Seed {i+1}/{k}...", end='\r')
        best = None
        best_spread = -1
        for v in G.nodes():
            if v in S:
                continue
            # Monte Carlo estimation of expected spread
            spread = np.mean([len(model_func(G, S | {v})) for _ in range(mc_runs)])
            if spread > best_spread:
                best, best_spread = v, spread
        if best is not None:
            S.add(best)
    print(f"    Selected {k} seeds for model.")
    return S


# ===============================
# RUN EXPERIMENT AND PLOT
# ===============================
for dataset_name, snapshots in datasets.items():
    print(f"\n{'='*60}")
    print(f"Running target-oriented diffusion on {dataset_name}")
    print(f"{'='*60}")

    # Aggregate graph for seed selection
    Gagg = nx.compose_all(snapshots)
    print(f"Aggregated graph: {Gagg.number_of_nodes()} nodes, {Gagg.number_of_edges()} edges")

    # Create figure for plotting
    fig, axs = plt.subplots(1, 4, figsize=(11, 3))
    lines_for_legend = []

    # Define markers and line styles for different models
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "8"]
    line_styles = ["-", "--", "-.", ":"] * 4

    for idx, k in enumerate(K_VALUES):
        print(f"\nProcessing k={k}")
        print(f"Running model-specific greedy with {MC_RUNS} MC runs per model...")

        # --- EXACT GREEDY PER MODEL ---
        # Each model gets its own optimal seed set
        seeds_per_model = {}
        for m, func in MODELS.items():
            print(f"  Model: {m}")
            seeds_per_model[m] = greedy_per_model(Gagg, k, func, mc_runs=MC_RUNS)

        # Initialize cumulative activation tracking
        cumulative = {m: set(seeds_per_model[m]) for m in MODELS}
        spreads = {m: [] for m in MODELS}

        # --- Temporal diffusion across snapshots ---
        print(f"Running diffusion models across {T_SNAPSHOTS} snapshots...")
        for G in snapshots:
            for m, func in MODELS.items():
                cumulative[m] |= func(G, cumulative[m])
                spreads[m].append(len(cumulative[m]))

        # --- Plotting ---
        ax = axs[idx]
        for (m, y), mk, ls in zip(spreads.items(), markers, line_styles):
            line, = ax.plot(
                range(T_SNAPSHOTS),
                y,
                linestyle=ls,
                marker=mk,
                linewidth=0.7,
                markersize=2.5
            )
            if idx == 0:
                lines_for_legend.append((line, m))

        ax.set_title(f"k={k}", fontsize=9)
        ax.set_xlabel("Snapshot", fontsize=8)
        ax.set_ylabel("Activated", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=7)

    # --- Global layout & legend ---
    fig.subplots_adjust(top=0.78, wspace=0.25, left=0.05, right=0.95)
    fig.legend(
        [l for l, _ in lines_for_legend],
        [m for _, m in lines_for_legend],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        fontsize=8
    )

    output_file = f"results/{dataset_name}_target_oriented_diffusion_plots.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"\nPlot saved: {output_file}")
    plt.show()

print(f"\n{'='*60}")
print("All experiments completed!")
print(f"{'='*60}")
