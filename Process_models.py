# PROCESS ORIENTED MODELS - Refactored
import gzip
import csv
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from models import MODELS, IC

# ===============================
# PARAMETERS
# ===============================
T_SNAPSHOTS = 10
K_VALUES = [8, 12, 16, 20]
MAX_STEPS = 8
SEED = 42
MC_SIMULATIONS = 10000  # Monte Carlo simulations for greedy algorithm

random.seed(SEED)
np.random.seed(SEED)

# ===============================
# HELPER FUNCTIONS TO LOAD DATA
# ===============================
def load_snap_txt_gz(path):
    """Load SNAP temporal network from .txt.gz file"""
    edges = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            u, v, t = line.strip().split()[:3]
            edges.append((int(u), int(v), int(float(t))))
    return edges


def load_snap_csv_gz(path, min_positive=1):
    """Load SNAP temporal network from .csv.gz file with ratings"""
    edges = []
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('#') or len(row) < 4:
                continue
            u, v, rating, t = row[:4]
            if int(rating) >= min_positive:
                edges.append((int(u), int(v), int(float(t))))
    return edges


def build_time_snapshots(edges, n_snapshots=T_SNAPSHOTS):
    """Build temporal snapshots from edge list"""
    if not edges:
        return [nx.DiGraph() for _ in range(n_snapshots)]

    all_nodes = set(u for u, v, _ in edges) | set(v for u, v, _ in edges)
    times = [t for _, _, t in edges]
    tmin, tmax = min(times), max(times) + 1
    interval = (tmax - tmin) / n_snapshots
    snapshots = []

    for i in range(n_snapshots):
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        snap_start = tmin + i * interval
        snap_end = tmin + (i + 1) * interval
        for u, v, t in edges:
            if snap_start <= t < snap_end:
                G.add_edge(u, v)
        snapshots.append(G)
    
    return snapshots


# ===============================
# LOAD DATASETS
# ===============================
print("Loading datasets...")
datasets = {}
datasets["CollegeMsg"] = build_time_snapshots(load_snap_txt_gz("Datasets/CollegeMsg.txt.gz"))
datasets["Email-Eu-Core"] = build_time_snapshots(load_snap_txt_gz("Datasets/email-Eu-core-temporal.txt.gz"))
datasets["Bitcoin-OTC"] = build_time_snapshots(load_snap_csv_gz("Datasets/soc-sign-bitcoinotc.csv.gz"))
datasets["Bitcoin-Alpha"] = build_time_snapshots(load_snap_csv_gz("Datasets/soc-sign-bitcoinalpha.csv.gz"))
print("Datasets loaded successfully.")

# ===============================
# GREEDY SEED SELECTION WITH MONTE CARLO
# ===============================
def greedy_with_mc(G, k, mc_simulations=MC_SIMULATIONS):
    """
    Greedy algorithm for influence maximization with Monte Carlo simulations.
    
    Args:
        G: NetworkX DiGraph
        k: Number of seeds to select
        mc_simulations: Number of Monte Carlo simulations to estimate influence spread
    
    Returns:
        Set of k selected seed nodes
    """
    seeds = set()
    
    for i in range(k):
        print(f"  Selecting seed {i+1}/{k}...", end='\r')
        best_node, best_spread = None, -1
        
        for v in G.nodes():
            if v in seeds:
                continue
            
            # Monte Carlo simulation to estimate expected spread
            total_spread = 0
            test_seeds = seeds | {v}
            
            for _ in range(mc_simulations):
                activated = IC(G, test_seeds, p=0.2)
                total_spread += len(activated)
            
            avg_spread = total_spread / mc_simulations
            
            if avg_spread > best_spread:
                best_node = v
                best_spread = avg_spread
        
        if best_node is not None:
            seeds.add(best_node)
    
    print(f"  Selected {k} seeds with Monte Carlo simulations.")
    return seeds


# ===============================
# RUN EXPERIMENT AND PLOT
# ===============================
for dataset_name, snapshots in datasets.items():
    print(f"\n{'='*60}")
    print(f"Running diffusion on {dataset_name}")
    print(f"{'='*60}")
    
    # Aggregate all snapshots into one graph for seed selection
    Gagg = nx.DiGraph()
    for G in snapshots:
        Gagg.add_edges_from(G.edges())
    
    print(f"Aggregated graph: {Gagg.number_of_nodes()} nodes, {Gagg.number_of_edges()} edges")

    # Create figure for plotting
    fig, axs = plt.subplots(1, 4, figsize=(11, 3))
    lines_for_legend = []

    # Define markers and line styles for different models
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "+", "x", "1", "2", "3", "4", "8", "H", "|", "_"]
    line_styles = ["-", "--", "-.", ":"] * 10

    for idx, k in enumerate(K_VALUES):
        print(f"\nProcessing k={k}")
        print(f"Running greedy algorithm with {MC_SIMULATIONS} MC simulations...")
        seeds = greedy_with_mc(Gagg, k, mc_simulations=MC_SIMULATIONS)
        print(f"Seeds selected: {seeds}")
        
        # Initialize tracking for all models
        spreads = {m: [] for m in MODELS}
        cumulative = {m: set(seeds) for m in MODELS}

        # Run diffusion on each snapshot
        print(f"Running diffusion models across {T_SNAPSHOTS} snapshots...")
        for t, G in enumerate(snapshots):
            for m, func in MODELS.items():
                new_active = func(G, cumulative[m])
                cumulative[m] |= new_active
                spreads[m].append(len(cumulative[m]))

        # Plot results for this k value
        ax = axs[idx]
        for (m, y), mk, ls in zip(spreads.items(), markers, line_styles):
            line, = ax.plot(range(T_SNAPSHOTS), y, linestyle=ls, marker=mk, 
                          linewidth=0.7, markersize=2.5)
            if idx == 0:
                lines_for_legend.append((line, m))

        ax.set_title(f"k={k}", fontsize=9)
        ax.set_xlabel("Snapshot", fontsize=8)
        ax.set_ylabel("Activated", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=7)

    # Finalize plot
    fig.subplots_adjust(top=0.78, wspace=0.25, left=0.05, right=0.95)
    fig.legend(
        [l for l, _ in lines_for_legend],
        [m for _, m in lines_for_legend],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=7,
        fontsize=8
    )
    
    output_file = f"{dataset_name}_diffusion_plots.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"\nPlot saved: {output_file}")
    plt.show()

print(f"\n{'='*60}")
print("All experiments completed!")
print(f"{'='*60}")
