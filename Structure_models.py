# ===============================
# STRUCTURE ORIENTED DIFFUSION MODELS
# ===============================
import os
os.makedirs("results", exist_ok=True)

import gzip
import csv
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===============================
# PARAMETERS
# ===============================
T_SNAPSHOTS = 10
K_VALUES = [8, 12, 16, 20]
MAX_STEPS = 6   # intentionally small for Bitcoin
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ===============================
# DATA LOADING
# ===============================
def load_snap_txt_gz(path):
    edges = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            u, v, t = line.split()[:3]
            edges.append((int(u), int(v), int(float(t))))
    return edges


def load_snap_csv_gz(path, min_positive=1):
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


def build_time_snapshots(edges):
    if not edges:
        return [nx.DiGraph() for _ in range(T_SNAPSHOTS)]

    nodes = set(u for u, v, _ in edges) | set(v for u, v, _ in edges)
    times = [t for _, _, t in edges]
    tmin, tmax = min(times), max(times) + 1
    interval = (tmax - tmin) / T_SNAPSHOTS

    snapshots = []
    for i in range(T_SNAPSHOTS):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for u, v, t in edges:
            if tmin + i * interval <= t < tmin + (i + 1) * interval:
                G.add_edge(u, v)
        snapshots.append(G)
    return snapshots


# ===============================
# LOAD DATASETS
# ===============================
datasets = {
    "CollegeMsg": build_time_snapshots(load_snap_txt_gz("Datasets/CollegeMsg.txt.gz")),
    "Email-Eu-Core": build_time_snapshots(load_snap_txt_gz("Datasets/email-Eu-core-temporal.txt.gz")),
    "Bitcoin-OTC": build_time_snapshots(load_snap_csv_gz("Datasets/soc-sign-bitcoinotc.csv.gz")),
    #"Bitcoin-Alpha": build_time_snapshots(load_snap_csv_gz("Datasets/soc-sign-bitcoinalpha.csv.gz")),
}

# ===============================
# STRUCTURE-ORIENTED MODELS
# ===============================
def AgentUtility(G, seeds):
    active = set(seeds)
    for _ in range(MAX_STEPS):
        new = set()
        for v in G.nodes():
            if v in active:
                continue
            nbrs = list(G.predecessors(v))
            if nbrs and sum(u in active for u in nbrs) / len(nbrs) >= 0.3:
                new.add(v)
        if not new:
            break
        active |= new
    return active


def LND(G, seeds):
    active = set(seeds)
    frontier = set(seeds)
    for _ in range(MAX_STEPS):
        nxt = set()
        for u in frontier:
            nxt |= set(G.successors(u))
        nxt -= active
        if not nxt:
            break
        active |= nxt
        frontier = nxt
    return active


def LowClustering(G, seeds):
    thresholds = {v: 0.4 for v in G.nodes()}
    active = set(seeds)
    for _ in range(MAX_STEPS):
        new = {v for v in G.nodes()
               if v not in active and
               sum(u in active for u in G.predecessors(v)) /
               max(1, G.in_degree(v)) >= thresholds[v]}
        if not new:
            break
        active |= new
    return active


def ProductAdopter(G, seeds):
    active = set(seeds)
    for _ in range(MAX_STEPS):
        new = set()
        for v in G.nodes():
            if v not in active:
                if random.random() < 0.2 * sum(u in active for u in G.predecessors(v)):
                    new.add(v)
        if not new:
            break
        active |= new
    return active


def HubDiffusion(G, seeds):
    active = set(seeds)
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    hubs = {v for v, _ in hubs[:len(hubs)//10]}
    for _ in range(MAX_STEPS):
        active |= hubs
    return active


def PAM(G, seeds):
    active = set(seeds)
    for _ in range(MAX_STEPS):
        new = set()
        for v in G.nodes():
            if v not in active:
                frac = sum(u in active for u in G.predecessors(v)) / max(1, G.in_degree(v))
                if frac >= 0.35:
                    new.add(v)
        if not new:
            break
        active |= new
    return active


def DensityMacro(G, seeds):
    active = set(seeds)
    density = nx.density(G)
    for _ in range(MAX_STEPS):
        for v in G.nodes():
            if v not in active and random.random() < density:
                active.add(v)
    return active


def BassDiscrete(G, seeds, p=0.02, q=0.4):
    active = set(seeds)
    for _ in range(MAX_STEPS):
        for v in G.nodes():
            if v not in active:
                f = sum(u in active for u in G.predecessors(v)) / max(1, G.in_degree(v))
                if random.random() < p + q * f:
                    active.add(v)
    return active


def ABBM(G, seeds):
    active = set(seeds)
    for _ in range(MAX_STEPS):
        for v in G.nodes():
            if v not in active:
                if random.random() < 0.05 + 0.3 * sum(u in active for u in G.predecessors(v)):
                    active.add(v)
    return active


# ===============================
# MODEL REGISTRY
# ===============================
models = {
    "AgentBased": AgentUtility,
    "LND": LND,
    "LowClustering": LowClustering,
    "PAM": ProductAdopter,
    "HighClustering": HubDiffusion,
    "PABM": PAM,
    "DensityBased": DensityMacro,
    "BIADM": BassDiscrete,
    "ABBM": ABBM
}

# ===============================
# GREEDY SEED SELECTION
# ===============================
def greedy(G, k):
    return set(v for v, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:k])

# ===============================
# RUN EXPERIMENT AND PLOT
# ===============================
for dataset_name, snapshots in datasets.items():
    print(f"Running structure diffusion on {dataset_name} ...")

    Gagg = nx.compose_all(snapshots)

    fig, axs = plt.subplots(1, 4, figsize=(11, 3))
    lines_for_legend = []

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    line_styles = ["-", "--", "-.", ":"] * 3

    for idx, k in enumerate(K_VALUES):
        seeds = greedy(Gagg, k)
        cumulative = {m: set(seeds) for m in models}
        spreads = {m: [] for m in models}

        for G in snapshots:
            for m, func in models.items():
                cumulative[m] |= func(G, cumulative[m])
                spreads[m].append(len(cumulative[m]))

        ax = axs[idx]
        for (m, y), mk, ls in zip(spreads.items(), markers, line_styles):
            line, = ax.plot(range(T_SNAPSHOTS), y,
                            linestyle=ls, marker=mk,
                            linewidth=0.7, markersize=2.5)
            if idx == 0:
                lines_for_legend.append((line, m))

        ax.set_title(f"k={k}", fontsize=9)
        ax.set_xlabel("Snapshot", fontsize=8)
        ax.set_ylabel("Activated", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=7)

    fig.subplots_adjust(top=0.78, wspace=0.25, left=0.05, right=0.95)
    fig.legend(
        [l for l, _ in lines_for_legend],
        [m for _, m in lines_for_legend],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        fontsize=8
    )

    plt.savefig(f"results/{dataset_name}_structure_diffusion_plots.pdf",
                bbox_inches="tight")
    plt.show()
