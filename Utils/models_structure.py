"""
Structure-Oriented Diffusion Models for Social Network Analysis
Contains models based on network structure and topology
"""

import random
import networkx as nx

MAX_STEPS = 6  # intentionally small for Bitcoin datasets

# ===============================
# STRUCTURE-ORIENTED MODELS
# ===============================

def AgentUtility(G, seeds):
    """
    Agent-Based Utility Model:
    Nodes adopt when utility from active neighbors exceeds threshold.
    """
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
    """
    Layer-by-Layer Network Diffusion (LND):
    Spreads to all direct neighbors at each step.
    """
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
    """
    Low Clustering Coefficient Model:
    Threshold-based adoption with fixed threshold.
    """
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
    """
    Product Adopter Model (PAM):
    Probabilistic adoption based on number of active neighbors.
    """
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
    """
    Hub-Based Diffusion (High Clustering):
    High-degree nodes (hubs) adopt immediately.
    """
    active = set(seeds)
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    hubs = {v for v, _ in hubs[:len(hubs)//10]}
    for _ in range(MAX_STEPS):
        active |= hubs
    return active


def PAM(G, seeds):
    """
    Preferential Attachment Model (PAAM):
    Nodes adopt when fraction of active neighbors exceeds threshold.
    """
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
    """
    Density-Based Macro Model:
    Adoption probability proportional to network density.
    """
    active = set(seeds)
    density = nx.density(G)
    for _ in range(MAX_STEPS):
        for v in G.nodes():
            if v not in active and random.random() < density:
                active.add(v)
    return active


def BassDiscrete(G, seeds, p=0.02, q=0.4):
    """
    Bass Innovation-Adoption Diffusion Model (BIADM):
    Combines innovation (p) and imitation (q) parameters.
    """
    active = set(seeds)
    for _ in range(MAX_STEPS):
        for v in G.nodes():
            if v not in active:
                f = sum(u in active for u in G.predecessors(v)) / max(1, G.in_degree(v))
                if random.random() < p + q * f:
                    active.add(v)
    return active


def ABBM(G, seeds):
    """
    Agent-Based Bass Model (ABBM):
    Combines spontaneous adoption and social influence.
    """
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

MODELS = {
    "AgentBased": AgentUtility,
    "LND": LND,
    "LowClustering": LowClustering,
    "PAM": ProductAdopter,
    "HighClustering": HubDiffusion,
    "PAAM": PAM,
    "DensityBased": DensityMacro,
    "BIADM": BassDiscrete,
    "ABBM": ABBM
}
