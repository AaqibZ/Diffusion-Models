"""
Competition-Oriented Diffusion Models for Social Network Analysis
Contains models that handle competitive influence and multi-source diffusion
"""

import random
import networkx as nx

MAX_STEPS = 8

# ===============================
# COMPETITION-ORIENTED MODELS
# ===============================

def DBM(G, seeds):
    """
    Distance-Based Model (DBM):
    All reachable nodes from seeds are activated based on shortest paths.
    """
    active = set(seeds)
    dist = nx.multi_source_dijkstra_path_length(G.reverse(), seeds)
    return set(dist.keys())


def WPM(G, seeds):
    """
    Wave Propagation Model (WPM):
    Layer-by-layer diffusion to all neighbors at each step.
    """
    active = set(seeds)
    frontier = set(seeds)
    for _ in range(MAX_STEPS):
        new = set()
        for u in frontier:
            for v in G.successors(u):
                if v not in active:
                    new.add(v)
        if not new:
            break
        active |= new
        frontier = new
    return active


def WPTM(G, seeds):
    """
    Weight-Proportional Threshold Model (WPTM):
    Threshold-based activation with random thresholds.
    """
    thresholds = {v: random.random() for v in G.nodes()}
    active = set(seeds)

    for _ in range(MAX_STEPS):
        new = set()
        for v in G.nodes():
            if v in active:
                continue
            preds = list(G.predecessors(v))
            influence = sum(1 for p in preds if p in active)
            if preds and influence / len(preds) >= thresholds[v]:
                new.add(v)
        if not new:
            break
        active |= new
    return active


def STM(G, seeds):
    """
    Separated Threshold Model (STM):
    Probabilistic threshold-based activation.
    """
    thresholds = {v: random.random() for v in G.nodes()}
    active = set(seeds)

    for _ in range(MAX_STEPS):
        new = set()
        for v in G.nodes():
            if v in active:
                continue
            preds = list(G.predecessors(v))
            if preds and random.random() < sum(p in active for p in preds) / len(preds):
                new.add(v)
        if not new:
            break
        active |= new
    return active


def DCM(G, seeds):
    """
    Decision-aware Competitive Model (DCM):
    Nodes activate when they have at least 2 active predecessors.
    """
    thinking = set()
    active = set(seeds)

    for _ in range(MAX_STEPS):
        for v in G.nodes():
            if v not in active:
                preds = list(G.predecessors(v))
                if preds and sum(p in active for p in preds) >= 2:
                    thinking.add(v)
        active |= thinking
    return active


def TIC(G, seeds):
    """
    Timeliness Independent Cascade (TIC):
    Nodes activate after sufficient exposure (exposure >= 2).
    """
    active = set(seeds)
    exposure = {v: 0 for v in G.nodes()}

    for _ in range(MAX_STEPS):
        for u in active:
            for v in G.successors(u):
                exposure[v] += 1
        for v in G.nodes():
            if v not in active and exposure[v] >= 2:
                active.add(v)
    return active


def AtI(G, seeds):
    """
    Awareness-to-Influence (AtI):
    
    Phase 1 (Awareness):
    Nodes become aware via neighbors' exposure.
    
    Phase 2 (Influence):
    Aware nodes probabilistically convert to active adopters.
    
    Ensures disjoint awareness and activation phases.
    """
    aware = set(seeds)
    active = set(seeds)

    for _ in range(MAX_STEPS):
        # -------- Awareness phase --------
        new_aware = set()
        for u in list(aware):  # iterate over snapshot
            new_aware |= set(G.successors(u))

        aware |= new_aware

        # -------- Influence phase --------
        new_active = set()
        for v in aware:
            if v not in active and random.random() < 0.4:
                new_active.add(v)

        if not new_active:
            break

        active |= new_active

    return active


def TrCID(G, seeds):
    """
    Trust-aware Competitive Influence Diffusion (TrCID):
    Probabilistic cascade with trust parameter.
    """
    active = set(seeds)
    for _ in range(MAX_STEPS):
        new = set()
        for u in active:
            for v in G.successors(u):
                if random.random() < 0.3:
                    new.add(v)
        active |= new
    return active


def TICC(G, seeds):
    """
    Targeted Influence Competition Cascade (TICC):
    Lower probability cascade for competitive scenarios.
    """
    active = set(seeds)
    for _ in range(MAX_STEPS):
        for u in list(active):
            for v in G.successors(u):
                if random.random() < 0.2:
                    active.add(v)
    return active


def TBCELF(G, seeds):
    """
    Temporal Budget-aware CELF (TBCELF):
    Budget-constrained diffusion with limited activations.
    """
    active = set(seeds)
    budget = len(seeds) * 2

    for _ in range(MAX_STEPS):
        if budget <= 0:
            break
        for u in list(active):
            for v in G.successors(u):
                if random.random() < 0.25:
                    active.add(v)
                    budget -= 1
    return active


# ===============================
# MODEL REGISTRY
# ===============================

MODELS = {
    "DBM": DBM,
    "WPM": WPM,
    "WPTM": WPTM,
    "STM": STM,
    "DCM": DCM,
    "TIC": TIC,
    "AtI": AtI,
    "TrCID": TrCID,
    "TICC": TICC,
    "TBCELF": TBCELF
}
