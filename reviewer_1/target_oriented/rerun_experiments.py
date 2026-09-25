
import os
import gzip
import csv
import random
import multiprocessing as mp

import networkx as nx
import numpy as np
import pandas as pd



OUTPUT_DIR = "Reviewer1_Target_Uncertainty_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)



T_SNAPSHOTS = 10
K_VALUES = [8, 12, 16, 20]
MAX_STEPS = 10

N_RUNS = 1000

MC_RUNS = 1000

SEED = 42

random.seed(SEED)
np.random.seed(SEED)



def load_snap_txt_gz(path):

    edges = []

    with gzip.open(path, "rt") as f:

        for line in f:

            if line.startswith("#") or not line.strip():
                continue

            u, v, t = line.split()[:3]

            edges.append(
                (
                    int(u),
                    int(v),
                    int(float(t))
                )
            )

    return edges


def load_snap_csv_gz(path):

    edges = []

    with gzip.open(path, "rt") as f:

        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            if row[0].startswith("#"):
                continue

            u, v, rating, t = row[:4]

            if int(rating) >= 1:

                edges.append(
                    (
                        int(u),
                        int(v),
                        int(float(t))
                    )
                )

    return edges


def build_snapshots(edges):

    if not edges:

        return [
            nx.DiGraph()
            for _ in range(T_SNAPSHOTS)
        ]

    nodes = (
        set(
            u for u, v, _ in edges
        )
        |
        set(
            v for u, v, _ in edges
        )
    )

    times = [
        t for _, _, t in edges
    ]

    tmin = min(times)
    tmax = max(times) + 1

    interval = (
        (tmax - tmin)
        /
        T_SNAPSHOTS
    )

    snaps = []

    for i in range(T_SNAPSHOTS):

        G = nx.DiGraph()

        G.add_nodes_from(nodes)

        lower = (
            tmin
            +
            i * interval
        )

        upper = (
            tmin
            +
            (i + 1) * interval
        )

        for u, v, t in edges:

            if lower <= t < upper:

                G.add_edge(
                    u,
                    v
                )

        snaps.append(G)

    return snaps



datasets = {

    "CollegeMsg":
        build_snapshots(
            load_snap_txt_gz(
                "Datasets/CollegeMsg.txt.gz"
            )
        ),

    "Email-Eu-Core":
        build_snapshots(
            load_snap_txt_gz(
                "Datasets/email-Eu-core-temporal.txt.gz"
            )
        ),

    "Bitcoin-OTC":
        build_snapshots(
            load_snap_csv_gz(
                "Datasets/soc-sign-bitcoinotc.csv.gz"
            )
        ),
}



def VMID(G, seeds):

    A = set(seeds)

    for _ in range(MAX_STEPS):

        for u in list(A):

            for v in G.successors(u):

                if (
                    v not in A
                    and random.random() < 0.3
                ):

                    A.add(v)

    return A


def MAT(G, seeds):

    trust = {
        (u, v): random.random()
        for u, v in G.edges()
    }

    A = set(seeds)

    for _ in range(MAX_STEPS):

        for v in G.nodes():

            if v in A:
                continue

            influence = sum(
                trust.get(
                    (u, v),
                    0
                )
                for u in G.predecessors(v)
                if u in A
            )

            if influence >= 0.5:

                A.add(v)

    return A


def CAND(G, seeds):

    A = set(seeds)

    for _ in range(MAX_STEPS):

        for v in G.nodes():

            if v not in A:

                if (
                    sum(
                        u in A
                        for u in G.predecessors(v)
                    )
                    >= 2
                ):

                    A.add(v)

    return A


def FSC_SB(G, seeds):

    A = set(seeds)
    B = set()

    for _ in range(MAX_STEPS):

        for u in list(A):

            for v in G.successors(u):

                if (
                    v in A
                    or v in B
                ):
                    continue

                r = random.random()

                if r < 0.3:

                    A.add(v)

                elif r < 0.5:

                    B.add(v)

    return A


def FSC_N(G, seeds):

    P = set(seeds)
    N = set()

    for _ in range(MAX_STEPS):

        for u in list(P):

            for v in G.successors(u):

                if (
                    v not in P
                    and random.random() < 0.3
                ):

                    P.add(v)

        for u in list(N):

            for v in G.successors(u):

                if (
                    v not in N
                    and random.random() < 0.3
                ):

                    N.add(v)

    return P


def FST_SB(G, seeds):

    A = set(seeds)
    B = set()

    for _ in range(MAX_STEPS):

        for v in G.nodes():

            if (
                v in A
                or v in B
            ):
                continue

            pos = sum(
                u in A
                for u in G.predecessors(v)
            )

            neg = sum(
                u in B
                for u in G.predecessors(v)
            )

            if pos >= 2:

                A.add(v)

            elif neg >= 2:

                B.add(v)

    return A


def FST_N(G, seeds):

    P = set(seeds)
    N = set()

    for _ in range(MAX_STEPS):

        for v in G.nodes():

            if (
                v in P
                or v in N
            ):
                continue

            if (
                sum(
                    u in P
                    for u in G.predecessors(v)
                )
                >= 2
            ):

                P.add(v)

            elif (
                sum(
                    u in N
                    for u in G.predecessors(v)
                )
                >= 2
            ):

                N.add(v)

    return P


def IC_u(G, seeds):

    EG = {
        v: random.random()
        for v in G.nodes()
    }

    FG = {
        v: random.random()
        for v in G.nodes()
    }

    A = set(seeds)

    for _ in range(MAX_STEPS):

        for u in list(A):

            for v in G.successors(u):

                if v not in A:

                    p = (
                        EG[u]
                        *
                        (
                            1
                            /
                            max(
                                1,
                                G.out_degree(u)
                            )
                        )
                        *
                        FG[u]
                    )

                    if random.random() < p:

                        A.add(v)

    return A


def LT_u(G, seeds):

    EG = {
        v: random.random()
        for v in G.nodes()
    }

    FG = {
        v: random.random()
        for v in G.nodes()
    }

    theta = {
        v: random.random()
        for v in G.nodes()
    }

    A = set(seeds)

    for _ in range(MAX_STEPS):

        for v in G.nodes():

            if v in A:
                continue

            influence = sum(
                EG[u] * FG[u]
                for u in G.predecessors(v)
                if u in A
            )

            if influence >= theta[v]:

                A.add(v)

    return A


def UAD(G, seeds):

    aware = set(seeds)
    active = set(seeds)

    for _ in range(MAX_STEPS):

        for u in list(aware):

            aware |= set(
                G.successors(u)
            )

        for v in aware:

            if (
                v not in active
                and random.random() < 0.4
            ):

                active.add(v)

    return active


def ISR(G, seeds):

    A = set(seeds)

    count = {
        v: 0
        for v in G.nodes()
    }

    for _ in range(MAX_STEPS):

        for u in list(A):

            for v in G.successors(u):

                count[v] += 1

                if (
                    v not in A
                    and random.random()
                    <
                    1 - (
                        0.7
                        **
                        count[v]
                    )
                ):

                    A.add(v)

    return A


def ACT(G, seeds):

    trust = {
        v: 0.5
        for v in G.nodes()
    }

    A = set(seeds)

    for _ in range(MAX_STEPS):

        for u in list(A):

            for v in G.successors(u):

                if random.random() < trust[u]:

                    A.add(v)

                    trust[u] = min(
                        1,
                        trust[u] + 0.05
                    )

                else:

                    trust[u] = max(
                        0,
                        trust[u] - 0.02
                    )

    return A



models = {

    "VMID": VMID,
    "MAT": MAT,
    "CAND": CAND,
    "FSC-SB": FSC_SB,
    "FSC-N": FSC_N,
    "FST-SB": FST_SB,
    "FST-N": FST_N,
    "IC-u": IC_u,
    "LT-u": LT_u,
    "UAD": UAD,
    "ISR": ISR,
    "ACT": ACT
}



def greedy(G, k, model):

    S = set()

    for _ in range(k):

        best = None
        best_spread = -1

        for v in G.nodes():

            if v in S:
                continue

            spread = np.mean(
                [
                    len(
                        model(
                            G,
                            S | {v}
                        )
                    )
                    for _ in range(MC_RUNS)
                ]
            )

            if spread > best_spread:

                best = v
                best_spread = spread

        S.add(best)

    return S



def generate_seed_sets():

    seed_records = []

    print()
    print("=" * 70)
    print("TARGET-ORIENTED GREEDY SEED SELECTION")
    print("=" * 70)
    print()

    for dataset_name, snapshots in datasets.items():

        print(
            f"Processing {dataset_name} ..."
        )

        Gagg = nx.compose_all(
            snapshots
        )

        for k in K_VALUES:

            for model_name, model_func in models.items():


                selection_seed = (
                    SEED
                    + k * 1000
                    + sum(
                        ord(c)
                        for c in dataset_name
                    ) * 10
                    + sum(
                        ord(c)
                        for c in model_name
                    )
                )

                random.seed(
                    selection_seed
                )

                np.random.seed(
                    selection_seed
                )

                seeds = greedy(
                    Gagg,
                    k,
                    model_func
                )

                for seed in sorted(seeds):

                    seed_records.append({

                        "Dataset":
                            dataset_name,

                        "k":
                            k,

                        "Model":
                            model_name,

                        "SeedNode":
                            seed

                    })

                print(
                    f"  {dataset_name} | "
                    f"k={k} | "
                    f"{model_name} | "
                    f"{len(seeds)} seeds"
                )

    seed_df = pd.DataFrame(
        seed_records
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "selected_seed_sets.csv"
    )

    seed_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Seed sets saved to:\n"
        f"{output_file}"
    )

    return seed_df



def run_single_experiment(args):

    (
        dataset_name,
        k,
        model_name,
        run_id,
        seeds,
        snapshots
    ) = args


    run_seed = (
        SEED
        + run_id * 1000000
        + k * 10000
        + sum(
            ord(c)
            for c in dataset_name
        ) * 100
        + sum(
            ord(c)
            for c in model_name
        )
    )

    random.seed(
        run_seed
    )

    np.random.seed(
        run_seed
    )

    model_func = models[
        model_name
    ]


    cumulative_active = set(
        seeds
    )

    snapshot_activations = []


    for G in snapshots:

        cumulative_active |= model_func(
            G,
            cumulative_active
        )

        snapshot_activations.append(
            len(cumulative_active)
        )

    return {

        "Dataset":
            dataset_name,

        "k":
            k,

        "Model":
            model_name,

        "Run":
            run_id,

        "SnapshotActivations":
            snapshot_activations,

        "FinalActivation":
            len(cumulative_active)

    }



def create_tasks(seed_df):

    tasks = []

    for dataset_name, snapshots in datasets.items():

        for k in K_VALUES:

            for model_name in models:

                model_seed_rows = seed_df[
                    (
                        seed_df["Dataset"]
                        ==
                        dataset_name
                    )
                    &
                    (
                        seed_df["k"]
                        ==
                        k
                    )
                    &
                    (
                        seed_df["Model"]
                        ==
                        model_name
                    )
                ]

                seeds = set(
                    model_seed_rows[
                        "SeedNode"
                    ].tolist()
                )

                for run_id in range(
                    1,
                    N_RUNS + 1
                ):

                    tasks.append(
                        (
                            dataset_name,
                            k,
                            model_name,
                            run_id,
                            seeds,
                            snapshots
                        )
                    )

    return tasks



def run_all_experiments(tasks):

    total_runs = len(tasks)

    print()
    print("=" * 70)
    print("TARGET-ORIENTED MONTE CARLO EXPERIMENT")
    print("=" * 70)
    print(
        f"Datasets           : {len(datasets)}"
    )
    print(
        f"Models             : {len(models)}"
    )
    print(
        f"Seed sizes         : {len(K_VALUES)}"
    )
    print(
        f"Runs/model         : {N_RUNS}"
    )
    print(
        f"Total realizations : {total_runs}"
    )
    print("=" * 70)
    print()

    n_processes = max(
        1,
        mp.cpu_count() - 1
    )

    print(
        f"Using {n_processes} CPU processes..."
    )
    print()

    results = []

    with mp.Pool(
        processes=n_processes
    ) as pool:

        for i, result in enumerate(
            pool.imap_unordered(
                run_single_experiment,
                tasks
            ),
            start=1
        ):

            results.append(
                result
            )

            if (
                i % 100 == 0
                or i == total_runs
            ):

                print(
                    f"Completed "
                    f"{i}/{total_runs}"
                )

    return results



def save_raw_results(results):

    rows = []

    for result in results:

        for snapshot_id, activation in enumerate(
            result["SnapshotActivations"],
            start=1
        ):

            rows.append({

                "Dataset":
                    result["Dataset"],

                "k":
                    result["k"],

                "Model":
                    result["Model"],

                "Run":
                    result["Run"],

                "Snapshot":
                    snapshot_id,

                "Activation":
                    activation

            })

    raw_df = pd.DataFrame(
        rows
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "all_MC_results.csv"
    )

    raw_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Raw Monte Carlo data saved to:\n"
        f"{output_file}"
    )

    return raw_df



def calculate_statistics(raw_df):

    stats = (
        raw_df
        .groupby(
            [
                "Dataset",
                "k",
                "Model",
                "Snapshot"
            ]
        )["Activation"]
        .agg(
            N="count",
            Mean="mean",
            SD="std"
        )
        .reset_index()
    )


    stats["SEM"] = (
        stats["SD"]
        /
        np.sqrt(
            stats["N"]
        )
    )


    stats["CI95_Lower"] = (
        stats["Mean"]
        -
        1.96 * stats["SEM"]
    )

    stats["CI95_Upper"] = (
        stats["Mean"]
        +
        1.96 * stats["SEM"]
    )

    stats["CI95_Error"] = (
        1.96 * stats["SEM"]
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "all_MC_statistics.csv"
    )

    stats.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Statistical results saved to:\n"
        f"{output_file}"
    )

    return stats



def calculate_final_summary(raw_df):

    final_df = raw_df[
        raw_df["Snapshot"]
        ==
        T_SNAPSHOTS
    ].copy()

    summary = (
        final_df
        .groupby(
            [
                "Dataset",
                "k",
                "Model"
            ]
        )["Activation"]
        .agg(
            N="count",
            MeanFinalActivation="mean",
            SD="std"
        )
        .reset_index()
    )


    summary["SEM"] = (
        summary["SD"]
        /
        np.sqrt(
            summary["N"]
        )
    )


    summary["CI95_Lower"] = (
        summary["MeanFinalActivation"]
        -
        1.96 * summary["SEM"]
    )

    summary["CI95_Upper"] = (
        summary["MeanFinalActivation"]
        +
        1.96 * summary["SEM"]
    )

    summary["CI95_Error"] = (
        1.96 * summary["SEM"]
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "final_activation_summary.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Final activation summary saved to:\n"
        f"{output_file}"
    )

    return summary



def print_final_table(summary):

    print()
    print("=" * 120)
    print("FINAL ACTIVATION SUMMARY")
    print("=" * 120)

    display_columns = [

        "Dataset",

        "k",

        "Model",

        "N",

        "MeanFinalActivation",

        "SD",

        "CI95_Lower",

        "CI95_Upper"

    ]

    print(
        summary[
            display_columns
        ]
        .sort_values(
            [
                "Dataset",
                "k",
                "Model"
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)



def main():

    print()
    print("=" * 70)
    print(
        "REVIEWER #1 STATISTICAL UNCERTAINTY"
    )
    print(
        "TARGET-ORIENTED DIFFUSION MODELS"
    )
    print("=" * 70)
    print()


    seed_df = generate_seed_sets()


    tasks = create_tasks(
        seed_df
    )


    results = run_all_experiments(
        tasks
    )


    raw_df = save_raw_results(
        results
    )


    stats_df = calculate_statistics(
        raw_df
    )


    summary_df = calculate_final_summary(
        raw_df
    )


    print_final_table(
        summary_df
    )


    print()
    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)

    print(
        os.path.join(
            OUTPUT_DIR,
            "selected_seed_sets.csv"
        )
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "all_MC_results.csv"
        )
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "all_MC_statistics.csv"
        )
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "final_activation_summary.csv"
        )
    )

    print("=" * 70)

    print(
        "TARGET-ORIENTED EXPERIMENT COMPLETED."
    )

    print("=" * 70)



if __name__ == "__main__":
    main()
