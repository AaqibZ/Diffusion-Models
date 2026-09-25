
import os
import gzip
import csv
import random
import multiprocessing as mp

import networkx as nx
import numpy as np
import pandas as pd



OUTPUT_DIR = "Reviewer1_Structure_Uncertainty_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)



T_SNAPSHOTS = 10
K_VALUES = [8, 12, 16, 20]
MAX_STEPS = 6

N_RUNS = 1000

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


def load_snap_csv_gz(path, min_positive=1):
    edges = []

    with gzip.open(path, "rt") as f:
        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            if row[0].startswith("#") or len(row) < 4:
                continue

            u, v, rating, t = row[:4]

            if int(rating) >= min_positive:
                edges.append(
                    (
                        int(u),
                        int(v),
                        int(float(t))
                    )
                )

    return edges


def build_time_snapshots(edges):

    if not edges:
        return [
            nx.DiGraph()
            for _ in range(T_SNAPSHOTS)
        ]

    nodes = (
        set(u for u, v, _ in edges)
        |
        set(v for u, v, _ in edges)
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

    snapshots = []

    for i in range(T_SNAPSHOTS):

        G = nx.DiGraph()
        G.add_nodes_from(nodes)

        lower = tmin + i * interval
        upper = tmin + (i + 1) * interval

        for u, v, t in edges:

            if lower <= t < upper:
                G.add_edge(u, v)

        snapshots.append(G)

    return snapshots



datasets = {

    "CollegeMsg":
        build_time_snapshots(
            load_snap_txt_gz(
                "Datasets/CollegeMsg.txt.gz"
            )
        ),

    "Email-Eu-Core":
        build_time_snapshots(
            load_snap_txt_gz(
                "Datasets/email-Eu-core-temporal.txt.gz"
            )
        ),

    "Bitcoin-OTC":
        build_time_snapshots(
            load_snap_csv_gz(
                "Datasets/soc-sign-bitcoinotc.csv.gz"
            )
        ),
}




def ProductAdopter(G, seeds):

    active = set(seeds)

    for _ in range(MAX_STEPS):

        new = set()

        for v in G.nodes():

            if v not in active:

                if random.random() < (
                    0.2
                    *
                    sum(
                        u in active
                        for u in G.predecessors(v)
                    )
                ):
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

            if (
                v not in active
                and random.random() < density
            ):
                active.add(v)

    return active


def BassDiscrete(G, seeds, p=0.02, q=0.4):

    active = set(seeds)

    for _ in range(MAX_STEPS):

        for v in G.nodes():

            if v not in active:

                f = (
                    sum(
                        u in active
                        for u in G.predecessors(v)
                    )
                    /
                    max(
                        1,
                        G.in_degree(v)
                    )
                )

                if random.random() < (
                    p + q * f
                ):
                    active.add(v)

    return active


def ABBM(G, seeds):

    active = set(seeds)

    for _ in range(MAX_STEPS):

        for v in G.nodes():

            if v not in active:

                if random.random() < (
                    0.05
                    +
                    0.3
                    *
                    sum(
                        u in active
                        for u in G.predecessors(v)
                    )
                ):
                    active.add(v)

    return active



models = {

    "PAM":
        ProductAdopter,

    "DensityBased":
        DensityMacro,

    "BIADM":
        BassDiscrete,

    "ABBM":
        ABBM,
}



def greedy(G, k):

    return set(
        v
        for v, _ in sorted(
            G.degree,
            key=lambda x: x[1],
            reverse=True
        )[:k]
    )



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
        + run_id * 100000
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

    random.seed(run_seed)
    np.random.seed(run_seed)

    model_func = models[model_name]


    cumulative_active = set(seeds)

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
            len(cumulative_active),
    }



def create_tasks():

    tasks = []

    for dataset_name, snapshots in datasets.items():

        Gagg = nx.compose_all(snapshots)

        for k in K_VALUES:

            seeds = greedy(
                Gagg,
                k
            )

            print(
                f"{dataset_name} | "
                f"k={k} | "
                f"Seeds={len(seeds)}"
            )

            for model_name in models:

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
    print("STRUCTURE-ORIENTED DIFFUSION")
    print("MONTE CARLO EXPERIMENT")
    print("=" * 70)
    print(
        f"Datasets          : {len(datasets)}"
    )
    print(
        f"Models            : {len(models)}"
    )
    print(
        f"Seed sizes        : {len(K_VALUES)}"
    )
    print(
        f"Runs/model        : {N_RUNS}"
    )
    print(
        f"Total realizations: {total_runs}"
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

            results.append(result)

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
                    activation,

            })

    raw_df = pd.DataFrame(rows)

    output_file = os.path.join(
        OUTPUT_DIR,
        "all_MC_results.csv"
    )

    raw_df.to_csv(
        output_file,
        index=False
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
        np.sqrt(stats["N"])
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

    return stats



def calculate_final_summary(raw_df):


    final_df = raw_df[
        raw_df["Snapshot"] == T_SNAPSHOTS
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
        np.sqrt(summary["N"])
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

    return summary



def save_seed_sets():

    rows = []

    for dataset_name, snapshots in datasets.items():

        Gagg = nx.compose_all(
            snapshots
        )

        for k in K_VALUES:

            seeds = greedy(
                Gagg,
                k
            )

            for seed in sorted(seeds):

                rows.append({

                    "Dataset":
                        dataset_name,

                    "k":
                        k,

                    "SeedNode":
                        seed,

                })

    seed_df = pd.DataFrame(rows)

    output_file = os.path.join(
        OUTPUT_DIR,
        "selected_seed_sets.csv"
    )

    seed_df.to_csv(
        output_file,
        index=False
    )

    return seed_df



def print_final_table(summary):

    print()
    print("=" * 100)
    print("FINAL ACTIVATION SUMMARY")
    print("=" * 100)

    display_columns = [

        "Dataset",

        "k",

        "Model",

        "N",

        "MeanFinalActivation",

        "SD",

        "CI95_Lower",

        "CI95_Upper",

    ]

    print(
        summary[
            display_columns
        ].sort_values(
            [
                "Dataset",
                "k",
                "Model"
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 100)



def main():

    print()
    print("=" * 70)
    print(
        "REVIEWER #1 STATISTICAL UNCERTAINTY"
    )
    print(
        "STRUCTURE-ORIENTED MODELS"
    )
    print("=" * 70)
    print()


    print(
        "Generating seed sets..."
    )

    save_seed_sets()

    print(
        "Seed sets saved."
    )
    print()


    tasks = create_tasks()


    results = run_all_experiments(
        tasks
    )


    raw_df = save_raw_results(
        results
    )

    print()
    print(
        "Raw Monte Carlo data saved."
    )


    stats = calculate_statistics(
        raw_df
    )

    print(
        "Mean, SD and 95% CI statistics saved."
    )


    summary = calculate_final_summary(
        raw_df
    )

    print(
        "Final activation summary saved."
    )


    print_final_table(
        summary
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
        "STRUCTURE-ORIENTED EXPERIMENT COMPLETED."
    )
    print("=" * 70)



if __name__ == "__main__":
    main()
