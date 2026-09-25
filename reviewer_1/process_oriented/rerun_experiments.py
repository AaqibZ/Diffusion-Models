
import gzip
import csv
import random
import multiprocessing as mp
import os

import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



T_SNAPSHOTS = 10

K_VALUES = [
    8,
    12,
    16,
    20
]

MAX_STEPS = 8


SEED = 42



N_RUNS = 1000


N_WORKERS = max(
    1,
    mp.cpu_count() - 1
)


Z_95 = 1.96




STOCHASTIC_MODELS = [
    "IC",
    "SI",
    "SIR",
    "SEIR",
    "SCIR",
    "FSIR",
    "irSIR"
]


DATASET_PATHS = {

    "CollegeMsg":
        (
            "txt",
            "Datasets/CollegeMsg.txt.gz"
        ),

    "Email-Eu-Core":
        (
            "txt",
            "Datasets/email-Eu-core-temporal.txt.gz"
        ),

    "Bitcoin-OTC":
        (
            "csv",
            "Datasets/soc-sign-bitcoinotc.csv.gz"
        )

}




def load_snap_txt_gz(path):

    """
    Load SNAP temporal edge-list file.

    Expected format:
        source destination timestamp
    """

    edges = []

    skipped = 0

    with gzip.open(
        path,
        "rt"
    ) as f:

        for line in f:

            if (
                line.startswith("#")
                or not line.strip()
            ):
                continue

            parts = line.strip().split()

            if len(parts) < 3:

                skipped += 1

                continue

            u, v, t = parts[:3]

            try:

                u = int(u)
                v = int(v)
                t = int(float(t))

                edges.append(
                    (
                        u,
                        v,
                        t
                    )
                )

            except ValueError:

                skipped += 1

    if skipped:

        print(
            f"[WARNING] "
            f"Skipped {skipped} malformed "
            f"lines in {path}"
        )

    return edges


def load_snap_csv_gz(
    path,
    min_positive=1
):

    """
    Load SNAP signed-network CSV file.

    Expected columns:
        source, target, rating, timestamp

    Only positive ratings >= min_positive
    are retained.
    """

    edges = []

    skipped = 0

    with gzip.open(
        path,
        "rt"
    ) as f:

        reader = csv.reader(f)

        for row in reader:

            if (
                not row
                or row[0].startswith("#")
                or len(row) < 4
            ):

                continue

            u, v, rating, t = row[:4]

            try:

                if int(rating) < min_positive:

                    continue

                edges.append(
                    (
                        int(u),
                        int(v),
                        int(float(t))
                    )
                )

            except ValueError:

                skipped += 1

    if skipped:

        print(
            f"[WARNING] "
            f"Skipped {skipped} malformed "
            f"rows in {path}"
        )

    return edges




def build_time_snapshots(
    edges,
    n_snapshots=T_SNAPSHOTS
):

   

    if not edges:

        return [
            nx.DiGraph()
            for _ in range(n_snapshots)
        ]

   

    all_nodes = (
        {
            u
            for u, v, _ in edges
        }
        |
        {
            v
            for u, v, _ in edges
        }
    )

   

    times = [
        t
        for _, _, t in edges
    ]

    tmin = min(times)

    tmax = max(times) + 1

    interval = (
        (tmax - tmin)
        / n_snapshots
    )

    snapshots = []

  

    for i in range(n_snapshots):

        G = nx.DiGraph()

        G.add_nodes_from(
            all_nodes
        )

        snap_start = (
            tmin
            +
            i * interval
        )

        snap_end = (
            tmin
            +
            (i + 1) * interval
        )

        for u, v, t in edges:

            if (
                snap_start <= t
                and t < snap_end
            ):

                G.add_edge(
                    u,
                    v
                )

        snapshots.append(G)

    return snapshots




def load_all_datasets():

    datasets = {}

    for (
        dataset_name,
        (
            file_type,
            path
        )
    ) in DATASET_PATHS.items():

        print(
            f"\nLoading {dataset_name}..."
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nDataset file not found:\n"
                f"{path}\n\n"
                f"Please check the Datasets/ directory."
            )

        if file_type == "txt":

            edges = load_snap_txt_gz(
                path
            )

        elif file_type == "csv":

            edges = load_snap_csv_gz(
                path
            )

        else:

            raise ValueError(
                f"Unknown file type: {file_type}"
            )

        print(
            f"  Edges loaded: {len(edges):,}"
        )

        snapshots = build_time_snapshots(
            edges,
            T_SNAPSHOTS
        )

        print(
            f"  Snapshots: {len(snapshots)}"
        )

        datasets[
            dataset_name
        ] = snapshots

    return datasets




def IC(
    G,
    seeds,
    p=0.2
):

    

    active = set(
        seeds
    )

    frontier = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for u in frontier:

            for v in G.successors(u):

                if (
                    v not in active
                    and random.random() < p
                ):

                    new.add(v)

        if not new:

            break

        active |= new

        frontier = new

    return active


def SI(
    G,
    seeds,
    beta=0.3
):

    

    infected = set(
        seeds
    )

    frontier = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for u in frontier:

            for v in G.successors(u):

                if (
                    v not in infected
                    and random.random() < beta
                ):

                    new.add(v)

        if not new:

            break

        infected |= new

        frontier = new

    return infected


def SIR(
    G,
    seeds,
    beta=0.3,
    gamma=0.2
):

  

    I = set(
        seeds
    )

    R = set()

    frontier = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for u in frontier:

            for v in G.successors(u):

                if (
                    v not in I
                    and v not in R
                    and random.random() < beta
                ):

                    new.add(v)

            if random.random() < gamma:

                R.add(u)

        if not new:

            break

        I |= new

        frontier = new

    return I | R


def SEIR(
    G,
    seeds,
    beta=0.3,
    sigma=0.3,
    gamma=0.2
):

    

    S = (
        set(G.nodes())
        -
        set(seeds)
    )

    E = set(
        seeds
    )

    I = set()

    R = set()

    for _ in range(
        MAX_STEPS
    ):

        newE = set()

        newI = set()

        newR = set()

      

        for u in I:

            for v in G.successors(u):

                if (
                    v in S
                    and random.random() < beta
                ):

                    newE.add(v)

            if random.random() < gamma:

                newR.add(u)

       

        for u in E:

            if random.random() < sigma:

                newI.add(u)

      

        if not (
            newE
            or newI
            or newR
        ):

            break

    

        S -= newE

        E |= newE

        E -= newI

        I |= newI

        I -= newR

        R |= newR

    return E | I | R


def SCIR(
    G,
    seeds,
    beta=0.3
):

   

    infected = set(
        seeds
    )

    frontier = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for u in frontier:

            for v in G.successors(u):

                prob = (
                    beta
                    /
                    max(
                        1,
                        G.in_degree(v)
                    )
                )

                if (
                    v not in infected
                    and random.random() < prob
                ):

                    new.add(v)

        if not new:

            break

        infected |= new

        frontier = new

    return infected


def FSIR(
    G,
    seeds,
    beta=0.25,
    gamma=0.1
):

    

    I = set(
        seeds
    )

    R = set()

    frontier = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for u in frontier:

            for v in G.successors(u):

                if (
                    v not in I
                    and v not in R
                    and random.random()
                    <
                    beta / 2
                ):

                    new.add(v)

            if random.random() < gamma:

                R.add(u)

        if not new:

            break

        I |= new

        frontier = new

    return I | R


def irSIR(
    G,
    seeds,
    beta=0.25,
    gamma=0.1
):

   

    I = set(
        seeds
    )

    R = set()

    frontier = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for u in frontier:

            for v in G.successors(u):

                if random.random() < beta:

                    new.add(v)

            if random.random() < gamma:

                R.add(u)

        if not new:

            break

        I |= new

        frontier = new

    return I | R




models = {

    "IC": IC,

    "SI": SI,

    "SIR": SIR,

    "SEIR": SEIR,

    "SCIR": SCIR,

    "FSIR": FSIR,

    "irSIR": irSIR
}




def validate_models():

    missing = [
        model
        for model in STOCHASTIC_MODELS
        if model not in models
    ]

    if missing:

        raise RuntimeError(
            "The following required stochastic "
            f"models are missing: {missing}"
        )

    if len(STOCHASTIC_MODELS) != 7:

        raise RuntimeError(
            "Exactly seven stochastic models "
            "must be included."
        )




def greedy(
    G,
    k,
    seed
):

  

    python_state = random.getstate()
    numpy_state = np.random.get_state()


    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    seeds = set()

    nodes = list(
        G.nodes()
    )

    for _ in range(k):

        best_node = None

        best_val = -1

        for v in nodes:

            if v in seeds:

                continue

            spread = len(
                IC(
                    G,
                    seeds | {v}
                )
            )

          

            if (
                spread > best_val
                or
                (
                    spread == best_val
                    and
                    (
                        best_node is None
                        or
                        v < best_node
                    )
                )
            ):

                best_node = v

                best_val = spread

        if best_node is None:

            break

        seeds.add(
            best_node
        )



    random.setstate(
        python_state
    )

    np.random.set_state(
        numpy_state
    )

    return seeds



_worker_snapshots = None


def _init_worker(
    snapshots
):

  

    global _worker_snapshots

    _worker_snapshots = snapshots


def _mc_run(
    args
):


    (
        model_name,
        seeds,
        run_seed
    ) = args



    random.seed(
        int(run_seed)
    )

    np.random.seed(
        int(run_seed)
    )

    func = models[
        model_name
    ]


    cumulative = set(
        seeds
    )

    spread = np.zeros(
        T_SNAPSHOTS,
        dtype=float
    )

    

    for t, G in enumerate(
        _worker_snapshots
    ):

        new_active = func(
            G,
            cumulative
        )

        cumulative |= new_active

        spread[t] = len(
            cumulative
        )

    return (
        model_name,
        spread
    )




def calculate_statistics(
    mc_runs
):

    """
    Calculate:

        mean
        SD
        SEM
        95% CI

    from actual Monte Carlo realizations.

    Input shape:
        (N_RUNS, T_SNAPSHOTS)
    """

    n = mc_runs.shape[0]

    if n < 2:

        raise ValueError(
            
        )

  

    mean = np.mean(
        mc_runs,
        axis=0
    )

  

    sd = np.std(
        mc_runs,
        axis=0,
        ddof=1
    )

   

    sem = (
        sd
        /
        np.sqrt(n)
    )


    margin = (
        Z_95
        *
        sem
    )

    lower = (
        mean
        -
        margin
    )

    upper = (
        mean
        +
        margin
    )

    

    lower = np.maximum(
        lower,
        0
    )

    return {

        "n": n,

        "mean": mean,

        "sd": sd,

        "sem": sem,

        "lower": lower,

        "upper": upper,

        "margin": margin
    }


def save_statistics_csv(
    all_stats,
    filename
):

    rows = []

    for (
        dataset_name,
        model_name,
        k
    ), stats in all_stats.items():

        for t in range(
            T_SNAPSHOTS
        ):

            rows.append({

                "Dataset":
                    dataset_name,

                "Model":
                    model_name,

                "Seed_Size":
                    k,

                "Snapshot":
                    t + 1,

                "N_MC":
                    stats["n"],

                "Mean":
                    stats["mean"][t],

                "SD":
                    stats["sd"][t],

                "SEM":
                    stats["sem"][t],

                "CI95_Lower":
                    stats["lower"][t],

                "CI95_Upper":
                    stats["upper"][t],

                "CI95_Margin":
                    stats["margin"][t]
            })

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        filename,
        index=False
    )

    print(
        f"\nSaved snapshot statistics:\n"
        f"{filename}"
    )



def save_final_summary(
    all_stats,
    filename
):

    rows = []

    for (
        dataset_name,
        model_name,
        k
    ), stats in all_stats.items():

        final_mean = (
            stats["mean"][-1]
        )

        final_sd = (
            stats["sd"][-1]
        )

        final_sem = (
            stats["sem"][-1]
        )

        final_lower = (
            stats["lower"][-1]
        )

        final_upper = (
            stats["upper"][-1]
        )

        final_margin = (
            stats["margin"][-1]
        )

        rows.append({

            "Dataset":
                dataset_name,

            "Model":
                model_name,

            "Seed_Size":
                k,

            "N_MC":
                stats["n"],

            "Final_Mean_Activation":
                final_mean,

            "Final_SD":
                final_sd,

            "Final_SEM":
                final_sem,

            "CI95_Lower":
                final_lower,

            "CI95_Upper":
                final_upper,

            "CI95_Margin":
                final_margin
        })

    df = pd.DataFrame(
        rows
    )

 

    model_order = {
        model: i
        for i, model
        in enumerate(
            STOCHASTIC_MODELS
        )
    }

    df["_model_order"] = (
        df["Model"]
        .map(model_order)
    )

    df = df.sort_values(
        [
            "Dataset",
            "Seed_Size",
            "_model_order"
        ]
    )

    df = df.drop(
        columns=[
            "_model_order"
        ]
    )

    df.to_csv(
        filename,
        index=False
    )

    print(
        f"\nSaved final activation summary:\n"
        f"{filename}"
    )


def save_seed_sets(
    seed_records,
    filename
):

    rows = []

    for record in seed_records:

        rows.append(record)

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        filename,
        index=False
    )

    print(
        f"\nSaved seed sets:\n"
        f"{filename}"
    )



def plot_ci_bands(
    dataset_name,
    k,
    stats_for_k,
    output_dir
):

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    snapshots = np.arange(
        1,
        T_SNAPSHOTS + 1
    )

    for model_name in STOCHASTIC_MODELS:

        stats = (
            stats_for_k[
                model_name
            ]
        )

        mean = (
            stats["mean"]
        )

        lower = (
            stats["lower"]
        )

        upper = (
            stats["upper"]
        )

      

        ax.plot(
            snapshots,
            mean,
            linewidth=1.5,
            label=model_name
        )

      

        ax.fill_between(
            snapshots,
            lower,
            upper,
            alpha=0.15
        )

    ax.set_xlabel(
        "Temporal Snapshot"
    )

    ax.set_ylabel(
        "Activated Nodes"
    )

    ax.set_title(
        f"{dataset_name}: "
        f"Mean Diffusion with 95% CI "
        f"(k={k})"
    )

    ax.set_xticks(
        snapshots
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        fontsize=8,
        loc="best"
    )

    fig.tight_layout()

    filename = os.path.join(
        output_dir,
        f"{dataset_name}_k{k}_CI_trajectory.pdf"
    )

    fig.savefig(
        filename,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved CI trajectory plot:\n"
        f"{filename}"
    )



def plot_final_error_bars(
    dataset_name,
    k,
    stats_for_k,
    output_dir
):

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    x = np.arange(
        len(
            STOCHASTIC_MODELS
        )
    )

    means = []

    lower_errors = []

    upper_errors = []

    for model_name in STOCHASTIC_MODELS:

        stats = (
            stats_for_k[
                model_name
            ]
        )

        mean = (
            stats["mean"][-1]
        )

        lower = (
            stats["lower"][-1]
        )

        upper = (
            stats["upper"][-1]
        )

        means.append(
            mean
        )

        lower_errors.append(
            mean - lower
        )

        upper_errors.append(
            upper - mean
        )

    means = np.asarray(
        means
    )

    errors = np.vstack(
        [
            lower_errors,
            upper_errors
        ]
    )

    

    hatch_patterns = [
        "",
        "//",
        "\\\\",
        "xx",
        "..",
        "--",
        "++"
    ]

    bars = ax.bar(
        x,
        means,
        width=0.72,
        color="white",
        edgecolor="black",
        linewidth=1.0
    )

    for bar, hatch in zip(
        bars,
        hatch_patterns
    ):

        bar.set_hatch(
            hatch
        )


    ax.errorbar(
        x,
        means,
        yerr=errors,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=5
    )

    ax.set_xlabel(
        "Diffusion Model"
    )

    ax.set_ylabel(
        "Final Activated Nodes"
    )

    ax.set_title(
        f"{dataset_name}: "
        f"Final Activation with 95% CI "
        f"(k={k})"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        STOCHASTIC_MODELS
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    filename = os.path.join(
        output_dir,
        f"{dataset_name}_k{k}_final_CI_bars.pdf"
    )

    fig.savefig(
        filename,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved final CI bar plot:\n"
        f"{filename}"
    )



def plot_final_ci_magnified(
    dataset_name,
    k,
    stats_for_k,
    output_dir
):



    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    x = np.arange(
        len(
            STOCHASTIC_MODELS
        )
    )

    means = []

    lower_dev = []

    upper_dev = []

    for model_name in STOCHASTIC_MODELS:

        stats = (
            stats_for_k[
                model_name
            ]
        )

        mean = (
            stats["mean"][-1]
        )

        lower = (
            stats["lower"][-1]
        )

        upper = (
            stats["upper"][-1]
        )

        means.append(
            mean
        )

        lower_dev.append(
            mean - lower
        )

        upper_dev.append(
            upper - mean
        )

    means = np.asarray(
        means
    )

    lower_dev = np.asarray(
        lower_dev
    )

    upper_dev = np.asarray(
        upper_dev
    )

    

    ax.axhline(
        0,
        linewidth=1.0
    )


    for i in range(
        len(STOCHASTIC_MODELS)
    ):

        ax.errorbar(
            x[i],
            0,
            yerr=[
                [lower_dev[i]],
                [upper_dev[i]]
            ],
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=5,
            capthick=1.2
        )

    ax.set_xlabel(
        "Diffusion Model"
    )

    ax.set_ylabel(
        "95% CI deviation from mean (nodes)"
    )

    ax.set_title(
        f"{dataset_name}: "
        f"95% CI Width at Final Snapshot "
        f"(k={k})"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        STOCHASTIC_MODELS
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    filename = os.path.join(
        output_dir,
        f"{dataset_name}_k{k}_CI_magnified.pdf"
    )

    fig.savefig(
        filename,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved magnified CI plot:\n"
        f"{filename}"
    )



def run_dataset_k_experiment(
    dataset_name,
    snapshots,
    k,
    pool,
    output_dir
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"DATASET: {dataset_name}"
    )

    print(
        f"SEED SIZE: k = {k}"
    )

    print(
        "=" * 70
    )

   

    Gagg = nx.DiGraph()

    for G in snapshots:

        Gagg.add_nodes_from(
            G.nodes()
        )

        Gagg.add_edges_from(
            G.edges()
        )

    print(
        f"Aggregate graph: "
        f"{Gagg.number_of_nodes():,} nodes, "
        f"{Gagg.number_of_edges():,} edges"
    )

 

    greedy_seed = (
        SEED
        +
        hash(dataset_name) % 100000
        +
        k * 1000
    )


    dataset_index = list(
        DATASET_PATHS.keys()
    ).index(
        dataset_name
    )

    greedy_seed = (
        SEED
        +
        dataset_index * 100000
        +
        k * 1000
    )

    seeds = greedy(
        Gagg,
        k,
        greedy_seed
    )

    print(
        f"Selected {len(seeds)} seeds."
    )

    if len(seeds) != k:

        raise RuntimeError(
            f"Requested k={k}, "
            f"but only {len(seeds)} seeds "
            f"were selected."
        )

   

    seed_record = {

        "Dataset":
            dataset_name,

        "Seed_Size":
            k,

        "Seed_Nodes":
            ";".join(
                map(
                    str,
                    sorted(seeds)
                )
            )
    }

   

    tasks = []

    for model_index, model_name in enumerate(
        STOCHASTIC_MODELS
    ):

        for run_index in range(
            N_RUNS
        ):

            

            run_seed = (
                SEED
                +
                dataset_index * 1_000_000_000
                +
                k * 10_000_000
                +
                model_index * 1_000_000
                +
                run_index
            )

            tasks.append(
                (
                    model_name,
                    tuple(seeds),
                    run_seed
                )
            )

    total_tasks = len(
        tasks
    )

    print(
        f"\nMonte Carlo simulations:"
    )

    print(
        f"  Models       : "
        f"{len(STOCHASTIC_MODELS)}"
    )

    print(
        f"  Runs/model   : "
        f"{N_RUNS:,}"
    )

    print(
        f"  Total runs   : "
        f"{total_tasks:,}"
    )

    

    print(
        "\nStarting Monte Carlo simulations..."
    )

    results = pool.imap_unordered(
        _mc_run,
        tasks,
        chunksize=20
    )

   

    mc_runs = {

        model_name:
        np.zeros(
            (
                N_RUNS,
                T_SNAPSHOTS
            ),
            dtype=float
        )

        for model_name
        in STOCHASTIC_MODELS
    }

    counters = {

        model_name: 0

        for model_name
        in STOCHASTIC_MODELS
    }


    completed = 0

    for model_name, spread in results:

        index = counters[
            model_name
        ]

        mc_runs[
            model_name
        ][index] = spread

        counters[
            model_name
        ] += 1

        completed += 1

       
        if (
            completed % 10000 == 0
            or
            completed == total_tasks
        ):

            print(
                f"  Completed "
                f"{completed:,} / "
                f"{total_tasks:,}"
            )

   

    for model_name in STOCHASTIC_MODELS:

        if (
            counters[model_name]
            != N_RUNS
        ):

            raise RuntimeError(
                f"{model_name}: "
                f"Expected {N_RUNS} runs but "
                f"received {counters[model_name]}."
            )

   

    stats_for_k = {}

    for model_name in STOCHASTIC_MODELS:

        stats = calculate_statistics(
            mc_runs[
                model_name
            ]
        )

        stats_for_k[
            model_name
        ] = stats

        print(
            f"\n{model_name}"
        )

        print(
            f"  Final mean = "
            f"{stats['mean'][-1]:.4f}"
        )

        print(
            f"  Final SD   = "
            f"{stats['sd'][-1]:.4f}"
        )

        print(
            f"  Final SEM  = "
            f"{stats['sem'][-1]:.4f}"
        )

        print(
            f"  Final 95% CI = "
            f"["
            f"{stats['lower'][-1]:.4f}, "
            f"{stats['upper'][-1]:.4f}"
            f"]"
        )

    return (
        stats_for_k,
        seed_record
    )




def main():

    

    validate_models()

  

    output_dir = (
        "Reviewer1_Uncertainty_Results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

   

    print("\n")

    print(
        "=" * 70
    )

    print(
        "REVIEWER #1 "
        "STATISTICAL UNCERTAINTY EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Models       : "
        f"{', '.join(STOCHASTIC_MODELS)}"
    )

    print(
        f"Monte Carlo  : "
        f"{N_RUNS:,} runs/model"
    )

    print(
        f"Datasets     : "
        f"{len(DATASET_PATHS)}"
    )

    print(
        f"Snapshots    : "
        f"{T_SNAPSHOTS}"
    )

    print(
        f"Seed sizes   : "
        f"{K_VALUES}"
    )

    print(
        f"Workers      : "
        f"{N_WORKERS}"
    )

    print(
        f"Confidence   : "
        f"95%"
    )

    print(
        "=" * 70
    )

   

    datasets = load_all_datasets()

  

    all_stats = {}

    seed_records = []


    first_dataset = next(
        iter(datasets.values())
    )

    with mp.Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
        initargs=(
            first_dataset,
        )
    ) as pool:

        

        for dataset_name, snapshots in datasets.items():

            print(
                "\n"
                + "#" * 70
            )

            print(
                f"# DATASET: {dataset_name}"
            )

            print(
                "#" * 70
            )

          

            pool.close()

            pool.join()

            pool = mp.Pool(
                processes=N_WORKERS,
                initializer=_init_worker,
                initargs=(
                    snapshots,
                )
            )

           

            for k in K_VALUES:

                (
                    stats_for_k,
                    seed_record
                ) = run_dataset_k_experiment(
                    dataset_name,
                    snapshots,
                    k,
                    pool,
                    output_dir
                )

                seed_records.append(
                    seed_record
                )

               

                for model_name in STOCHASTIC_MODELS:

                    key = (
                        dataset_name,
                        model_name,
                        k
                    )

                    all_stats[
                        key
                    ] = stats_for_k[
                        model_name
                    ]

               

                plot_ci_bands(
                    dataset_name,
                    k,
                    stats_for_k,
                    output_dir
                )

               

                plot_final_error_bars(
                    dataset_name,
                    k,
                    stats_for_k,
                    output_dir
                )

               

                plot_final_ci_magnified(
                    dataset_name,
                    k,
                    stats_for_k,
                    output_dir
                )




    save_statistics_csv(
        all_stats,
        os.path.join(
            output_dir,
            "all_MC_statistics.csv"
        )
    )

    save_final_summary(
        all_stats,
        os.path.join(
            output_dir,
            "final_activation_summary.csv"
        )
    )

    save_seed_sets(
        seed_records,
        os.path.join(
            output_dir,
            "selected_seed_sets.csv"
        )
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "EXPERIMENT COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Results saved in:\n"
        f"{output_dir}/"
    )

    print(
        "\nFiles generated:"
    )

    print(
        "  1. all_MC_statistics.csv"
    )

    print(
        "  2. final_activation_summary.csv"
    )

    print(
        "  3. selected_seed_sets.csv"
    )

    print(
        "  4. *_CI_trajectory.pdf"
    )

    print(
        "  5. *_final_CI_bars.pdf"
    )

    print(
        "  6. *_CI_magnified.pdf"
    )

    print(
        "\nThe reported SDs and 95% CIs are "
        "computed from the actual Monte Carlo "
        "realizations."
    )



if __name__ == "__main__":

   

    mp.freeze_support()

    main()
