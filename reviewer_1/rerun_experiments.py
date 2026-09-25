

import gzip
import csv
import random
import multiprocessing as mp
import os

import networkx as nx
import numpy as np
import pandas as pd



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

    "Voter",
    "EVM",
    "DVM",
    "OM-WTD",
    "TAM",
    "POE",
    "LabelCascade",
    "CTMM"

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

    Only positive ratings are retained.
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
            for _ in range(
                n_snapshots
            )
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
        /
        n_snapshots
    )

    snapshots = []

  

    for i in range(
        n_snapshots
    ):

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
                and
                t < snap_end
            ):

                G.add_edge(
                    u,
                    v
                )

        snapshots.append(
            G
        )

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
            f"  Edges loaded: "
            f"{len(edges):,}"
        )

        snapshots = build_time_snapshots(
            edges,
            T_SNAPSHOTS
        )

        print(
            f"  Snapshots: "
            f"{len(snapshots)}"
        )

        datasets[
            dataset_name
        ] = snapshots

    return datasets




def Voter(
    G,
    seeds,
    p=0.5
):

    """
    Voter Model.

    Each node randomly adopts a neighbor's opinion
    with probability p.
    """

    opinion = {

        v:
        (
            1
            if v in seeds
            else 0
        )

        for v in G.nodes()

    }

    for _ in range(
        MAX_STEPS
    ):

        new_opinion = (
            opinion.copy()
        )

        for v in G.nodes():

            nbrs = list(
                G.predecessors(v)
            )

            if (
                nbrs
                and
                random.random() < p
            ):

                u = random.choice(
                    nbrs
                )

                new_opinion[v] = (
                    opinion[u]
                )

        opinion = new_opinion

    return {

        v
        for v, o
        in opinion.items()
        if o == 1

    }




def EVM(
    G,
    seeds
):

    """
    Extended Voter Model.

    Nodes adopt an opinion based on the fraction
    of active predecessors.
    """

    opinion = {

        v:
        (
            1
            if v in seeds
            else 0
        )

        for v in G.nodes()

    }

    for _ in range(
        MAX_STEPS
    ):

        for v in G.nodes():

            nbrs = list(
                G.predecessors(v)
            )

            if nbrs:

                weight = (

                    sum(
                        opinion[u]
                        for u in nbrs
                    )
                    /
                    len(nbrs)

                )

                if (
                    random.random()
                    <
                    weight
                ):

                    opinion[v] = 1

    return {

        v
        for v, o
        in opinion.items()
        if o == 1

    }




def DVM(
    G,
    seeds
):

    """
    Dynamic Voter Model.

    Uses the Voter Model with dynamic neighbor
    adoption probability.
    """

    return Voter(
        G,
        seeds,
        p=0.6
    )




def OMWTD(
    G,
    seeds
):

    """
    OM-WTD Model.

    Voter-like diffusion with a waiting-time
    mechanism that slows adoption.
    """

    opinion = {

        v:
        (
            1
            if v in seeds
            else 0
        )

        for v in G.nodes()

    }

    for _ in range(
        MAX_STEPS // 2
    ):

        for v in G.nodes():

            if random.random() < 0.3:

                nbrs = list(
                    G.predecessors(v)
                )

                if nbrs:

                    opinion[v] = opinion[
                        random.choice(nbrs)
                    ]

    return {

        v
        for v, o
        in opinion.items()
        if o == 1

    }




def TAM(
    G,
    seeds
):

    """
    Topic Adoption Model.

    Probabilistic path-based adoption.
    """

    active = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for u in active:

            for v in G.successors(u):

                if (
                    v not in active
                    and
                    random.random() < 0.2
                ):

                    new.add(v)

        if not new:

            break

        active |= new

    return active



def POE(
    G,
    seeds
):

  

    active = set(
        seeds
    )

    bias = {

        v:
        random.random()

        for v in G.nodes()

    }

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for v in G.nodes():

            if v in active:

                continue

            preds = list(
                G.predecessors(v)
            )

            social = (

                len(
                    [
                        p
                        for p in preds
                        if p in active
                    ]
                )
                /
                max(
                    1,
                    len(preds)
                )

            )

            if (

                bias[v] * 0.5
                +
                social * 0.5

                >
                0.5

            ):

                new.add(v)

        if not new:

            break

        active |= new

    return active




def LabelCascade(
    G,
    seeds
):

    """
    Label Propagation + Cascade Diffusion.

    Neighbor labels are randomly propagated.
    """

    labels = {

        v: v

        for v in G.nodes()

    }

    for _ in range(
        MAX_STEPS
    ):

        for v in G.nodes():

            nbrs = list(
                G.predecessors(v)
            )

            if nbrs:

                labels[v] = labels[
                    random.choice(nbrs)
                ]

    return {

        v
        for v, l
        in labels.items()
        if l in seeds

    }




def CTMM(
    G,
    seeds
):

    """
    Continuous-Time Markov Model.

    Adoption depends probabilistically on
    node connectivity.
    """

    active = set(
        seeds
    )

    for _ in range(
        MAX_STEPS
    ):

        new = set()

        for v in G.nodes():

            if v in active:

                continue

            rate = (

                G.in_degree(v)
                /
                max(
                    1,
                    G.number_of_edges()
                )

            )

            if (
                random.random()
                <
                rate
            ):

                new.add(v)

        if not new:

            break

        active |= new

    return active




models = {

    "Voter":
        Voter,

    "EVM":
        EVM,

    "DVM":
        DVM,

    "OM-WTD":
        OMWTD,

    "TAM":
        TAM,

    "POE":
        POE,

    "LabelCascade":
        LabelCascade,

    "CTMM":
        CTMM

}




def validate_models():

    missing = [

        model
        for model
        in STOCHASTIC_MODELS

        if model not in models

    ]

    if missing:

        raise RuntimeError(

            "The following required stochastic "
            f"models are missing: {missing}"

        )

    if len(
        STOCHASTIC_MODELS
    ) != 8:

        raise RuntimeError(

            "Exactly eight stochastic interaction-"
            "oriented models must be included."

        )




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

                    and

                    random.random() < p

                ):

                    new.add(v)

        if not new:

            break

        active |= new

        frontier = new

    return active




def greedy(
    G,
    k,
    seed
):

    

    python_state = (
        random.getstate()
    )

    numpy_state = (
        np.random.get_state()
    )

    

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

    """
    Initialize each worker with the temporal snapshots.
    """

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

 

    n = mc_runs.shape[0]

    if n < 2:

        raise ValueError(

            "At least two Monte Carlo runs "
            "are required to calculate SD."

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

        "n":
            n,

        "mean":
            mean,

        "sd":
            sd,

        "sem":
            sem,

        "lower":
            lower,

        "upper":
            upper,

        "margin":
            margin

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

    df = pd.DataFrame(
        seed_records
    )

    df.to_csv(

        filename,

        index=False

    )

    print(

        f"\nSaved seed sets:\n"
        f"{filename}"

    )




def run_dataset_k_experiment(

    dataset_name,
    snapshots,
    k

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

    for (

        model_index,
        model_name

    ) in enumerate(
        STOCHASTIC_MODELS
    ):

        for run_index in range(
            N_RUNS
        ):

          

            run_seed = (

                SEED

                +

                dataset_index
                * 1_000_000_000

                +

                k
                * 10_000_000

                +

                model_index
                * 1_000_000

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
        "\nMonte Carlo simulations:"
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

        model_name:
        0

        for model_name
        in STOCHASTIC_MODELS

    }

  

    print(
        "\nStarting Monte Carlo simulations..."
    )

    with mp.Pool(

        processes=N_WORKERS,

        initializer=_init_worker,

        initargs=(
            snapshots,
        )

    ) as pool:

        results = pool.imap_unordered(

            _mc_run,

            tasks,

            chunksize=10

        )

        completed = 0

        for (

            model_name,
            spread

        ) in results:

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

                completed % 100 == 0

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

            counters[
                model_name
            ]
            !=
            N_RUNS

        ):

            raise RuntimeError(

                f"{model_name}: "
                f"Expected {N_RUNS} runs but "
                f"received "
                f"{counters[model_name]}."

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

        "Reviewer1_Interaction_Uncertainty_Results"

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
        "REVIEWER #1"
    )

    print(
        "STATISTICAL UNCERTAINTY EXPERIMENT"
    )

    print(
        "INTERACTION-ORIENTED DIFFUSION MODELS"
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
        "Confidence   : 95%"
    )

    print(
        "=" * 70
    )

  

    datasets = load_all_datasets()

    

    all_stats = {}

    seed_records = []

    

    for (

        dataset_name,
        snapshots

    ) in datasets.items():

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

       

        for k in K_VALUES:

            (

                stats_for_k,

                seed_record

            ) = run_dataset_k_experiment(

                dataset_name,

                snapshots,

                k

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
        "\nThe reported SDs and 95% CIs are "
        "computed from the actual Monte Carlo "
        "realizations."
    )

    print(
        "\nTotal model evaluations:"
    )

    print(

        f"  "
        f"{len(DATASET_PATHS)} datasets × "
        f"{len(K_VALUES)} seed sizes × "
        f"{len(STOCHASTIC_MODELS)} models × "
        f"{N_RUNS} MC runs"

    )

    print(

        f"  = "
        f"{len(DATASET_PATHS) * len(K_VALUES) * len(STOCHASTIC_MODELS) * N_RUNS:,} "
        f"model realizations"

    )



if __name__ == "__main__":

    mp.freeze_support()

    main()
