"""
Final evaluation: every model on every instance size with fixed seeds.

Run after the tuning (tune_icb_alns.py --aggregate), from this folder:
    python run_seeds.py                                         # everything
    python run_seeds.py --sizes 100 --models pca --seeds 1 2 3  # a slice, e.g. one cluster job
    python run_seeds.py --summary                               # only rebuild the summary table

DR-ALNS needs the trained policies in models/dr_alns_<size>.zip.
Output: results_seeds/AI4-<model>-<size>-<seed>_results.csv and results_seeds/summary_seeds.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from cluster_alns.runners.alns.tsp.ai4_runner import AI4TSPRunner

HERE = Path(__file__).parent.resolve()
SIZES = (20, 50, 100)
SEEDS = tuple(range(1, 11))
MODELS = ("original", "kmeans", "pca", "rl")  # ALNS-BO, ICB-ALNS (No PCA), ICB-ALNS, DR-ALNS
TUNED_KEYS = ("w1", "w2", "w3", "decay", "dod", "t_start")


def config(model, size):
    return HERE / f"configs/ALNS_ai4_{model}_{size}.json"


def apply_tuned(size, tuned_dir):
    """Copies the tuned ICB-ALNS parameters into both ICB configs."""
    tuned = json.loads((tuned_dir / f"icb_alns_best_config_{size}.json").read_text())
    for model in ("kmeans", "pca"):
        params = json.loads(config(model, size).read_text())
        params.update({key: tuned[key] for key in TUNED_KEYS})
        config(model, size).write_text(json.dumps(params, indent=4) + "\n")


def run_alns(model, size, seed, data, out, instances):
    runner = AI4TSPRunner(config(model, size), data, seed=seed)
    runner.instances_size = instances or runner.instances_size
    runner()
    runner.write_results(out)


def _rl_instance(size, seed, instance, iterations):
    sys.path.insert(0, str(HERE))
    from rl_tsp import evaluate

    return evaluate(
        HERE / f"configs/ai4_rl_{size}_original.yml",
        HERE / f"models/dr_alns_{size}.zip",
        instance,
        seed * 1000 + instance,
        iterations,
    )


def run_rl(size, seed, out, instances):
    # same iteration budget as the ALNS-based methods of that size
    iterations = json.loads(config("original", size).read_text())["iterations"]
    rows = Parallel(n_jobs=-1)(
        delayed(_rl_instance)(size, seed, i, iterations) for i in range(instances or 250)
    )
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out / f"AI4-RL-{size}-{seed}_results.csv", index=False, sep=";")


def summary(out):
    """Mean prize over the instances of each run, then mean and std over the seeds."""
    rows = []
    for path in out.glob("AI4-*_results.csv"):
        _, model, size, seed = path.stem.removesuffix("_results").split("-")
        prize = -pd.read_csv(path, sep=";")["best_objective"].mean()
        rows.append({"size": int(size), "model": model, "seed": int(seed), "prize": prize})
    table = pd.DataFrame(rows).groupby(["size", "model"])["prize"].agg(["mean", "std", "count"])
    table.to_csv(out / "summary_seeds.csv")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES, choices=SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=HERE / "results_seeds")
    parser.add_argument("--tuned-dir", type=Path, default=HERE.parents[1] / "tuning_bo")
    parser.add_argument("--instances", type=int, default=None, help="Only for smoke tests")
    parser.add_argument("--summary", action="store_true", help="Skip the runs, only summarise")
    args = parser.parse_args()

    for size in [] if args.summary else args.sizes:
        if {"kmeans", "pca"} & set(args.models):
            apply_tuned(size, args.tuned_dir)
        for seed in args.seeds:
            for model in args.models:
                if model == "rl":
                    run_rl(size, seed, args.out, args.instances)
                else:
                    run_alns(model, size, seed, args.data, args.out, args.instances)
    print(summary(args.out))
