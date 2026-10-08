"""
Final evaluation: every model on every instance size with fixed seeds.

Run after the tuning (tune_icb_alns.py --aggregate), from this folder:
    python run_seeds.py --prepare                               # copy the tuned parameters into the configs (once)
    python run_seeds.py                                         # everything, one node
    python run_seeds.py --sizes 100 --models pca --seeds 1 2 3  # a slice, e.g. one cluster job
    python run_seeds.py --summary                               # only rebuild the tables

DR-ALNS uses the trained policies in models/dr_alns_<size>.zip.
Output in results_seeds/: one AI4-<model>-<size>-<seed>_results.csv per run,
summary_seeds.csv (mean and std over seeds) and paired_seeds.csv (paired tests per instance).
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

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
        new = {**params, **{key: tuned[key] for key in TUNED_KEYS}}
        if new != params:  # parallel jobs must not rewrite a config another job is reading
            config(model, size).write_text(json.dumps(new, indent=4) + "\n")


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
    """Mean prize per run and seed, and paired comparisons per instance (prize averaged over seeds)."""
    runs = []
    for path in out.glob("AI4-*_results.csv"):
        _, model, size, seed = path.stem.removesuffix("_results").split("-")
        prize = -pd.read_csv(path, sep=";")["best_objective"]
        runs.append(pd.DataFrame({"size": int(size), "model": model, "seed": int(seed),
                                  "instance": range(len(prize)), "prize": prize}))
    runs = pd.concat(runs)
    per_seed = runs.groupby(["size", "model", "seed"])["prize"].mean()
    table = per_seed.groupby(["size", "model"]).agg(["mean", "std", "count"])
    table.to_csv(out / "summary_seeds.csv")

    paired = []
    for size, group in runs.groupby("size"):
        wide = group.groupby(["instance", "model"])["prize"].mean().unstack().dropna()
        for a, b in combinations(wide.columns, 2):
            diff = wide[a] - wide[b]
            half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
            paired.append({
                "size": size, "a": a, "b": b, "instances": len(diff),
                "mean_diff": diff.mean(), "ci_low": diff.mean() - half, "ci_high": diff.mean() + half,
                "pct": 100 * diff.mean() / wide[b].mean(), "median_diff": diff.median(),
                "wins_a": int((diff > 0.01).sum()), "wins_b": int((diff < -0.01).sum()),
                "wilcoxon_p": wilcoxon(wide[a], wide[b]).pvalue if diff.abs().sum() else 1.0,
            })
    pd.DataFrame(paired).to_csv(out / "paired_seeds.csv", index=False)
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
    parser.add_argument("--prepare", action="store_true", help="Only copy the tuned parameters")
    parser.add_argument("--summary", action="store_true", help="Skip the runs, only summarise")
    args = parser.parse_args()

    for size in [] if args.summary else args.sizes:
        if args.prepare or {"kmeans", "pca"} & set(args.models):
            apply_tuned(size, args.tuned_dir)
        for seed in [] if args.prepare else args.seeds:
            for model in args.models:
                if model == "rl":
                    run_rl(size, seed, args.out, args.instances)
                else:
                    run_alns(model, size, seed, args.data, args.out, args.instances)
    if not args.prepare:
        print(summary(args.out))
