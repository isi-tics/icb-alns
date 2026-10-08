"""
Bayesian Optimisation tuning for ICB-ALNS using SMAC3.
100% faithful to the ALNS-BO methodology from Reijnen et al. (2024):

  - 25 independent SMAC runs, each one a separate process with the FULL
    walltime (12h for 20/50 customers, 24h for 100), see --run-id
  - 25 dedicated tuning instances (fixed seeds, shared by all runs)
  - 100 ALNS iterations per evaluation, for every instance size
  - SA: LINEAR cooling to end_temperature ≈ 0 (1e-6)
  - w4 fixed at 0
  - t_start ∈ [0.01, 5]
  - 25 tuning instances avaliadas EM PARALELO (N_WORKERS processos)
  - Pool criado UMA vez (evita overhead de spawn no Windows a cada target())
  - Usa custom_alns (com neighbor graph atualizado a cada iteração),
    idêntico ao runner de produção

Usage:
    # all 25 runs from one node, K at a time (each run uses --workers processes)
    python tune_icb_alns.py --customers 20 --all --parallel 5 --workers 8
    # or one run per process, e.g. a SLURM array (paper_reproduction/ICB_tune.sh)
    python tune_icb_alns.py --customers 20 --run-id 0 --workers 8
    # afterwards, pick the best configuration over the runs
    python tune_icb_alns.py --customers 20 --aggregate

Output: tuning_bo/icb_alns_best_config_<customers>.json
"""

import argparse
import collections
import json
import os
import random
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.random as rnd
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac import Scenario
from smac.facade import HyperparameterOptimizationFacade
from tqdm import tqdm

from generator.op.instances import InstanceGenerator

# IMPORTANT: use custom ALNS (updates neighbor graph every iteration),
# identical to the production runner. Do NOT use vanilla `from alns import ALNS`.
from cluster_alns.custom_alns import ALNS

from cluster_alns.tsp.operators.ai4.destroy import (
    cluster_representative_removal_op,
    neighbor_graph_removal,
    random_removal,
    relatedness_removal,
)
from cluster_alns.tsp.operators.ai4.repair import (
    cluster_priority_repair_op,
    random_best_distance_repair,
    random_best_prize_repair,
    random_best_ratio_repair,
)
from cluster_alns.tsp.problem.ai4_state import AI4TSPState
from cluster_alns.tsp.problem.initial_solution import ai4_initial_solution
from cluster_alns.runners.alns.tsp.ai4_runner import CLUSTER_COUPLING
from cluster_alns.tsp.utils import find_optimal_k_elbow_op

# ── Constants (directly from paper) ───────────────────────────────────────────
ITERATIONS_MAP  = {20: 100,   50: 100,   100: 100}
WALLTIME_MAP    = {20: 43200, 50: 43200, 100: 86400}  # 12h / 12h / 24h, per run
N_TUNING        = 25
N_SMAC_RUNS     = 25
TUNING_GEN_SEED = 999_999

# ── Config space (Table 2, Reijnen et al. 2024) ────────────────────────────────
CS = ConfigurationSpace(seed=42)
CS.add_hyperparameters([
    UniformFloatHyperparameter("w1",      0.0,  50.0, default_value=5.0),
    UniformFloatHyperparameter("w2",      0.0,  50.0, default_value=3.0),
    UniformFloatHyperparameter("w3",      0.0,  50.0, default_value=1.0),
    UniformFloatHyperparameter("decay",   0.5,  1.0,  default_value=0.8),
    UniformFloatHyperparameter("dod",     0.1,  1.0,  default_value=0.3),
    UniformFloatHyperparameter("t_start", 0.01, 5.0,  default_value=1.0),
])


# ── Tuning instance generation ─────────────────────────────────────────────────

def generate_tuning_instances(customers: int):
    xs, dists = [], []
    for i in range(N_TUNING):
        # InstanceGenerator reseeds numpy/random itself, so the seed must go through it
        gen = InstanceGenerator(n_instances=1, n_nodes=customers, seed=TUNING_GEN_SEED + i)
        inst_optw, adj = gen.generate_instance_files(save=False)
        x = inst_optw[["XCOORD", "YCOORD", "TW_LOW", "TW_HIGH", "PRIZE", "MAXTIME"]].to_numpy(dtype=float)
        # same column layout as ai4.pkl: node id first
        xs.append(np.column_stack([np.arange(1, customers + 1), x]))
        dists.append(adj.to_numpy(dtype=float))
    return xs, dists


# ── Single evaluation ──────────────────────────────────────────────────────────

def _eval_one(args):
    warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    X, dist_matrix, cfg, iterations, seed, inst_idx = args

    rs    = rnd.RandomState(seed)
    nodes = list(range(1, len(X) + 1))
    state = AI4TSPState(nodes, [], X, dist_matrix, seed, find_optimal_k_elbow_op(X, rs))

    # custom_alns: atualiza o neighbor graph a cada iteração (igual ao runner)
    alns = ALNS(rs, problem_type="ai4tsp")
    for op in (random_removal, relatedness_removal,
               neighbor_graph_removal, cluster_representative_removal_op):
        alns.add_destroy_operator(op)
    for op in (random_best_distance_repair, random_best_prize_repair,
               random_best_ratio_repair, cluster_priority_repair_op):
        alns.add_repair_operator(op)

    current = random_best_prize_repair(ai4_initial_solution(state, 1), rs)

    t_start = cfg["t_start"]

    # FIX: step correto para linear cooling de t_start até ~0 em (iterations-1) passos.
    # step=iterations-1 estava ERRADO (tornava temperatura negativa na 1ª iteração).
    accept = SimulatedAnnealing(
        start_temperature=t_start,
        end_temperature=1e-6,
        step=(t_start - 1e-6) / (iterations - 1),
        method="linear",
    )

    select = RouletteWheel(
        scores=[cfg["w1"], cfg["w2"], cfg["w3"], 0.0],
        decay=cfg["decay"],
        num_destroy=4,
        num_repair=4,
        op_coupling=CLUSTER_COUPLING,
    )

    result = alns.iterate(
        current, select, accept, MaxIterations(iterations),
        degree_of_destruction=cfg["dod"],
        use_pca=True,
    )

    # O custom_alns já atualiza o graph internamente a cada iteração.
    # NÃO é necessário atualizar manualmente aqui.
    obj = result.best_state.objective()
    return obj, f"  inst {inst_idx+1:02d}/25  best={obj:.4f}"


# ── Target function factory ────────────────────────────────────────────────────

def make_target(tuning_x, tuning_dist, iterations, executor):
    """
    executor é um ProcessPoolExecutor persistente criado uma vez em tune().
    Evita overhead de spawn no Windows a cada chamada de target().
    """
    eval_counter = [0]

    def target(config, seed: int = 0) -> float:
        eval_counter[0] += 1
        cfg  = dict(config)
        args = [
            (tuning_x[i], tuning_dist[i], cfg, iterations, seed + i, i)
            for i in range(N_TUNING)
        ]
        futures = [executor.submit(_eval_one, a) for a in args]
        results, log_lines = [], []
        for f in futures:
            obj, log = f.result()
            results.append(obj)
            log_lines.append(log)

        mean_obj = float(np.mean(results))
        tqdm.write(f"\n[eval #{eval_counter[0]}]  mean={mean_obj:.4f}")
        for line in log_lines:
            tqdm.write(line)
        return mean_obj

    return target


# ── Main ───────────────────────────────────────────────────────────────────────

def tune(customers, run_id, output_dir="tuning_bo", n_workers=4, walltime=None):
    os.makedirs(output_dir, exist_ok=True)
    iterations = ITERATIONS_MAP[customers]
    walltime   = walltime or WALLTIME_MAP[customers]

    print(f"\nICB-ALNS BO | customers={customers} | run {run_id} | {n_workers} workers | {walltime/3600:.0f}h")
    tuning_x, tuning_dist = generate_tuning_instances(customers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        target_fn = make_target(tuning_x, tuning_dist, iterations, executor)
        scenario = Scenario(
            configspace=CS,
            deterministic=False,
            walltime_limit=walltime,
            n_trials=1_000_000,  # walltime is the only budget
            seed=run_id,
            output_directory=Path(output_dir) / f"smac_{customers}_{run_id}",
        )
        # small initial design so the surrogate model is used within the walltime
        initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=12)
        smac = HyperparameterOptimizationFacade(scenario, target_fn, initial_design=initial_design)
        incumbent = smac.optimize()
        print(f"\n>>> run {run_id} incumbent_cost={smac.runhistory.get_cost(incumbent):.4f}")


def tune_all(customers, parallel, output_dir, n_workers, walltime):
    """All independent runs, `parallel` at a time, each one in its own process."""
    def launch(run_id):
        cmd = [sys.executable, __file__, "--customers", str(customers), "--run-id", str(run_id),
               "--output_dir", output_dir, "--workers", str(n_workers)]
        if walltime:
            cmd += ["--walltime", str(walltime)]
        with open(Path(output_dir) / f"run_{customers}_{run_id}.log", "w") as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode

    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        codes = list(pool.map(launch, range(N_SMAC_RUNS)))
    print(f"exit codes: {codes}")
    aggregate(customers, output_dir)


def aggregate(customers, output_dir="tuning_bo"):
    """Best configuration over the independent runs, read from the runhistories."""
    best = None
    for path in sorted(Path(output_dir).glob(f"smac_{customers}_*/*/*/runhistory.json")):
        rh = json.loads(path.read_text())
        costs = collections.defaultdict(list)
        for trial in rh["data"]:
            costs[trial["config_id"]].append(trial["cost"])
        # only configurations the intensifier evaluated as often as the most evaluated one
        n_max = max(map(len, costs.values()))
        cost, config_id = min((np.mean(c), i) for i, c in costs.items() if len(c) == n_max)
        if best is None or cost < best["best_cost"]:
            best = {"best_cost": float(cost), "run": path.parts[-4], "evaluations": n_max,
                    **rh["configs"][str(config_id)]}
    if best is None:
        sys.exit(f"No runhistory found in {output_dir} for {customers} customers")
    result = {"customers": customers, "iterations": ITERATIONS_MAP[customers],
              "tuning_gen_seed": TUNING_GEN_SEED, "w4": 0.0, **best}
    out = Path(output_dir) / f"icb_alns_best_config_{customers}.json"
    out.write_text(json.dumps(result, indent=4))
    print(f"Saved -> {out}\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers",  type=int, choices=[20, 50, 100], required=True)
    parser.add_argument("--run-id",     type=int, default=0, help="Independent SMAC run (0..24), also its seed")
    parser.add_argument("--all",        action="store_true", help="Launch all 25 runs from this process")
    parser.add_argument("--parallel",   type=int, default=1, help="With --all: runs executed at the same time")
    parser.add_argument("--aggregate",  action="store_true", help="Pick the best configuration over finished runs")
    parser.add_argument("--output_dir", default="tuning_bo")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Workers paralelos para as 25 instâncias de tuning")
    parser.add_argument("--walltime",   type=float, default=None,
                        help="Seconds per run; only for smoke tests (default: 12h/12h/24h)")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.customers, args.output_dir)
    elif args.all:
        tune_all(args.customers, args.parallel, args.output_dir, args.workers, args.walltime)
    else:
        tune(args.customers, args.run_id, args.output_dir, args.workers, args.walltime)
