# An Intelligent Clustering-Based ALNS: A Strong Baseline for Neural Combinatorial Optimization

Adaptive Large Neighborhood Search (ALNS) remains a dominant metaheuristic for vehicle routing, yet the design of its destroy and repair operators relies heavily on manual engineering. Although Neural Combinatorial Optimization with Deep Reinforcement Learning (DRL) offers automation, it introduces severe computational overhead. We challenge this paradigm by proposing novel ALNS operators based on Unsupervised Learning. Our approach employs dynamic K-Means clustering applied to both raw feature spaces and latent PCA projections to identify spatial groups and guide solution reconstruction. Evaluating on the Orienteering Problem (OPSWTW) and the Capacitated Vehicle Routing Problem (CVRP), we demonstrate that a classical adaptive ALNS equipped with these operators significantly outperforms a state-of-the-art DRL baseline. We report substantial gains on large-scale instances with competitive execution times, suggesting that employing smarter and more robust heuristic operators at the execution level impacts performance more significantly than the sophistication of DRL-based control.

## Usage

To use DR-ALNS for solving COPs, follow these steps:

1. Install [uv](https://docs.astral.sh/uv)!
2. Install dependencies: `uv sync`
3. Unzip the data file: `DR-ALNS/cluster_alns/rl/environments/data.7z`
4. Copy the data files to the paper_reproduction directories:
   1. `DR-ALNS/paper_reproduction/tsp/data`
   2. `DR-ALNS/paper_reproduction/cvrp/data`
5. Configure experiment configuration in the config file (e.g., in 'DR-ALNS/paper_reproduction/tsp/configs/tsp_rl.yml')
6. Go to the example directory of the problem you want to solve (e.g., 'DR-ALNS/paper_reproduction/tsp/')
7. Run DR-ALNS algorithm: `uv run python rl_tsp.py`

## ICB-ALNS on the OPSWTW: tuning and final evaluation

Everything below runs from the repository root unless stated otherwise.

### Setup (once)

```bash
uv sync --group tune
```

Unzip `cluster_alns/rl/environments/data.7z` and place `ai4.pkl` in both
`cluster_alns/rl/environments/data/` and `paper_reproduction/tsp/data/`.

Quick check that SMAC works on the machine (20 minutes):

```bash
uv run python tune_icb_alns.py --customers 20 --run-id 0 --workers 8 --walltime 1200 --output_dir tuning_smoke
```

### Step 1: tune ICB-ALNS (same protocol as ALNS-BO in Reijnen et al., 2024)

25 independent SMAC3 runs per instance size, 12 hours each (24 hours for 100 customers),
25 fixed tuning instances, 100 ALNS iterations per evaluation. Tuned parameters: `w1`, `w2`, `w3`
in [0, 50], `decay` in [0.5, 1], `dod` in [0.1, 1], `t_start` in [0.01, 5]; `w4 = 0`.

As a SLURM array, one run per task (set the partition in the script header first):

```bash
cd paper_reproduction
sbatch ICB_tune.sh 20
sbatch ICB_tune.sh 50
sbatch ICB_tune.sh 100
# to cap how many runs execute at once: sbatch --array=0-24%5 ICB_tune.sh 20
```

Or from a single node, `--parallel` runs at a time with `--workers` processes each
(the node needs `parallel x workers` cores):

```bash
uv run python tune_icb_alns.py --customers 20 --all --parallel 5 --workers 5
```

When the runs of a size are finished, pick the best configuration:

```bash
uv run python tune_icb_alns.py --customers 20 --aggregate
uv run python tune_icb_alns.py --customers 50 --aggregate
uv run python tune_icb_alns.py --customers 100 --aggregate
```

This writes `tuning_bo/icb_alns_best_config_<size>.json`.

### Step 2: final evaluation, 10 fixed seeds

Four models (ALNS-BO, ICB-ALNS without PCA, ICB-ALNS, DR-ALNS), three sizes, seeds 1 to 10,
250 test instances each. All methods get the same iteration budget (100, 100 and 200).

```bash
cd paper_reproduction/tsp
uv run python run_seeds.py --prepare     # copies the tuned parameters into the ICB configs
cd ..
sbatch ICB_seeds.sh                      # 120 tasks, one (size, model, seed) each
cd tsp
uv run python run_seeds.py --summary     # after all tasks finish
```

Without SLURM, `uv run python run_seeds.py` runs everything on one node, and
`--sizes`, `--models` and `--seeds` select a slice.

Results go to `paper_reproduction/tsp/results_seeds/`: one CSV per run,
`summary_seeds.csv` (mean and standard deviation over seeds) and `paired_seeds.csv`
(paired differences per instance with 95% confidence interval and Wilcoxon test).
Runs are reproducible: the same seed gives the same routes.
