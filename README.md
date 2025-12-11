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

