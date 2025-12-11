from gymnasium.envs.registration import register

register(
    id="TSP100-20-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 1000,
        "instance_file": "data/tsp_20_10000.pkl",
    },
)

register(
    id="TSP100-20-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 10000,
        "instance_file": "data/tsp_20_10000.pkl",
    },
)

register(
    id="TSP1000-20-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 1000,
        "instance_file": "data/tsp_20_10000.pkl",
    },
)

register(
    id="TSP1000-20-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 10000,
        "instance_file": "data/tsp_20_10000.pkl",
    },
)

register(
    id="TSP100-50-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 1000,
        "instance_file": "data/tsp_50_10000.pkl",
    },
)

register(
    id="TSP100-50-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 10000,
        "instance_file": "data/tsp_50_10000.pkl",
    },
)

register(
    id="TSP1000-50-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 1000,
        "instance_file": "data/tsp_50_10000.pkl",
    },
)

register(
    id="TSP1000-50-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 10000,
        "instance_file": "data/tsp_50_10000.pkl",
    },
)

register(
    id="TSP100-100-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 1000,
        "instance_file": "data/tsp_100_10000.pkl",
    },
)

register(
    id="TSP100-100-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 10000,
        "instance_file": "data/tsp_100_10000.pkl",
    },
)

register(
    id="TSP1000-100-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 1000,
        "instance_file": "data/tsp_100_10000.pkl",
    },
)

register(
    id="TSP1000-100-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 10000,
        "instance_file": "data/tsp_100_10000.pkl",
    },
)

register(
    id="TSP100-200-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 1000,
        "instance_file": "data/tsp_200_10000.pkl",
    },
)

register(
    id="TSP100-200-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 100,
        "n_instances": 10000,
        "instance_file": "data/tsp_200_10000.pkl",
    },
)

register(
    id="TSP1000-200-1000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 1000,
        "instance_file": "data/tsp_200_10000.pkl",
    },
)

register(
    id="TSP1000-200-10000-v0",
    entry_point="cluster_alns.rl.environments.tsp_env:TSPEnv",
    kwargs={
        "iterations": 1000,
        "n_instances": 10000,
        "instance_file": "data/tsp_200_10000.pkl",
    },
)

register(
    id="AI4-Original-20-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 20,
        "instance_file": "data/ai4.pkl",
        "use_cluster": False,
        "use_pca": False,
    },
)

register(
    id="AI4-KMeans-20-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 20,
        "instance_file": "data/ai4.pkl",
        "use_cluster": True,
        "use_pca": False,
    },
)

register(
    id="AI4-PCA-20-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 20,
        "instance_file": "data/ai4.pkl",
        "use_cluster": True,
        "use_pca": True,
    },
)

register(
    id="AI4-Original-50-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 50,
        "instance_file": "data/ai4.pkl",
        "use_cluster": False,
        "use_pca": False,
    },
)

register(
    id="AI4-KMeans-50-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 50,
        "instance_file": "data/ai4.pkl",
        "use_cluster": True,
        "use_pca": False,
    },
)

register(
    id="AI4-PCA-50-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 50,
        "instance_file": "data/ai4.pkl",
        "use_cluster": True,
        "use_pca": True,
    },
)

register(
    id="AI4-Original-100-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 100,
        "instance_file": "data/ai4.pkl",
        "use_cluster": False,
        "use_pca": False,
    },
)

register(
    id="AI4-KMeans-100-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 100,
        "instance_file": "data/ai4.pkl",
        "use_cluster": True,
        "use_pca": False,
    },
)

register(
    id="AI4-PCA-100-v0",
    entry_point="cluster_alns.rl.environments.ai4tsp_env:AI4TSPEnv",
    kwargs={
        "iterations": 100,
        "customers": 100,
        "instance_file": "data/ai4.pkl",
        "use_cluster": True,
        "use_pca": True,
    },
)

register(
    id="CVRP-Original-100-v0",
    entry_point="cluster_alns.rl.environments.cvrp_env:CVRPEnv",
    kwargs={
        "iterations": 100,
        "instance_file": "data/cvrp_100_10000.pkl",
        "use_cluster": False,
        "use_pca": False,
    },
)

register(
    id="CVRP-Original-1000-v0",
    entry_point="cluster_alns.rl.environments.cvrp_env:CVRPEnv",
    kwargs={
        "iterations": 1000,
        "instance_file": "data/cvrp_100_10000.pkl",
        "use_cluster": False,
        "use_pca": False,
    },
)

register(
    id="CVRP-Original-10000-v0",
    entry_point="cluster_alns.rl.environments.cvrp_env:CVRPEnv",
    kwargs={
        "iterations": 10000,
        "instance_file": "data/cvrp_100_10000.pkl",
        "use_cluster": False,
        "use_pca": False,
    },
)

register(
    id="CVRP-KMeans-1000-v0",
    entry_point="cluster_alns.rl.environments.cvrp_env:CVRPEnv",
    kwargs={
        "iterations": 1000,
        "instance_file": "data/cvrp_100_10000.pkl",
        "use_cluster": True,
        "use_pca": False,
    },
)

register(
    id="CVRP-KMeans-10000-v0",
    entry_point="cluster_alns.rl.environments.cvrp_env:CVRPEnv",
    kwargs={
        "iterations": 10000,
        "instance_file": "data/cvrp_100_10000.pkl",
        "use_cluster": True,
        "use_pca": False,
    },
)

register(
    id="CVRP-PCA-1000-v0",
    entry_point="cluster_alns.rl.environments.cvrp_env:CVRPEnv",
    kwargs={
        "iterations": 1000,
        "instance_file": "data/cvrp_100_10000.pkl",
        "use_cluster": True,
        "use_pca": True,
    },
)

register(
    id="CVRP-PCA-10000-v0",
    entry_point="cluster_alns.rl.environments.cvrp_env:CVRPEnv",
    kwargs={
        "iterations": 10000,
        "instance_file": "data/cvrp_100_10000.pkl",
        "use_cluster": True,
        "use_pca": True,
    },
)