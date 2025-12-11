import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import cluster_alns.rl.environments


def evaluate(config_path, model_path="results/model.zip", instance=1) -> pd.DataFrame:
    import time

    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO

    from cluster_alns.runners.rl.config import RLConfig

    config = RLConfig.from_yaml(config_path)
    model = PPO.load(model_path, device="cpu")
    env = gym.make(config.environment.gym_id)

    training_objectives = np.zeros(100)

    obs, _ = env.reset(options={"instance": instance})
    begin = time.time()
    for step in range(101):
        actions, _ = model.predict(obs)
        obs, _, done, truncated, info = env.step(actions)
        training_objectives[step] = info["objective"]
        if done or truncated:
            break
    exp_time = time.time() - begin
    df = pd.DataFrame(
        {
            "instance": instance,
            "exp_time": exp_time,
            "best_objective": info["objective"],
            "solution": [info["route"]],
            "training_objectives": [training_objectives.tolist()],
        }
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="DR-ALNS CVRP",
        description="Runs the DR-ALNS algorithm on CVRP instances with or without cluster operators.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The JSON filename containing the parameters for the ALNDR-ALNSS algorithm.",
        default="original_rl_1000.yml",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--instances-eval",
        type=str,
        default="0:10",
    )
    args = parser.parse_args()
    current_path = Path(__file__).parent.resolve()
    config_path = current_path / f"configs/{args.config}"
    if int(args.train):
        from cluster_alns.runners.rl.trainer import Trainer

        trainer = Trainer(config_path=config_path)
        trainer.train()
    if int(args.evaluate):
        start_instance, end_instance = map(int, args.instances_eval.split(":"))
        result_dfs = []
        for i in tqdm(range(start_instance, end_instance)):
            result_dfs.append(evaluate(config_path, "results/model.zip", i))
        # gather all results into one dataframe
        all_results_df = pd.concat(result_dfs, ignore_index=True)
        all_results_df.to_csv(
            f"rl_cvrp_result_{start_instance}_{end_instance}.csv",
            index=False,
            sep=";",
        )
