import datetime
from pathlib import Path

import gymnasium as gym
import wandb
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecEnv,
)
from wandb.integration.sb3 import WandbCallback

import cluster_alns.rl.environments
from cluster_alns.runners.rl.config import RLConfig

MODEL_MAPPING = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


class Trainer:

    envs: VecEnv
    model: BaseAlgorithm
    callback: WandbCallback
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space

    def __init__(self, config_path: Path):
        self.config = RLConfig.from_yaml(config_path)
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.env_name = "-".join(self.config.environment.gym_id.split("-")[:-1])
        self.exp_name = f"{self.env_name}_{time_now}"
        self.wandb_run = wandb.init(
            project="DRL-ALNS",
            name=f"{self.env_name}_{time_now}",
            config=self.config.to_dict(),
            sync_tensorboard=True,
        )
        self.__env_setup()
        self.__setup_model()

    def __env_setup(self):
        self.envs = make_vec_env(
            env_id=self.config.environment.gym_id,
            n_envs=self.config.environment.n_workers,
            monitor_dir="logs/",
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn"},
        )

    def __setup_model(self):
        model_class = MODEL_MAPPING[self.config.model.model_name]
        self.model = model_class(
            policy=self.config.model.policy,
            env=self.envs,
            tensorboard_log=f"logs/{self.exp_name}",
            **self.config.models_hp.get(self.config.model.model_name, {}),
        )
        self.callback = WandbCallback(
            model_save_path=f"trained-models/{self.exp_name}",
            model_save_freq=1000,
            log="all",
        )

    def train(self):
        self.model.learn(
            total_timesteps=self.config.model.n_steps,
            reset_num_timesteps=True,
            callback=self.callback,
        )
