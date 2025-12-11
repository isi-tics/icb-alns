from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Environment:
    gym_id: str
    n_workers: int


@dataclass
class Model:
    model_name: str
    policy: str
    n_steps: int
    save_interval: int


@dataclass
class RLConfig:

    environment: Environment
    model: Model
    models_hp: Dict[str, Any]

    @classmethod
    def from_yaml(cls, file_path: Path) -> "RLConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        environment = Environment(**config_dict["environment"])
        model = Model(**config_dict["main"])
        models_hp = config_dict.get("models", {})

        return cls(environment=environment, model=model, models_hp=models_hp)

    def to_dict(self) -> Dict[str, Any]:
        config_dict = {}
        config_dict.update(vars(self.environment))
        config_dict.update(vars(self.model))
        for key, value in self.models_hp.items():
            config_dict[key] = value
        return config_dict
