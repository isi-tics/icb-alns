import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib.parallel import Parallel, delayed
from numpy.random import RandomState
from tqdm import tqdm

from cluster_alns.custom_alns import ALNS
from cluster_alns.tsp.problem.state import TSPState


class ALNSRunner(ABC):

    problem_type: str = "tsp"
    parameters: dict
    exp_name: str
    seed: int
    random_state: RandomState
    iterations: int
    initial_solution: TSPState
    instances_size: int
    result: List[Tuple[float, List[int], float]]
    exp_time: float
    alns: ALNS

    def __init__(self, path_parameters: Path, path_instance: Path) -> None:
        self.path_parameters = path_parameters.resolve()
        self.path_instance = path_instance.resolve()
        self._set_parameters()
        self._set_instances()
        self._set_alns()

    @abstractmethod
    def _set_parameters(self) -> None: ...

    @abstractmethod
    def _set_instances(self): ...

    @abstractmethod
    def _add_destroy_operators(self): ...

    @abstractmethod
    def _add_repair_operators(self): ...

    def _set_alns(self):
        self.alns = ALNS(self.random_state, self.problem_type)
        self._add_destroy_operators()
        self._add_repair_operators()

    @abstractmethod
    def _setup(self, ith_instance: int): ...

    def write_results(self, path: Path) -> None:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        objectives, routes, times, training_objectives = zip(*self.result)
        df = pd.DataFrame(
            {
                "exp_time": times,
                "best_objective": objectives,
                "solution": routes,
                "training_objectives": training_objectives,
            }
        )
        df.to_csv(path / f"{self.exp_name}_results.csv", index=False, sep=";")

    def _run(self, ith_instance: int) -> Tuple[float, List[int], float, np.ndarray]:
        start_time = time.time()
        init_solution, select, accept, stop, nr_nodes_to_remove = self._setup(
            ith_instance
        )  # type: ignore
        result = self.alns.iterate(
            init_solution,
            select,
            accept,
            stop,
            nr_nodes_to_remove=nr_nodes_to_remove,
        )
        end_time = time.time()
        instance_time = end_time - start_time
        return (
            result.best_state.objective(),
            result.best_state.routes, # type: ignore
            instance_time,
            result.statistics.objectives.tolist(),
        )

    def __call__(self) -> List[Tuple[float, List[int], float]]:
        self.result = Parallel(n_jobs=-1)(
            delayed(self._run)(i) for i in tqdm(range(self.instances_size))
        )  # type: ignore
        return self.result
