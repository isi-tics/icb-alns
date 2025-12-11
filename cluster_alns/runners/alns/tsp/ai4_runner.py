import pickle as pkl
import time
from typing import List, Tuple

import numpy as np
import numpy.random as rnd
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from cluster_alns import autofit_weights
from cluster_alns.runners.alns.runner import ALNSRunner
from cluster_alns.tsp.operators.ai4.destroy import (
    cluster_representative_removal_op,  # Added
    neighbor_graph_removal,
    random_removal,
    relatedness_removal,
)
from cluster_alns.tsp.operators.ai4.repair import (
    cluster_priority_repair_op,  # Added
    random_best_distance_repair,
    random_best_prize_repair,
    random_best_ratio_repair,
)
from cluster_alns.tsp.problem.ai4_state import AI4TSPState as TSPState
from cluster_alns.tsp.problem.initial_solution import ai4_initial_solution
from cluster_alns.tsp.utils import find_optimal_k_elbow_op  # Added
from cluster_alns.utils import readJSONFile

DATA_MAP = {
    20: (0, 250),
    50: (250, 500),
    100: (500, 750),
    200: (750, 1000),
}


class AI4TSPRunner(ALNSRunner):
    problem_type: str = "ai4tsp"
    instances_customers_x: np.ndarray
    instances_customers_y: np.ndarray
    distance_matrix: np.ndarray
    use_cluster: bool

    def _set_parameters(self) -> None:
        self.parameters = readJSONFile(self.path_parameters)
        self.path_instance = self.path_instance / self.parameters["instance_file"]
        self.seed = self.parameters["rseed"]
        self.iterations = self.parameters["iterations"]
        self.instances_size = self.parameters["instance_nr"]
        self.n_customers = self.parameters["customers"]
        self.use_cluster = self.parameters.get("use_cluster", True)
        use_pca = self.parameters.get("use_pca", True)
        mode = "original"
        if self.use_cluster:
            mode = "PCA" if use_pca else "KMeans"
        self.exp_name = f"AI4-{mode}-{self.n_customers}-{self.seed}"
        self.random_state = rnd.RandomState(self.seed)

    def _set_instances(self):
        with open(self.path_instance, "rb") as file:
            data = pkl.load(file)
        self.instances_customers_x, self.distance_matrix = zip(*data)
        self.instances_customers_x = np.array(
            self.instances_customers_x[
                DATA_MAP[self.n_customers][0] : DATA_MAP[self.n_customers][1]
            ]
        )

        self.distance_matrix = np.array(
            self.distance_matrix[
                DATA_MAP[self.n_customers][0] : DATA_MAP[self.n_customers][1]
            ]
        )

    def _add_destroy_operators(self):
        self.alns.add_destroy_operator(random_removal)
        self.alns.add_destroy_operator(relatedness_removal)
        self.alns.add_destroy_operator(neighbor_graph_removal)

        if self.use_cluster:
            self.alns.add_destroy_operator(cluster_representative_removal_op)

    def _add_repair_operators(self):
        self.alns.add_repair_operator(random_best_distance_repair)
        self.alns.add_repair_operator(random_best_prize_repair)
        self.alns.add_repair_operator(random_best_ratio_repair)

        if self.use_cluster:
            self.alns.add_repair_operator(cluster_priority_repair_op)

    def _setup(self, ith_instance: int):
        X = self.instances_customers_x[ith_instance - 1]
        dst_mtx = self.distance_matrix[ith_instance - 1]
        nodes = [(i + 1) for i in range(0, len(X))]

        # Calculate K-Optimal (Elbow) if clustering is active
        k_optimal = 5
        if self.use_cluster:
            k_optimal = find_optimal_k_elbow_op(X, self.random_state)

        state = TSPState(nodes, [], X, dst_mtx, self.seed, k_optimal)

        init_solution = ai4_initial_solution(state, init_node=1)

        weights = [
            self.parameters["w1"],
            self.parameters["w2"],
            self.parameters["w3"],
            0,
        ]

        select = RouletteWheel(
            weights,
            decay=self.parameters["decay"],
            num_destroy=4 if self.use_cluster else 3,
            num_repair=4 if self.use_cluster else 3,
        )

        init_solution = random_best_prize_repair(init_solution, 0)

        accept = autofit_weights.autofit(
            SimulatedAnnealing,
            init_obj=init_solution.objective(),
            worse=0.05,
            accept_prob=0.5,
            num_iters=self.parameters["iterations"],
        )
        stop = MaxIterations(self.parameters["iterations"])
        return init_solution, select, accept, stop

    def _run(self, ith_instance: int) -> Tuple[float, List[int], float, np.ndarray]:
        start_time = time.time()
        init_solution, select, accept, stop = self._setup(ith_instance)
        pool = None
        result = self.alns.iterate(
            init_solution,
            select,
            accept,
            stop,
            degree_of_destruction=self.parameters["dod"],
            pool=pool,
            use_pca=self.parameters["use_pca"],
        )
        end_time = time.time()
        instance_time = end_time - start_time
        return (
            result.best_state.objective(),
            result.best_state.routes,  # type: ignore
            instance_time,
            result.statistics.objectives.tolist(),
        )
