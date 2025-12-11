import numpy as np
import numpy.random as rnd
import pandas as pd
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from scipy.spatial import distance_matrix

from cluster_alns import autofit_weights
from cluster_alns.runners.alns.runner import ALNSRunner
from cluster_alns.tsp.operators.destroy import (
    neighbor_graph_removal,
    random_removal,
    relatedness_removal,
)
from cluster_alns.tsp.operators.repair import (
    multi_processing_regret_insertion,
    regret_insertion,
)
from cluster_alns.tsp.problem.initial_solution import compute_initial_solution
from cluster_alns.tsp.problem.state import TSPState
from cluster_alns.utils import readJSONFile


class TSPRunner(ALNSRunner):

    instances_customers_x: np.ndarray
    instances_customers_y: np.ndarray
    distance_matrix: np.ndarray

    def _set_parameters(self) -> None:
        self.parameters = readJSONFile(self.path_parameters)
        self.path_instance = self.path_instance / self.parameters["instance_file"]
        self.seed = self.parameters["rseed"]
        self.iterations = self.parameters["iterations"]
        self.instances_size = self.parameters["instance_nr"]
        self.use_cluster = self.parameters.get("use_cluster", True)
        use_pca = self.parameters.get("use_pca", True)
        mode = "original"
        if self.use_cluster:
            mode = "PCA" if use_pca else "KMeans"
        self.exp_name = (
            f"TSP-{mode}-{self.iterations}-{self.seed}"
        )
        self.random_state = rnd.RandomState(self.seed)

    def _set_instances(self):
        data = np.array(pd.read_pickle(self.path_instance))
        self.instances_customers_x = data[: self.instances_size, :, 0]
        self.instances_customers_y = data[: self.instances_size, :, 1]
        distance_matrixes = []
        for i in range(self.instances_size):
            customers_x = self.instances_customers_x[i].reshape(-1, 1)
            customers_y = self.instances_customers_y[i].reshape(-1, 1)
            dst_mtx = distance_matrix(customers_x, customers_y)
            distance_matrixes.append(dst_mtx)
        self.distance_matrix = np.array(distance_matrixes)

    def _add_destroy_operators(self):
        self.alns.add_destroy_operator(random_removal)
        self.alns.add_destroy_operator(relatedness_removal)
        self.alns.add_destroy_operator(neighbor_graph_removal)

    def _add_repair_operators(self):
        if self.instances_customers_x.shape[1] <= 100:
            self.alns.add_repair_operator(regret_insertion)
        else:
            self.alns.add_repair_operator(multi_processing_regret_insertion)

    def _setup(self, ith_instance: int):
        dst_mtx = self.distance_matrix[ith_instance]
        n_customers = self.instances_customers_x.shape[1]
        state = TSPState([], n_customers, dst_mtx, self.seed)
        init_solution = compute_initial_solution(state)
        weigts = [
            self.parameters["w1"],
            self.parameters["w2"],
            self.parameters["w3"],
            0,
        ]
        select = RouletteWheel(
            weigts,
            decay=self.parameters["decay"],
            num_destroy=3,
            num_repair=1,
        )
        accept = autofit_weights.autofit(
            SimulatedAnnealing,
            init_obj=init_solution.objective(),
            worse=0.05,
            accept_prob=0.5,
            num_iters=self.parameters["iterations"],
        )
        stop = MaxIterations(self.parameters["iterations"])
        nr_nodes_to_remove = None
        if self.parameters["degree_of_destruction"] is not None:
            nr_nodes_to_remove = round(
                self.parameters["degree_of_destruction"] * n_customers
            )
        return init_solution, select, accept, stop, nr_nodes_to_remove
