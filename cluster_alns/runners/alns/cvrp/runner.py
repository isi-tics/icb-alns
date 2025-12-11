import numpy as np
import numpy.random as rnd
import pandas as pd
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from scipy.spatial import distance_matrix

from cluster_alns import autofit_weights
from cluster_alns.cvrp.operators.destroy import (
    cluster_representative_removal,
    neighbor_graph_removal,
    random_removal,
    relatedness_removal,
)
from cluster_alns.cvrp.operators.repair import (
    cluster_priority_repair,
    regret_insertion,
)
from cluster_alns.cvrp.problem.initial_solution import compute_initial_solution
from cluster_alns.cvrp.problem.state import CVRPState
from cluster_alns.cvrp.utils import compute_distance_depots, find_optimal_k_elbow
from cluster_alns.runners.alns.runner import ALNSRunner
from cluster_alns.utils import readJSONFile


class CVRPRunner(ALNSRunner):

    problem_type: str = "cvrp"
    instances_customers_x: np.ndarray
    instances_customers_y: np.ndarray
    distance_matrix: np.ndarray
    distance_dpts: np.ndarray
    demands: np.ndarray
    use_cluster: bool

    def _set_parameters(self) -> None:
        self.parameters = readJSONFile(self.path_parameters)
        self.path_instance = self.path_instance / self.parameters["instance_file"]
        self.seed = self.parameters["rseed"]
        self.iterations = self.parameters["iterations"]
        self.instances_size = self.parameters["instance_nr"]

        # Configurable flag: Default to True if not present
        self.use_cluster = self.parameters.get("use_cluster", True)
        use_pca = self.parameters.get("use_pca", True)
        mode = "original"
        if self.use_cluster:
            mode = "PCA" if use_pca else "KMeans"
        self.exp_name = (
            f"CVRP-{mode}-{self.iterations}-{self.seed}"
        )
        self.random_state = rnd.RandomState(self.seed)

    def _set_instances(self):
        data = pd.read_pickle(self.path_instance)
        data = data[: self.instances_size]
        depots = np.array([instance[0] for instance in data])
        customers = np.array([instance[1] for instance in data])
        self.demands = np.array([instance[2] for instance in data])
        self.capacities = np.array([instance[3] for instance in data])
        self.instances_customers_x = customers[:, :, 0]
        self.instances_customers_y = customers[:, :, 1]
        distance_matrixes = []
        distance_depots = []
        for i in range(self.instances_size):
            customers_x = self.instances_customers_x[i].reshape(-1, 1)
            customers_y = self.instances_customers_y[i].reshape(-1, 1)
            depot_x, depot_y = depots[i]
            dst_mtx = distance_matrix(customers_x, customers_y)
            dst_dpt = compute_distance_depots(
                depot_x,
                depot_y,
                self.instances_customers_x[i],
                self.instances_customers_y[i],
            )
            distance_matrixes.append(dst_mtx)
            distance_depots.append(dst_dpt)
        self.distance_matrix = np.array(distance_matrixes)
        self.distance_dpts = np.array(distance_depots)

    def _add_destroy_operators(self):
        self.alns.add_destroy_operator(random_removal)
        self.alns.add_destroy_operator(relatedness_removal)
        self.alns.add_destroy_operator(neighbor_graph_removal)

        if self.use_cluster:
            self.alns.add_destroy_operator(cluster_representative_removal)

    def _add_repair_operators(self):
        self.alns.add_repair_operator(regret_insertion)

        if self.use_cluster:
            self.alns.add_repair_operator(cluster_priority_repair)

    def _setup(self, ith_instance: int):
        dst_mtx = self.distance_matrix[ith_instance]
        truck_capacity = self.capacities[ith_instance]
        dist_depot = self.distance_dpts[ith_instance]
        demands = self.demands[ith_instance]

        n_customers = self.instances_customers_x.shape[1]

        k_optimal = 5
        if self.use_cluster:
            k_optimal = find_optimal_k_elbow(
                self.instances_customers_x[ith_instance],
                self.instances_customers_y[ith_instance],
                demands,
                self.random_state,
            )

        state = CVRPState(
            initial_solution=[],
            nb_customers=n_customers,
            truck_capacity=truck_capacity,
            dist_matrix_data=dst_mtx,
            dist_depot_data=dist_depot,
            demands_data=demands,
            customers_x=self.instances_customers_x[ith_instance],
            customers_y=self.instances_customers_y[ith_instance],
            k_optimal=k_optimal,
            seed=self.seed,
        )

        init_solution = compute_initial_solution(state)

        # Dynamic weights based on configuration
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
            num_repair=2 if self.use_cluster else 1,
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
