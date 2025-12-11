import copy
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import numpy.random as rnd
import pandas as pd
from alns import ALNS
from scipy.spatial import distance_matrix

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
from cluster_alns.cvrp.utils import (
    compute_distance_depots,
    find_optimal_k_elbow,
    update_neighbor_graph,
)

N_INSTANCES = 300


class CVRPEnv(gym.Env):
    def __init__(
        self,
        iterations: int,
        instance_file: str,
        use_cluster: bool = False,
        use_pca: bool = False,
        render_mode=None,
    ):
        # Parameters
        self.rnd_state = rnd.RandomState()

        # Simulated annealing acceptance criteria
        self.max_temperature = 5
        self.temperature = 5

        # LOAD INSTANCE
        current_path = Path(__file__).parent.resolve()
        self.path_instance = current_path / instance_file
        self._set_instances()

        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        self.improvement = None
        self.cost_difference_from_best = None
        self.current_updated = None
        self.current_improved = None

        self.use_cluster = use_cluster
        self.use_pca = use_pca
        self.optimal_ks = np.full(N_INSTANCES, 5)

        # Gym-related part
        self.reward = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations)
        self.iteration = 0  # Current gen in the episode
        self.max_iterations = iterations  # max number of generations in an episode

        # Action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete(
            [
                4 if self.use_cluster else 3,
                2 if self.use_cluster else 1,
                10,
                100,
            ]
        )
        self.observation_space = gym.spaces.Box(
            shape=(8,), low=0, high=100, dtype=np.float64
        )

    def _set_instances(self):
        data = pd.read_pickle(self.path_instance)

        data = data[:N_INSTANCES]
        depots = np.array([instance[0] for instance in data])
        customers = np.array([instance[1] for instance in data])
        self.demands = np.array([instance[2] for instance in data])
        self.capacities = np.array([instance[3] for instance in data])
        self.instances_customers_x = customers[:, :, 0]
        self.instances_customers_y = customers[:, :, 1]
        distance_matrixes = []
        distance_depots = []
        for i in range(N_INSTANCES):
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

    def make_observation(self):
        """
        Return the environment's current state
        """

        is_current_best = 0
        if self.current_solution.objective() == self.best_solution.objective():
            is_current_best = 1

        state = np.array(
            [
                self.improvement,
                self.cost_difference_from_best,
                is_current_best,
                self.temperature,
                self.stagcount,
                self.iteration / self.max_iterations,
                self.current_updated,
                self.current_improved,
            ],
            dtype=np.float64,
        ).squeeze()

        return state

    def reset(self, seed=None, options=None):
        """
        The reset method: returns the current state of the environment (first state after initialization/reset)
        """

        SEED = random.randint(0, 100000000)
        random_state = rnd.RandomState(SEED)

        # randomly select problem instance
        self.instance = random.randint(1, N_INSTANCES)
        if options is not None:
            self.instance = options["instance"]
        dst_mtx = self.distance_matrix[self.instance - 1]
        truck_capacity = self.capacities[self.instance - 1]
        dist_depot = self.distance_dpts[self.instance - 1]
        demands = self.demands[self.instance - 1]

        n_customers = self.instances_customers_x.shape[1]

        if self.use_cluster and self.optimal_ks[self.instance - 1] == 5:
            self.optimal_ks[self.instance - 1] = find_optimal_k_elbow(
                self.instances_customers_x[self.instance - 1],
                self.instances_customers_y[self.instance - 1],
                demands,
                random_state,
            )
        state = CVRPState(
            initial_solution=[],
            nb_customers=n_customers,
            truck_capacity=truck_capacity,
            dist_matrix_data=dst_mtx,
            dist_depot_data=dist_depot,
            demands_data=demands,
            customers_x=self.instances_customers_x[self.instance - 1],
            customers_y=self.instances_customers_y[self.instance - 1],
            k_optimal=self.optimal_ks[self.instance - 1],
            seed=SEED,
        )

        self.initial_solution = compute_initial_solution(state)
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)

        # add operators to the dr_alns class
        self.dr_alns = ALNS(random_state)
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)
        if self.use_cluster:
            self.dr_alns.add_destroy_operator(cluster_representative_removal)

        self.dr_alns.add_repair_operator(regret_insertion)
        if self.use_cluster:
            self.dr_alns.add_repair_operator(cluster_priority_repair)

        # reset tracking values
        self.stagcount = 0
        self.current_improved = 0
        self.current_updated = 0
        self.episode += 1
        self.temperature = self.max_temperature
        self.improvement = 0
        self.cost_difference_from_best = 0

        self.iteration, self.reward = 0, 0
        self.done = False

        return self.make_observation(), {}

    def step(self, action, **kwargs):
        self.iteration += 1
        self.stagcount += 1
        self.current_updated = 0
        self.reward = 0
        self.improvement = 0
        self.cost_difference_from_best = 0
        self.current_improved = 0

        current = self.current_solution
        best = self.best_solution

        d_idx, r_idx = action[0], action[1]
        d_name, d_operator = self.dr_alns.destroy_operators[d_idx]

        factors = {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.4,
            4: 0.5,
            5: 0.6,
            6: 0.7,
            7: 0.8,
            8: 0.9,
            9: 1.0,
        }
        nr_nodes_to_remove = round(factors[action[2]] * current.nb_customers)

        self.temperature = (1 / (action[3] + 1)) * self.max_temperature

        if nr_nodes_to_remove == current.nb_customers:
            nr_nodes_to_remove -= 1

        destroyed = d_operator(
            current,
            self.rnd_state,
            nr_nodes_to_remove,
            use_pca=self.use_pca,
        )

        r_name, r_operator = self.dr_alns.repair_operators[r_idx]
        candidate = r_operator(
            destroyed,
            self.rnd_state,
            use_pca=self.use_pca,
        )

        new_best, new_current = self.consider_candidate(best, current, candidate)

        if new_best != best and new_best is not None:
            # found new best solution
            self.best_solution = new_best
            self.current_solution = new_best
            self.current_updated = 1
            self.reward += 5
            self.stagcount = 0
            self.current_improved = 1

        elif new_current != current and new_current.objective() > current.objective():
            # solution accepted, because better than current, but not better than best
            self.current_solution = new_current
            self.current_updated = 1
            self.current_improved = 1
            # self.reward += 3

        elif new_current != current and new_current.objective() <= current.objective():
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1
            # self.reward += 1

        if new_current.objective() > current.objective():
            self.improvement = 1

        self.cost_difference_from_best = (
            self.current_solution.objective() / self.best_solution.objective()
        ) * 100

        # update graph of current and best solutions
        self.current_solution.graph = self.best_solution.graph = update_neighbor_graph(
            candidate, candidate.routes, candidate.objective()
        )

        state = self.make_observation()

        if self.iteration == self.max_iterations:
            self.done = True
        info = {
            "instance": self.instance,
            "objective": self.best_solution.objective(),
            "route": self.best_solution.routes,
        }
        return state, self.reward, self.done, False, info

    # --------------------------------------------------------------------------------------------------------------------

    def consider_candidate(self, best, curr, cand):
        # Simulated Annealing
        probability = np.exp((curr.objective() - cand.objective()) / self.temperature)

        # best:
        if cand.objective() < best.objective():
            return cand, cand

        # accepted:
        elif probability >= rnd.random():
            return None, cand

        else:
            return None, curr
