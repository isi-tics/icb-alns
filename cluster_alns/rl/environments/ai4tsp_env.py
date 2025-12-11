import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import copy
import pickle as pkl
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import numpy.random as rnd
from alns import ALNS

from cluster_alns.tsp.operators.ai4.destroy import (
    cluster_representative_removal_op,
    neighbor_graph_removal,
    random_removal,
    relatedness_removal,
)
from cluster_alns.tsp.operators.ai4.repair import (
    cluster_priority_repair_op,
    random_best_distance_repair,
    random_best_prize_repair,
    random_best_ratio_repair,
)
from cluster_alns.tsp.problem.ai4_state import AI4TSPState
from cluster_alns.tsp.problem.initial_solution import ai4_initial_solution
from cluster_alns.tsp.utils import ai4_update_neighbor_graph, find_optimal_k_elbow_op

DATA_MAP = {
    20: (0, 250),
    50: (250, 500),
    100: (500, 750),
    200: (750, 1000),
}

N_INSTANCES = 250


class AI4TSPEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        iterations: int,
        instance_file: str,
        customers: int,
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
        self._set_instances(customers)

        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        self.improvement = None
        self.cost_difference_from_best = None
        self.current_updated = None
        self.current_improved = None

        self.use_cluster = use_cluster
        self.use_pca = use_pca

        self.best_routes = []

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
                4 if self.use_cluster else 3,
                10,
                100,
            ]
        )
        self.observation_space = gym.spaces.Box(
            shape=(7,), low=0, high=100, dtype=np.float64
        )

    def _set_instances(self, customers: int):
        with open(self.path_instance, "rb") as file:
            data = pkl.load(file)
        self.instances_customers_x, self.distance_matrix = zip(*data)
        self.instances_customers_x = np.array(
            (
                self.instances_customers_x[
                    DATA_MAP[customers][0] : DATA_MAP[customers][1]
                ]
            )
        )
        self.distance_matrix = np.array(
            self.distance_matrix[DATA_MAP[customers][0] : DATA_MAP[customers][1]]
        )

    def make_observation(self):
        """
        Return the environment's current state
        """

        is_current_best = 0
        if -self.current_solution.objective() == -self.best_solution.objective():
            is_current_best = 1

        state = np.array(
            [
                self.improvement,
                self.cost_difference_from_best,
                is_current_best,
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

        # randomly select problem instance
        ith_instance = random.randint(1, N_INSTANCES)
        if options is not None:
            ith_instance = options["instance"]
        self.instance = ith_instance
        X_ = self.instances_customers_x[ith_instance - 1]
        dist_matrix_data = self.distance_matrix[ith_instance - 1]

        nodes = [(i + 1) for i in range(0, len(X_))]
        random_state = rnd.RandomState()
        # Calculate K-Optimal (Elbow) if clustering is active
        k_optimal = 5
        if self.use_cluster:
            k_optimal = find_optimal_k_elbow_op(X_, random_state)

        state = AI4TSPState(nodes, [], X_, dist_matrix_data, SEED, k_optimal)
        self.current_solution = ai4_initial_solution(state, init_node=1)
        self.initial_solution = copy.deepcopy(self.current_solution)
        self.best_solution = copy.deepcopy(self.current_solution)

        # add operators to the dr_alns class
        self.dr_alns = ALNS(random_state)
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)
        if self.use_cluster:
            self.dr_alns.add_destroy_operator(cluster_representative_removal_op)

        self.dr_alns.add_repair_operator(random_best_distance_repair)
        self.dr_alns.add_repair_operator(random_best_prize_repair)
        self.dr_alns.add_repair_operator(random_best_ratio_repair)
        if self.use_cluster:
            self.dr_alns.add_repair_operator(cluster_priority_repair_op)

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

        self.temperature = (1 / (action[3] + 1)) * self.max_temperature

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

        destroyed = d_operator(
            current,
            self.rnd_state,
            degree_of_destruction=factors[action[2]],
            use_pca=self.use_pca,
        )

        r_name, r_operator = self.dr_alns.repair_operators[r_idx]
        candidate = r_operator(destroyed, self.rnd_state, use_pca=self.use_pca)

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

        elif new_current != current and new_current.objective() <= current.objective():
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1

        if -new_current.objective() > -current.objective():
            self.improvement = 1

        a = self.current_solution.objective()
        b = self.best_solution.objective()
        if abs(a) == 0 or abs(b) == 0:
            self.cost_difference_from_best = -1
        else:
            self.cost_difference_from_best = (-a / -b) * 100

        self.current_solution.graph = self.best_solution.graph = (
            ai4_update_neighbor_graph(candidate, candidate.routes, candidate.objective())
        )
        state = self.make_observation()

        # Check if episode is finished (max ngen per episode)
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
