import copy
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import numpy.random as rnd
import pandas as pd
from alns import ALNS
from scipy.spatial import distance_matrix

from cluster_alns.tsp.operators.destroy import (
    neighbor_graph_removal,
    random_removal,
    relatedness_removal,
)
from cluster_alns.tsp.operators.repair import regret_insertion
from cluster_alns.tsp.problem.initial_solution import compute_initial_solution
from cluster_alns.tsp.problem.state import TSPState
from cluster_alns.tsp.utils import update_neighbor_graph


class TSPEnv(gym.Env):
    
    metadata = {"render_modes": []}
    
    def __init__(self, iterations: int, instance_file: str, n_instances: int, render_mode=None):

        # Parameters
        self.rnd_state = rnd.RandomState()

        # Simulated annealing acceptance criteria
        self.max_temperature = 5
        self.temperature = 5

        # LOAD INSTANCE
        current_path = Path(__file__).parent.resolve()
        self.path_instance = current_path / instance_file
        self.instances_size = n_instances
        self._set_instances()

        self.instances = list(range(1, n_instances + 1))
        self.instance = None

        self.initial_solution = None
        self.best_solution = None
        self.current_solution = None

        self.improvement = None
        self.cost_difference_from_best = None
        self.current_updated = None
        self.current_improved = None

        # Gym-related part
        self.reward = 0  # Total episode reward
        self.done = False  # Termination
        self.episode = 0  # Episode number (one episode consists of ngen generations)
        self.iteration = 0  # Current gen in the episode
        self.max_iterations = iterations  # max number of generations in an episode

        # Action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete([3, 1, 10, 100])
        self.observation_space = gym.spaces.Box(
            shape=(8,), low=0, high=100, dtype=np.float64
        )

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

        # randomly select problem instance
        self.instance = random.choice(self.instances)
        n_customers = self.instances_customers_x.shape[1]
        dist_matrix_data = self.distance_matrix[self.instance - 1]

        random_state = rnd.default_rng(seed)
        state = TSPState([], n_customers, dist_matrix_data, seed)

        self.initial_solution = compute_initial_solution(state)
        self.current_solution = copy.deepcopy(self.initial_solution)
        self.best_solution = copy.deepcopy(self.initial_solution)

        # add operators to the dr_alns class
        self.dr_alns = ALNS(random_state)
        self.dr_alns.add_destroy_operator(random_removal)
        self.dr_alns.add_destroy_operator(relatedness_removal)
        self.dr_alns.add_destroy_operator(neighbor_graph_removal)

        self.dr_alns.add_repair_operator(regret_insertion)

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

        destroyed = d_operator(current, self.rnd_state, nr_nodes_to_remove)

        r_name, r_operator = self.dr_alns.repair_operators[r_idx]
        candidate = r_operator(destroyed, self.rnd_state)

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
            # solution accepted
            self.current_solution = new_current
            self.current_updated = 1
            self.current_improved = 1

        elif new_current != current and new_current.objective() <= current.objective():
            self.current_solution = new_current
            self.current_updated = 1

        if new_current.objective() > current.objective():
            self.improvement = 1

        self.cost_difference_from_best = (
            self.current_solution.objective() / self.best_solution.objective()
        ) * 100

        # update graph of current and best solutions
        self.current_solution.graph = self.best_solution.graph = update_neighbor_graph(
            candidate, candidate.route, candidate.objective()
        )

        state = self.make_observation()

        # Check if episode is finished (max ngen per episode)
        if self.iteration == self.max_iterations:
            self.done = True

        return state, self.reward, self.done, False, {}

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
