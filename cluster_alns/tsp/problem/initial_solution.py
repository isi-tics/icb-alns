import random
from copy import deepcopy

from cluster_alns.utils import NeighborGraph


# --- init solution ---
def compute_initial_solution(current):
    # uses Nearest Neighbor heuristic
    unvisited = [i for i in range(0, current.nb_customers)]
    current_node = random.choice(unvisited)
    solution = [current_node]
    unvisited.remove(current_node)

    while unvisited:
        next_node = min(
            unvisited, key=lambda x: current.dist_matrix_data[current_node][x]
        )
        unvisited.remove(next_node)
        solution.append(next_node)
        current_node = next_node

    # Add the depot as the last node in the solution
    current.routes = solution
    current.graph = NeighborGraph(current.nb_customers)

    return current


def ai4_initial_solution(state, init_node):
    state = deepcopy(state)
    state.routes = [init_node, init_node]

    return state
