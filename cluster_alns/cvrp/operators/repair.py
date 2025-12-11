import copy
import random

from cluster_alns.cvrp.utils import compute_route_load


def get_regret_single_insertion(
    routes,
    customer,
    truck_capacity,
    distance_matrix_data,
    distance_depot_data,
    demands_data,
):
    # print('python repair')
    insertions = {}
    for route_idx in range(len(routes)):
        if (
            compute_route_load(routes[route_idx], demands_data)
            + demands_data[customer - 1]
            <= truck_capacity
        ):
            for i in range(len(routes[route_idx]) + 1):
                updated_route = (
                    routes[route_idx][:i] + [customer] + routes[route_idx][i:]
                )
                updated_routes = (
                    routes[:route_idx] + [updated_route] + routes[route_idx + 1 :]
                )
                if i == 0:
                    cost_difference = (
                        distance_depot_data[updated_route[0] - 1]
                        + distance_matrix_data[
                            updated_route[0] - 1, updated_route[1] - 1
                        ]
                        - distance_depot_data[updated_route[1] - 1]
                    )
                elif i == len(routes[route_idx]):
                    cost_difference = (
                        distance_depot_data[updated_route[-1] - 1]
                        + distance_matrix_data[
                            updated_route[i - 1] - 1, updated_route[i] - 1
                        ]
                        - distance_depot_data[updated_route[i - 1] - 1]
                    )
                else:
                    cost_difference = (
                        distance_matrix_data[
                            updated_route[i - 1] - 1, updated_route[i] - 1
                        ]
                        + distance_matrix_data[
                            updated_route[i] - 1, updated_route[i + 1] - 1
                        ]
                        - distance_matrix_data[
                            updated_route[i - 1] - 1, updated_route[i + 1] - 1
                        ]
                    )

                insertions[tuple(map(tuple, updated_routes))] = cost_difference

    if len(insertions) == 1:
        best_insertion = min(insertions, key=insertions.get)
        return best_insertion, 0

    elif len(insertions) > 1:
        best_insertion = min(insertions, key=insertions.get)

        if len(set(insertions.values())) == 1:  # when all options are of equal value:
            regret = 0
        else:
            regret = sorted(list(insertions.values()))[1] - min(insertions.values())
        return best_insertion, regret
    else:
        # no insertions possible for this customer
        return -1, -1


def regret_insertion(current, random_state, prob=1.5, **kwargs):
    visited_customers = [customer for route in current.routes for customer in route]
    all_customers = set(range(1, current.nb_customers + 1))
    unvisited_customers = all_customers - set(visited_customers)

    repaired = copy.deepcopy(current)
    while unvisited_customers:
        insertion_options = {}
        for customer in unvisited_customers:
            best_insertion, regret = get_regret_single_insertion(
                repaired.routes,
                customer,
                repaired.truck_capacity,
                repaired.dist_matrix_data,
                repaired.dist_depot_data,
                repaired.demands_data,
            )
            if best_insertion != -1:
                insertion_options[best_insertion] = regret

        if not insertion_options:
            repaired.routes.append([random.choice(list(unvisited_customers))])
        else:
            insertion_option = 0
            while (
                random.random() < 1 / prob
                and insertion_option < len(insertion_options) - 1
            ):
                insertion_option += 1
            repaired.routes = list(
                map(list, sorted(insertion_options, reverse=True)[insertion_option])
            )

        visited_customers = [
            customer for route in repaired.routes for customer in route
        ]
        unvisited_customers = all_customers - set(visited_customers)
    return repaired


def _calculate_insertion_cost(route, node, dist_matrix, depot_idx=0):
    best_pos = None
    min_increase = float("inf")

    for i in range(len(route) + 1):
        prev_node = depot_idx if i == 0 else route[i - 1] - 1
        next_node = depot_idx if i == len(route) else route[i] - 1
        node_idx = node - 1

        cost_removed = dist_matrix[prev_node][next_node]
        cost_added = dist_matrix[prev_node][node_idx] + dist_matrix[node_idx][next_node]
        increase = cost_added - cost_removed

        if increase < min_increase:
            min_increase = increase
            best_pos = i

    return best_pos, min_increase


def cluster_priority_repair(current, random_state, **kwargs):
    """
    Repairs the solution using the priority list generated by the cluster destroy operator.
    Prioritizes 'semi-central' nodes before extreme outliers.
    """
    if not hasattr(current, "priority_list"):
        return regret_insertion(current, random_state)

    repaired = copy.deepcopy(current)
    priority_list = repaired.priority_list
    delattr(repaired, "priority_list")

    nodes_in_routes = {node for route in repaired.routes for node in route}

    for node in priority_list:
        if node in nodes_in_routes:
            continue

        best_route_idx = None
        best_pos = None
        best_cost = float("inf")

        for r_idx, route in enumerate(repaired.routes):
            current_load = sum(repaired.demands[n - 1] for n in route)
            if current_load + repaired.demands[node - 1] <= repaired.truck_capacity:
                pos, increase = _calculate_insertion_cost(
                    route, node, repaired.dist_matrix
                )
                if increase < best_cost:
                    best_cost = increase
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx is not None:
            repaired.routes[best_route_idx].insert(best_pos, node)
        else:
            repaired.routes.append([node])

        nodes_in_routes.add(node)

    return repaired
