import copy
import random

import numpy as np

from cluster_alns.tsp.utils import tour_check

NR_INTERMEDIATE_SOLUTION_EVALUATIONS = 2


def get_best_distance_insertion_for_node(node, tour, adj):
    tour = np.array(tour)
    adj = np.array(adj)

    predecessor_nodes = tour[:-1]
    successor_nodes = tour[1:]
    distances = (
        adj[node - 1, predecessor_nodes - 1]
        + adj[node - 1, successor_nodes - 1]
        - adj[predecessor_nodes - 1, successor_nodes - 1]
    )

    min_index = np.argmin(distances)
    return min_index + 1


def random_best_distance_repair(current, random_state, **kwargs):
    curr_obj = current.objective()
    visited = current.routes[:-1]
    not_visited = [x for x in current.nodes if x not in visited]

    nodes_to_include = random.sample(not_visited, random.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())

    for node in nodes_to_include:
        index = get_best_distance_insertion_for_node(node, current.routes, current.adj)
        candidate = copy.deepcopy(current)
        candidate.routes.insert(index, node)
        cand_obj = candidate.objective()
        if cand_obj < curr_obj:
            curr_obj = cand_obj
            current = copy.deepcopy(candidate)

    return current


# -------- price repair ----------------------
def multiprocess_best_prize_insertions_for_node(args):
    nodes, tour, inx, node, x, adj = args
    new_tour = tour[:inx] + [node] + tour[inx:]
    total_reward, total_pen = 0, 0
    for i in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
        tour_time, rewards, pen, feas = tour_check(
            new_tour, x, adj, -1.0, -1.0, len(nodes)
        )
        total_reward += rewards
        total_pen += pen
    score = -(total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    return {tuple(new_tour): score}


def get_best_prize_insertion_for_node(
    node, nodes, tour, input_score, adj, x, pool=None
):
    best_new_tour, best_score = None, input_score
    args_list = [(nodes, tour, inx, node, x, adj) for inx in range(1, len(tour))]
    for arg in args_list:
        item = multiprocess_best_prize_insertions_for_node(arg)
        for new_tour, score in item.items():
            if best_score > score:
                best_new_tour = list(new_tour)
                best_score = score

    if best_new_tour is None:
        return tour
    else:
        return best_new_tour


def random_best_prize_repair(current, random_state, **kwargs):
    current = copy.copy(current)
    curr_obj = current.objective()

    visited = current.routes[:-1]
    not_visited = [x for x in current.nodes if x not in visited]

    nodes_to_include = random.sample(not_visited, random.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())

    pool = kwargs.get("pool", None)
    for node in nodes_to_include:
        candidate = copy.copy(current)
        cand_obj = curr_obj
        candidate.routes = get_best_prize_insertion_for_node(
            node,
            candidate.nodes,
            candidate.routes,
            cand_obj,
            candidate.adj,
            candidate.x,
            pool,
        )
        cand_obj = candidate.objective()
        if cand_obj < curr_obj:
            curr_obj = cand_obj
            current = copy.copy(candidate)

    return current


# -------- ratio repair ----------------------
def multiprocess_best_ratio_insertions_for_node(args):
    nodes, tour, current_score, current_tour_time, inx, node, x, adj = args
    new_tour = tour[:inx] + [node] + tour[inx:]

    total_tour_time, total_reward, total_pen = 0, 0, 0
    for i in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
        tour_time, rewards, pen, feas = tour_check(
            new_tour, x, adj, -1.0, -1.0, len(nodes)
        )
        total_tour_time += tour_time
        total_reward += rewards
        total_pen += pen
    tour_time = total_tour_time / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    score = -(total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    if tour_time - current_tour_time == 0:
        ratio = 0
    elif score < 0 and current_score < 0.00000001 and score < current_score:
        if tour_time < current_tour_time:
            ratio = abs(score - current_score)
        else:
            ratio = abs(score - current_score) / (tour_time - current_tour_time)
    else:
        ratio = 0

    return {tuple(new_tour): {"score": score, "ratio": ratio, "time": tour_time}}


def get_best_ratio_insertion_for_node(
    node, nodes, tour, input_score, input_time, adj, x, pool=None
):
    best_new_tour, best_ratio, best_score = None, 0, 0
    for inx in range(1, len(tour)):
        args = (nodes, tour, input_score, input_time, inx, node, x, adj)
        item = multiprocess_best_ratio_insertions_for_node(args)
        for new_tour, result in item.items():
            if result["ratio"] > best_ratio:
                best_new_tour = list(new_tour)
                best_ratio = result["ratio"]
                best_time = result["time"]

    if best_new_tour is None:
        return None, None
    else:
        return (
            best_new_tour,
            best_time,
        )


def random_best_ratio_repair(current, random_state, **kwargs):
    pool = kwargs.get("pool", None)
    current = copy.deepcopy(current)

    visited = current.routes[:-1]
    not_visited = [x for x in current.nodes if x not in visited]

    nodes_to_include = random.sample(not_visited, random.randint(0, len(not_visited)))
    nodes_to_include = sorted(nodes_to_include, key=lambda k: random.random())

    total_route_time, total_reward, total_pen = 0, 0, 0

    for i in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
        route_time, rewards, pen, feas = tour_check(
            current.routes, current.x, current.adj, -1.0, -1.0, len(current.nodes)
        )
        total_reward += rewards
        total_pen += pen
        total_route_time += route_time

    curr_obj = -(total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
    curr_route_time = total_route_time / NR_INTERMEDIATE_SOLUTION_EVALUATIONS

    for node in nodes_to_include:
        candidate = copy.copy(current)
        candidate.routes, cand_route_time = get_best_ratio_insertion_for_node(
            node,
            current.nodes,
            current.routes,
            curr_obj,
            curr_route_time,
            current.adj,
            current.x,
            pool,
        )
        if candidate.routes != None:
            cand_obj = candidate.objective()
            if cand_obj < curr_obj:
                curr_obj = cand_obj
                current = copy.copy(candidate)
                curr_route_time = cand_route_time

    return current


# -------- regret repair ----------------------
def get_regret_single_insertion(args):
    route, customer, nr_customers, adj, x = args
    insertions = {}
    for i in range(1, len(route)):
        updated_route = route[:i] + [customer] + route[i:]
        total_reward, total_pen = 0, 0
        for j in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
            tour_time, rewards, pen, feas = tour_check(
                updated_route, x, adj, -1.0, -1.0, nr_customers
            )
            total_reward += rewards
            total_pen += pen
        score = (total_reward + total_pen) / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
        if score > 0:
            insertions[tuple(updated_route)] = score

    if len(insertions) == 1:
        best_insertion = min(insertions, key=insertions.get)
        return best_insertion, 0

    elif len(insertions) > 1:
        best_insertion = max(insertions, key=insertions.get)
        if len(set(insertions.values())) == 1:
            regret = 0
        else:
            regret = (
                max(insertions.values()) - sorted(insertions.values(), reverse=True)[1]
            )
        return best_insertion, regret
    else:
        return None, None


def regret_insertion(current, random_state, prob=1.5, **kwargs):
    repaired_solution = copy.deepcopy(current)
    visited_customers = list(repaired_solution.routes[:-1])
    all_customers = repaired_solution.nodes
    unvisited_customers = [x for x in all_customers if x not in visited_customers]
    pool = kwargs.get("pool", None)

    while True:
        insertion_options = {}
        route = repaired_solution.routes[:]
        for customer in unvisited_customers:
            args = (
                route,
                customer,
                len(all_customers),
                repaired_solution.adj,
                repaired_solution.x,
            )
            best_insertion, regret = get_regret_single_insertion(args)
            if best_insertion is not None:
                insertion_options[best_insertion] = regret

        if len(insertion_options) > 0:
            insertion_option = 0
            while (
                random.random() < 1 / prob
                and insertion_option < len(insertion_options) - 1
            ):
                insertion_option += 1

            repaired_solution.routes = list(
                sorted(insertion_options, reverse=False)[insertion_option]
            )
            visited_customers = list(repaired_solution.routes[:-1])
            unvisited_customers = [
                x for x in all_customers if x not in visited_customers
            ]

        else:
            return repaired_solution


def beam_search(current, random_state, **kwargs):
    beam_width = 10
    repaired_solution = copy.deepcopy(current)
    all_customers = repaired_solution.nodes
    all_paths = [[current.routes, current.objective()]]

    while True:
        temp_paths = []
        for route, objective in all_paths:
            for i in range(len(route) - 1):
                for node in all_customers:
                    if node not in route:
                        new_solution = route[: i + 1] + [node] + route[i + 1 :]
                        total_reward, total_pen = 0, 0
                        for j in range(NR_INTERMEDIATE_SOLUTION_EVALUATIONS):
                            tour_time, rewards, pen, feas = tour_check(
                                new_solution,
                                current.x,
                                current.adj,
                                -1.0,
                                -1.0,
                                len(current.nodes),
                            )
                            total_reward += rewards
                            total_pen += pen
                        new_objective = (
                            -(total_reward + total_pen)
                            / NR_INTERMEDIATE_SOLUTION_EVALUATIONS
                        )
                        if new_objective <= objective:
                            temp_paths.append([new_solution, new_objective])

        if len(temp_paths) == 0:
            break

        all_paths = sorted(temp_paths, key=lambda x: x[1])[:beam_width]

    repaired_solution.routes = all_paths[0][0]
    return repaired_solution


def cluster_priority_repair_op(current, random_state, **kwargs):
    """
    Repairs the solution using the priority list from the cluster destroy operator.
    Inserts nodes sequentially based on their proximity to cluster centers.
    """
    if not hasattr(current, "priority_list"):
        return random_best_prize_repair(current, random_state, **kwargs)

    unvisited_ordered = current.priority_list
    repaired = copy.deepcopy(current)
    delattr(repaired, "priority_list")

    curr_score = repaired.objective()
    pool = kwargs.get("pool", None)

    for node in unvisited_ordered:
        if node in repaired.routes:
            continue

        new_route = get_best_prize_insertion_for_node(
            node,
            repaired.nodes,
            repaired.routes,
            curr_score,
            repaired.adj,
            repaired.x,
            pool,
        )

        if new_route != repaired.routes:
            repaired.routes = new_route
            curr_score = repaired.objective()

    return repaired
