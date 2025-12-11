import copy
import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_alns.cvrp.utils import determine_nr_nodes_to_remove
from cluster_alns.utils import normalize_data


def random_removal(current, random_state, nr_nodes_to_remove=None, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    visited_customers = [c for route in destroyed_solution.routes for c in route]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(
            destroyed_solution.nb_customers
        )

    nodes_to_remove = random.sample(visited_customers, nr_nodes_to_remove)

    new_routes = []
    for route in destroyed_solution.routes:
        new_route = [node for node in route if node not in nodes_to_remove]
        if new_route:
            new_routes.append(new_route)
    destroyed_solution.routes = new_routes

    return destroyed_solution


# --- relatedness destroy method ---

# see: Shaw - Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems
# see: Santini, Ropke - A comparison of acceptance criteria for the adaptive large neighbourhood search metaheuristic


def relatedness_removal(current, random_state, nr_nodes_to_remove=None, prob=5, **kwargs):
    destroyed_solution = copy.deepcopy(current)
    visited_customers = [
        customer for route in destroyed_solution.routes for customer in route
    ]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(
            destroyed_solution.nb_customers
        )

    node_to_remove = random_state.choice(visited_customers)
    for route in destroyed_solution.routes:
        while node_to_remove in route:
            route.remove(node_to_remove)
            visited_customers.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        related_nodes = []
        normalized_distances = normalize_data(
            destroyed_solution.dist_matrix_data[node_to_remove - 1]
        )
        route_node_to_remove = [
            route for route in current.routes if node_to_remove in route
        ][0]
        for route in destroyed_solution.routes:
            for node in route:
                if node in route_node_to_remove:
                    related_nodes.append((node, normalized_distances[node - 1]))
                else:
                    related_nodes.append((node, normalized_distances[node - 1] + 1))

        if random_state.random() < 1 / prob:
            node_to_remove = random_state.choice(visited_customers)
        else:
            node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
        for route in destroyed_solution.routes:
            while node_to_remove in route:
                route.remove(node_to_remove)
                visited_customers.remove(node_to_remove)
    destroyed_solution.routes = [
        route for route in destroyed_solution.routes if route != []
    ]

    return destroyed_solution


# --- neighbor/history graph removal
# see: A unified heuristic for a large class of Vehicle Routing Problems with Backhauls
def neighbor_graph_removal(current, random_state, nr_nodes_to_remove=None, prob=5, **kwargs):
    destroyed_solution = copy.deepcopy(current)

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(
            destroyed_solution.nb_customers
        )

    values = {}
    for route in destroyed_solution.routes:
        if len(route) == 1:
            values[route[0]] = current.graph.get_edge_weight(
                0, route[0]
            ) + current.graph.get_edge_weight(route[0], 0)
        else:
            for i in range(len(route)):
                if i == 0:
                    values[route[i]] = current.graph.get_edge_weight(
                        0, route[i]
                    ) + current.graph.get_edge_weight(route[i], route[1])
                elif i == len(route) - 1:
                    values[route[i]] = current.graph.get_edge_weight(
                        route[i - 1], route[i]
                    ) + current.graph.get_edge_weight(route[i], 0)
                else:
                    values[route[i]] = current.graph.get_edge_weight(
                        route[i - 1], route[i]
                    ) + current.graph.get_edge_weight(route[i], route[i + 1])

    removed_nodes = []
    while len(removed_nodes) < nr_nodes_to_remove:
        # sort the nodes based on their neighbor graph scores in descending order
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
        # select the node to remove
        removal_option = 0
        while (
            random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1
        ):
            removal_option += 1
        node_to_remove, score = sorted_nodes[removal_option]

        # remove the node from its route
        for route in destroyed_solution.routes:
            if node_to_remove in route:
                route.remove(node_to_remove)
                removed_nodes.append(node_to_remove)

                values.pop(node_to_remove)
                if len(route) == 0:
                    destroyed_solution.routes.remove([])

                elif len(route) == 1:
                    values[route[0]] = current.graph.get_edge_weight(
                        0, route[0]
                    ) + current.graph.get_edge_weight(route[0], 0)
                else:
                    for i in range(len(route)):
                        if i == 0:
                            values[route[i]] = current.graph.get_edge_weight(
                                0, route[i]
                            ) + current.graph.get_edge_weight(route[i], route[1])
                        elif i == len(route) - 1:
                            values[route[i]] = current.graph.get_edge_weight(
                                route[i - 1], route[i]
                            ) + current.graph.get_edge_weight(route[i], 0)
                        else:
                            values[route[i]] = current.graph.get_edge_weight(
                                route[i - 1], route[i]
                            ) + current.graph.get_edge_weight(route[i], route[i + 1])

                break

    return destroyed_solution


def _get_customer_features(current, served_customers):
    features = []
    for customer_id in served_customers:
        idx = customer_id - 1
        x = current.customers_x[idx]
        y = current.customers_y[idx]
        demand = current.demands[idx]
        features.append([customer_id, x, y, demand])
    return pd.DataFrame(features, columns=["id", "x", "y", "demand"])


def cluster_representative_removal(
    current, random_state, nr_nodes_to_remove=None, **kwargs
):
    """
    Destroys the solution by clustering customers based on spatial and demand features.
    It preserves cluster representatives (centroids) and removes outliers.
    """
    destroyed_solution = copy.deepcopy(current)
    served_customers = [c for route in destroyed_solution.routes for c in route]

    k_optimal = getattr(current, "k_optimal", 5)
    use_pca = kwargs.get("use_pca", True)

    if len(served_customers) < max(k_optimal, 3):
        return random_removal(current, random_state, nr_nodes_to_remove)

    try:
        df_customers = _get_customer_features(current, served_customers)
        features = df_customers[["x", "y", "demand"]].values

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)

        if use_pca:
            pca = PCA(n_components=2)
            x_scaled = pca.fit_transform(x_scaled)

        current_k = min(k_optimal, len(served_customers))
        if current_k <= 1:
            return random_removal(current, random_state, nr_nodes_to_remove)

        kmeans = KMeans(n_clusters=current_k, random_state=random_state, n_init=1)
        df_customers["cluster"] = kmeans.fit_predict(x_scaled)
        centers = kmeans.cluster_centers_

    except Exception:
        return random_removal(current, random_state, nr_nodes_to_remove)

    ids_to_remove_list = []

    total_customers = len(served_customers)
    target_remove = (
        nr_nodes_to_remove
        if nr_nodes_to_remove is not None
        else int(total_customers * 0.2)
    )
    ratio = target_remove / total_customers if total_customers > 0 else 0.2

    for i in range(current_k):
        cluster_mask = df_customers["cluster"] == i
        cluster_nodes = df_customers[cluster_mask].copy()

        if cluster_nodes.empty:
            continue
        if i >= len(centers):
            continue

        centroid = centers[i]
        indices = cluster_nodes.index
        x_local = x_scaled[indices]

        dists = np.linalg.norm(x_local - centroid, axis=1)
        cluster_nodes["dist_centroid"] = dists

        n_local = len(cluster_nodes)
        n_remove_local = int(np.round(n_local * ratio))
        n_remove_local = max(1, min(n_remove_local, n_local - 1))
        n_keep = n_local - n_remove_local

        to_remove = cluster_nodes.sort_values("dist_centroid", ascending=True).iloc[
            n_keep:
        ]
        ids_to_remove_list.append(to_remove[["id", "dist_centroid"]])

    if not ids_to_remove_list:
        return destroyed_solution

    df_outliers = pd.concat(ids_to_remove_list).sort_values(
        "dist_centroid", ascending=True
    )
    ids_to_remove_set = set(df_outliers["id"].tolist())

    destroyed_solution.priority_list = df_outliers["id"].tolist()

    new_routes = []
    for route in destroyed_solution.routes:
        new_route = [node for node in route if node not in ids_to_remove_set]
        if new_route:
            new_routes.append(new_route)

    destroyed_solution.routes = new_routes

    return destroyed_solution
