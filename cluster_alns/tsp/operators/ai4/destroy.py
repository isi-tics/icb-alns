import copy
import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_alns.utils import normalize_data


def random_removal(current, random_state, degree_of_destruction=None, **kwargs):
    if current.routes == [1, 1]:
        return current

    nodes = current.routes[:-1]
    destroyed_solution = copy.copy(current)

    nr_nodes_to_remove = max(1, round(degree_of_destruction * len(nodes) - 1))

    idx_to_remove = random_state.choice(
        range(1, len(nodes)), nr_nodes_to_remove, replace=False
    )
    destroyed_solution.routes = [
        i for j, i in enumerate(current.routes) if j not in idx_to_remove
    ]

    return destroyed_solution


def relatedness_removal(
    current, random_state, prob=5, degree_of_destruction=None, **kwargs
):
    if current.routes == [1, 1]:
        return current

    destroyed_solution = copy.deepcopy(current)
    visited_customers = list(destroyed_solution.routes[1:-1])

    nr_nodes_to_remove = max(1, round(degree_of_destruction * len(visited_customers)))

    node_to_remove = random_state.choice(visited_customers)
    while node_to_remove in destroyed_solution.routes:
        destroyed_solution.routes.remove(node_to_remove)
        visited_customers.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        normalized_distances = normalize_data(current.adj[node_to_remove - 1])
        related_nodes = [
            (node, normalized_distances[node - 1]) for node in visited_customers
        ]

        if random_state.random() < 1 / prob:
            node_to_remove = random_state.choice(visited_customers)
        else:
            node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
        while node_to_remove in destroyed_solution.routes:
            destroyed_solution.routes.remove(node_to_remove)
            visited_customers.remove(node_to_remove)
    return destroyed_solution


def neighbor_graph_removal(
    current, random_state, degree_of_destruction=None, prob=5, **kwargs
):
    if current.routes == [1, 1]:
        return current
    destroyed_solution = copy.deepcopy(current)
    visited_customers = list(destroyed_solution.routes[1:-1])

    nr_nodes_to_remove = max(1, round(degree_of_destruction * len(visited_customers)))

    values = {}
    route = destroyed_solution.routes[1:-1]

    if len(route) == 1:
        values[route[0]] = current.graph.get_edge_weight(
            1, route[0]
        ) + current.graph.get_edge_weight(route[0], 1)
    else:
        for i in range(len(route)):
            if i == 0:
                values[route[i]] = current.graph.get_edge_weight(
                    1, route[i]
                ) + current.graph.get_edge_weight(route[i], route[1])
            elif i == len(route) - 1:
                values[route[i]] = current.graph.get_edge_weight(
                    route[i - 1], route[i]
                ) + current.graph.get_edge_weight(route[i], 1)
            else:
                values[route[i]] = current.graph.get_edge_weight(
                    route[i - 1], route[i]
                ) + current.graph.get_edge_weight(route[i], route[i + 1])

    removed_nodes = []
    while len(removed_nodes) < nr_nodes_to_remove:
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=False)
        removal_option = 0
        while (
            random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1
        ):
            removal_option += 1
        node_to_remove, score = sorted_nodes[removal_option]

        route.remove(node_to_remove)
        destroyed_solution.routes.remove(node_to_remove)
        removed_nodes.append(node_to_remove)
        values.pop(node_to_remove)

        if len(route) == 0:
            continue

        elif len(route) == 1:
            values[route[0]] = current.graph.get_edge_weight(
                1, route[0]
            ) + current.graph.get_edge_weight(route[0] - 1, 1)
        else:
            for i in range(len(route)):
                if i == 0:
                    values[route[i]] = current.graph.get_edge_weight(
                        1, route[i]
                    ) + current.graph.get_edge_weight(route[i], route[1])
                elif i == len(route) - 1:
                    values[route[i]] = current.graph.get_edge_weight(
                        route[i - 1], route[i]
                    ) + current.graph.get_edge_weight(route[i], 1)
                else:
                    values[route[i]] = current.graph.get_edge_weight(
                        route[i - 1], route[i]
                    ) + current.graph.get_edge_weight(route[i], route[i + 1])

    return destroyed_solution


def _get_node_features(current, nodes_list):
    features = []
    all_features = current.x
    for node_id in nodes_list:
        idx = node_id - 1
        x_coord = all_features[idx][0]
        y_coord = all_features[idx][1]
        prize = all_features[idx][-2]
        features.append([node_id, x_coord, y_coord, prize])
    return pd.DataFrame(features, columns=["id", "x", "y", "prize"])


def cluster_representative_removal_op(
    current, random_state, degree_of_destruction=None, **kwargs
):
    """
    Destroys the solution using K-Means clustering on (X, Y, Prize).
    Preserves cluster representatives and removes outliers based on destruction degree.
    """
    if len(current.routes) <= 2:
        return current

    destroyed_solution = copy.deepcopy(current)

    served_nodes = list(set(destroyed_solution.routes))
    if 1 in served_nodes:
        served_nodes.remove(1)

    k_optimal = getattr(current, "k_optimal_instance", 5)
    if "k_optimal" in kwargs:
        k_optimal = kwargs["k_optimal"]

    # Configuration flag: Default to True (Latent Space)
    use_pca = kwargs.get("use_pca", True)

    if len(served_nodes) < max(k_optimal, 3):
        return random_removal(
            current, random_state, degree_of_destruction=degree_of_destruction
        )

    try:
        df_nodes = _get_node_features(current, served_nodes)
        features = df_nodes[["x", "y", "prize"]].values

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(features)

        if use_pca:
            pca = PCA(n_components=2)
            data_to_cluster = pca.fit_transform(x_scaled)
        else:
            data_to_cluster = x_scaled

        current_k = min(k_optimal, len(served_nodes))
        if current_k <= 1:
            return destroyed_solution

        kmeans = KMeans(n_clusters=current_k, random_state=random_state, n_init=1)
        df_nodes["cluster"] = kmeans.fit_predict(data_to_cluster)
        centers = kmeans.cluster_centers_
    except Exception:
        return random_removal(
            current, random_state, degree_of_destruction=degree_of_destruction
        )

    ids_to_remove_list = []
    deg = degree_of_destruction if degree_of_destruction is not None else 0.3

    for i in range(current_k):
        cluster_mask = df_nodes["cluster"] == i
        nodes_in_cluster = df_nodes[cluster_mask].copy()
        if nodes_in_cluster.empty:
            continue
        if i >= len(centers):
            continue

        centroid = centers[i]
        indices = nodes_in_cluster.index
        x_local = data_to_cluster[indices]

        dists = np.linalg.norm(x_local - centroid, axis=1)
        nodes_in_cluster["dist_centroid"] = dists

        n_total = len(nodes_in_cluster)
        n_remove = int(np.round(n_total * deg))
        n_remove = max(1, min(n_remove, n_total - 1))
        n_keep = n_total - n_remove

        remove_df = nodes_in_cluster.sort_values("dist_centroid", ascending=True).iloc[
            n_keep:
        ]
        ids_to_remove_list.append(remove_df[["id", "dist_centroid"]])

    if not ids_to_remove_list:
        return destroyed_solution

    df_outliers = pd.concat(ids_to_remove_list).sort_values(
        "dist_centroid", ascending=True
    )
    priority_list = df_outliers["id"].tolist()

    destroyed_solution.priority_list = priority_list

    ids_set = set(priority_list)
    destroyed_solution.routes = [n for n in destroyed_solution.routes if n not in ids_set]

    return destroyed_solution
