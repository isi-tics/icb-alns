import copy
import math
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator


def read_input_tsp(filename, instance_nr):
    data = pd.read_pickle(filename)
    customers_x = [x for x, y in data[instance_nr]]
    customers_y = [y for x, y in data[instance_nr]]
    distance_matrix = compute_distance_matrix(customers_x, customers_y)

    return len(customers_x), distance_matrix


# Compute the distance matrix
def compute_distance_matrix(customers_x, customers_y):
    nb_customers = len(customers_x)
    distance_matrix = np.zeros((nb_customers + 1, nb_customers + 1))
    for i in range(nb_customers):
        distance_matrix[i][i] = 0
        for j in range(nb_customers):
            dist = compute_dist(
                customers_x[i], customers_x[j], customers_y[i], customers_y[j]
            )
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix


def compute_dist(xi, xj, yi, yj):
    exact_dist = math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2))
    return exact_dist


def determine_nr_nodes_to_remove(
    nb_customers, omega_bar_minus=5, omega_minus=0.1, omega_bar_plus=30, omega_plus=0.4
):
    n_plus = min(omega_bar_plus, omega_plus * nb_customers)
    n_minus = min(n_plus, max(omega_bar_minus, omega_minus * nb_customers))
    r = random.randint(round(n_minus), round(n_plus))
    return r


def update_neighbor_graph(current, route, new_route_quality):
    prev_node = route[-1]
    for i in range(len(route)):
        curr_node = route[i]
        prev_edge_weight = current.graph.get_edge_weight(prev_node, curr_node)
        if new_route_quality < prev_edge_weight:
            current.graph.update_edge(prev_node, curr_node, new_route_quality)
        prev_node = curr_node
    return current.graph


def ai4_update_neighbor_graph(current, route, new_route_quality):
    graph = copy.copy(current.graph)
    edge_weights = [
        graph.get_edge_weight(route[i - 1], route[i]) for i in range(1, len(route))
    ]
    updated_edges = [
        (route[i - 1], route[i])
        for i in range(1, len(route))
        if new_route_quality < edge_weights[i - 1]
    ]
    for edge in updated_edges:
        graph.update_edge(edge[0], edge[1], new_route_quality)
    return graph


def tour_check(tour, x, time_matrix, maxT_pen, tw_pen, n_nodes):
    """
    Calculate a tour times and the penalties for constraint violation
    """
    tw_high = x[:, -3]
    tw_low = x[:, -4]
    prizes = x[:, -2]
    maxT = x[0, -1]

    feas = True
    return_to_depot = False
    tour_time = 0
    rewards = 0
    pen = 0

    for i in range(len(tour) - 1):
        node = int(tour[i])
        if i == 0:
            assert node == 1, "A tour must start from the depot - node: 1"

        succ = int(tour[i + 1])
        time = time_matrix[node - 1][succ - 1]
        noise = np.random.randint(1, 101, size=1)[0] / 100
        tour_time += np.round(noise * time, 2)
        if tour_time > tw_high[succ - 1]:
            feas = False
            # penalty added for each missed tw
            pen += tw_pen
        elif tour_time < tw_low[succ - 1]:
            tour_time += tw_low[succ - 1] - tour_time
            rewards += prizes[succ - 1]
        else:
            rewards += prizes[succ - 1]

        if succ == 1:
            return_to_depot = True
            break

    if not return_to_depot:
        raise Exception("A tour must reconnect back to the depot - node: 1")

    if tour_time > maxT:
        # penalty added for each
        pen += maxT_pen * n_nodes
        feas = False

    return tour_time, rewards, pen, feas

def find_optimal_k_elbow_op(x_matrix, random_state, max_k=15):
    """
    Calculates K-Optimal for Orienteering using (X, Y, Prize).
    Assumes x_matrix columns: [0]=X, [1]=Y, [-2]=Prize
    """
    data = x_matrix[:, [0, 1, -2]]
    
    if len(data) < 3:
        return 2
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data)
    
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)
    
    inertias = []
    limit_k = min(max_k + 1, len(x_pca))
    k_range = range(2, limit_k)
    
    if len(k_range) == 0:
        return 2
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(x_pca)
        inertias.append(km.inertia_)
        
    kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
    return kl.elbow if kl.elbow else 3