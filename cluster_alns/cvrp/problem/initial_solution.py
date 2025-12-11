import random

from cluster_alns.cvrp.utils import (
    compute_route_load,
    get_closest_customer_to_add,
    get_customers_that_can_be_added_to_route,
)
from cluster_alns.utils import NeighborGraph


def compute_initial_solution(current):
    routes = []
    route = []
    unvisited_customers = [i for i in range(1, current.nb_customers + 1)]
    while len(unvisited_customers) != 0:
        if len(route) == 0:
            random_customer = random.choice(unvisited_customers)
            route.append(random_customer)
            unvisited_customers.remove(random_customer)
        else:
            route_load = compute_route_load(route, current.demands_data)
            unvisited_eligible_customers = get_customers_that_can_be_added_to_route(
                route_load,
                current.truck_capacity,
                unvisited_customers,
                current.demands_data,
            )
            if len(unvisited_eligible_customers) == 0:
                routes.append(route)
                route = []  # new_route
                random_customer = random.choice(unvisited_customers)
                route.append(random_customer)
                unvisited_customers.remove(random_customer)
            else:
                closest_unvisited_customer = get_closest_customer_to_add(
                    route,
                    unvisited_eligible_customers,
                    current.dist_matrix_data,
                    current.dist_depot_data,
                )
                route.append(closest_unvisited_customer)
                unvisited_customers.remove(closest_unvisited_customer)

    if route != []:
        routes.append(route)

    current.routes = routes
    current.graph = NeighborGraph(current.nb_customers)

    return current
