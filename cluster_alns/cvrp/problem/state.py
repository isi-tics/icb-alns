import copy


class CVRPState:
    def __init__(
        self,
        initial_solution,
        nb_customers,
        truck_capacity,
        dist_matrix_data,
        dist_depot_data,
        demands_data,
        customers_x,
        customers_y,
        k_optimal,
        seed,
    ):
        self.nb_customers = nb_customers
        self.truck_capacity = truck_capacity
        self.dist_matrix_data = dist_matrix_data
        self.dist_depot_data = dist_depot_data
        self.demands_data = demands_data
        self.customers_x = customers_x
        self.customers_y = customers_y
        self.k_optimal = k_optimal
        self.seed = seed

        self.routes = initial_solution

    def objective(self, best=False):
        score = self.evaluate_solution(
            self.routes,
            self.dist_matrix_data,
            self.dist_depot_data,
        )
        return score

    @staticmethod
    def evaluate_solution(routes, dist_matrix_data, dist_depot_data):
        total_distance_travelled = 0

        for route in routes:
            total_distance_travelled += (
                dist_depot_data[route[0] - 1] + dist_depot_data[route[-1] - 1]
            )
            for i in range(len(route) - 1):
                total_distance_travelled += dist_matrix_data[route[i] - 1][
                    route[i + 1] - 1
                ]

        return total_distance_travelled

    def copy(self):
        """
        Cria uma cópia do estado atual.
        """
        new_routes = copy.deepcopy(self.routes)
        new_env = CVRPState(
            new_routes,
            self.nb_customers,
            self.truck_capacity,
            self.dist_matrix_data,
            self.dist_depot_data,
            self.demands_data,
            self.customers_x,
            self.customers_y,
            self.problem_instance,
            self.seed,
        )
        if hasattr(self, "graph"):
            new_env.graph = copy.deepcopy(self.graph)
        return new_env

    def remove_clientes(self, ids_para_remover_set):
        """
        Remove um conjunto (set) de IDs de cliente de todas as rotas.
        """
        for route in self.routes:
            route[:] = [cust for cust in route if cust not in ids_para_remover_set]

        # Limpar rotas vazias
        self.routes = [route for route in self.routes if route != []]
