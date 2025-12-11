import argparse
import os

import numpy as np
import pandas as pd


def my_sort(x):
    method_sort_dict = {
        "original": 0,
        "KMeans": 1,
        "PCA": 2,
        "KMeans_Vanilla": 3,
        "PCA_Vanilla": 4,
    }
    _, method, customers, _ = x.split("-")
    customers = int(customers)
    return customers + method_sort_dict[method]


if __name__ == "__main__":
    results = [
        path
        for path in os.listdir("results")
        if path.startswith("AI4") and path.endswith(".csv")
    ]
    results.sort(key=my_sort)
    result_tables = {
        customers: pd.DataFrame(columns=["Avg", "Nr. Best", "Time/Instance"])
        for customers in [20, 50, 100]
    }
    for path in results:
        table = pd.read_csv(f"results/{path}", sep=";")
        _, method, customers, _ = path.split("-")
        avg_score = -table["best_objective"].mean()
        avg_time = np.round(table["exp_time"].mean(), 2)
        nr_best = table["best_objective"].argmin() + 1
        result_tables[int(customers)].loc[method] = [
            avg_score,
            nr_best,
            avg_time,
        ]
    for customers, table in result_tables.items():
        table.index.name = "Method"
        table["Nr. Best"] = table["Nr. Best"].astype(int)
        table.to_csv(f"AI4_results_{customers}.csv")
