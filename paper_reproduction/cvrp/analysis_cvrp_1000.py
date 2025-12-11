import argparse
import os

import numpy as np
import pandas as pd


def my_sort(x):
    method_sort_dict = {
        "Original": 0,
        "RL": 1,
        "Cluster": 2,
    }
    method = x.split("_")[0]
    return method_sort_dict[method]


if __name__ == "__main__":
    results = [path for path in os.listdir(".") if path.endswith("1000_result.csv")]
    results.sort(key=my_sort)
    df = pd.DataFrame(columns=["Avg", "Nr. Best", "Time/Instance"])
    for path in results:
        table = pd.read_csv(path, sep=";")
        method = path.split("_")[0]
        avg_score = table["best_objective"].mean()
        avg_time = np.round(table["exp_time"].mean(), 2)
        nr_best = table["best_objective"].argmax() + 1
        df.loc[method] = [
            avg_score,
            nr_best,
            avg_time,
        ]
    df.index.name = "Method"
    df["Nr. Best"] = df["Nr. Best"].astype(int)
    df.to_csv("CVRP_1000_results.csv")
