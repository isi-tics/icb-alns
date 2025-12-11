import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def my_sort(x):
    method_sort_dict = {
        "original": 0,
        "RL": 1,
        "KMeans": 2,
        "PCA": 3,
        "KMeans_Vanilla": 4,
        "PCA_Vanilla": 5,
    }
    _, method, customers, _ = x.split("-")
    customers = int(customers)
    return customers + method_sort_dict[method]


if __name__ == "__main__":
    results = [
        path
        for path in os.listdir("results")
        if path.startswith("AI4") and path.endswith("20-1.csv")
    ]
    results.sort(key=my_sort)
    objectives = {
        path.split("-")[1]: pd.read_csv(f"results/{path}", sep=";")[
            "training_objectives"
        ].values[70]
        for path in results
    }
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    for method, objective in objectives.items():
        obj = objective[1:-1]
        obj = np.fromstring(obj, sep=",")[:100]
        ax.plot(
            range(len(obj)),
            -obj,
            label=method,
        )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Training Objective")
    ax.legend()
    plt.tight_layout()
    plt.show()
