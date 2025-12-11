import csv
import json
import pathlib
from pathlib import Path

import numpy as np


def write_output(
    folder,
    exp_name,
    problem_instance,
    seed,
    iterations,
    solution,
    best_objective,
    instance_file,
):
    """Save outputs in files"""
    output_dir = folder
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Final pop
    with open(output_dir + exp_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "problem_instance",
                "rseed",
                "iterations",
                "solution",
                "best_objective",
                "instance_file",
            ]
        )
        writer.writerow(
            [
                problem_instance,
                seed,
                iterations,
                solution,
                best_objective,
                instance_file,
            ]
        )


def writeJSONfile(data, path):
    with open(path, "w") as write_file:
        json.dump(data, write_file, indent=4)
        write_file.write("\n")


def readJSONFile(file, check_if_exists=False):
    """This function reads any json file and returns a dictionary."""
    if (not Path(file).is_file()) and check_if_exists:
        raise FileNotFoundError(f"The file {file} does not exist.")
    with open(file) as f:
        data = json.load(f)
    return data


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class NeighborGraph:
    def __init__(self, num_nodes):
        self.graph = np.full((num_nodes + 1, num_nodes + 1), np.inf, dtype=np.float64)

    def update_edge(self, node_a, node_b, cost):
        # graph is kept single directional
        self.graph[node_a][node_b] = cost

    def get_edge_weight(self, node_a, node_b):
        return self.graph[node_a][node_b]
