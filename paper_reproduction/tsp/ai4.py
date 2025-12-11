import argparse
from pathlib import Path

from cluster_alns.runners.alns.tsp.ai4_runner import AI4TSPRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ALNS on AI4TSP Instances",
        description="Runs the ALNS algorithm on AI4 instances with or without cluster operators.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The JSON filename containing the parameters for the ALNS algorithm.",
        default="ALNS_ai4_20_100.json",
    )
    args = parser.parse_args()
    current_path = Path(__file__).parent.resolve()
    params_path = current_path / f"configs/{args.config}"
    instance_path = current_path / "data"
    runner = AI4TSPRunner(params_path, instance_path)
    runner()
    runner.write_results(current_path / "results")
