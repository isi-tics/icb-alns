import argparse
from pathlib import Path

from cluster_alns.runners.alns.cvrp.runner import CVRPRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ALNS CVRP",
        description="Runs the ALNS algorithm on CVRP instances with or without cluster operators.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The JSON filename containing the parameters for the ALNS algorithm.",
        default="ALNS_pure_100_1000_5000.json",
    )
    args = parser.parse_args()
    current_path = Path(__file__).parent.resolve()
    params_path = current_path / f"configs/{args.config}"
    instance_path = current_path / "data"
    runner = CVRPRunner(params_path, instance_path)
    runner()
    runner.write_results(current_path / "results")
