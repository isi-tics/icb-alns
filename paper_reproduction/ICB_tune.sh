#!/bin/bash
#SBATCH --mem 16G
#SBATCH -c 25
#SBATCH --time 25:00:00
#SBATCH --array=0-24
#SBATCH --mail-type=FAIL,END

# One independent SMAC run per array task: sbatch ICB_tune.sh <customers>
# To cap how many runs execute at once: sbatch --array=0-24%5 ICB_tune.sh <customers>
# Afterwards: uv run python tune_icb_alns.py --customers <customers> --aggregate
module load Python3.10
cd ..
uv sync --group tune
uv run python tune_icb_alns.py --customers "$1" --run-id "$SLURM_ARRAY_TASK_ID" --workers 25
