#!/bin/bash
#SBATCH --mem 16G
#SBATCH -c 25
#SBATCH --time 25:00:00
#SBATCH --array=0-24
#SBATCH --mail-type=FAIL,END

# One independent SMAC run per array task (12h each, 24h for 100 customers).
# Before submitting, once:  uv sync --group tune   (from the repository root)
# Submit:                   sbatch ICB_tune.sh <customers>
# Cap concurrent runs:      sbatch --array=0-24%5 ICB_tune.sh <customers>
# Afterwards:               uv run python tune_icb_alns.py --customers <customers> --aggregate
module load Python3.10
cd ..
uv run --no-sync python tune_icb_alns.py --customers "$1" --run-id "$SLURM_ARRAY_TASK_ID" --workers 25
