#!/bin/bash
#SBATCH --mem 16G
#SBATCH -c 24
#SBATCH --time 12:00:00
#SBATCH --array=0-119
#SBATCH --mail-type=FAIL,END

# Final evaluation, one (size, model, seed) per array task: 3 sizes x 4 models x 10 seeds.
# Before submitting, once:  uv sync --group tune   (from the repository root)
#                           cd tsp && uv run python run_seeds.py --prepare
# Submit:                   sbatch ICB_seeds.sh        (cap concurrency with --array=0-119%20)
# Afterwards:               cd tsp && uv run python run_seeds.py --summary
SIZES=(20 50 100)
MODELS=(original kmeans pca rl)
i=$SLURM_ARRAY_TASK_ID
size=${SIZES[$((i / 40))]}
model=${MODELS[$((i / 10 % 4))]}
seed=$((i % 10 + 1))

module load Python3.10
cd tsp
uv run --no-sync python run_seeds.py --sizes "$size" --models "$model" --seeds "$seed"
