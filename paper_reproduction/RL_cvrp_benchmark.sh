#!/bin/bash
#SBATCH --mem 32G
#SBATCH -c 31
#SBATCH --gpus 1
#SBATCH -p short-simple
#SBATCH --mail-type=FAIL,END,ARRAY_TASKS
#SBATCH --mail-user=mgm4@cin.ufpe.br

# Load modules and activate python environment
module load Python3.10
cd ..
export UV_CACHE_DIR=/tmp/uv-cache/
PROJECT_DIR=$(pwd)
TMP_DIR=/tmp/$USER/DR-ALNS
mkdir -p $TMP_DIR

echo "Copying project files to $TMP_DIR"
rsync -av --exclude='.venv' --exclude='.git/' --exclude='__pycache__/' $PROJECT_DIR/ $TMP_DIR/

echo "Changing to temporary directory $TMP_DIR"
cd $TMP_DIR

echo "Creating Python virtual environment"
uv venv --clear --no-cache --link-mode hardlink

echo "Installing project dependencies"
uv sync --frozen
uv pip install -e .

echo "Starting ALNS experiments for AI4Benchmark instances"
INSTANCE_DIR="$TMP_DIR/paper_reproduction/cvrp"
cd $INSTANCE_DIR
for i in 1000 10000; do
  for j in original kmeans pca; do
    echo "Running instances with $i customers using $j mode"
    uv run python rl_cvrp.py --config $j"_rl_"$i".yml"
  done
done
