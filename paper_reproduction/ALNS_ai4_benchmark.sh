#!/bin/bash
#SBATCH --mem 16G
#SBATCH -c 24
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
INSTANCE_DIR="$TMP_DIR/paper_reproduction/tsp"
cd $INSTANCE_DIR
for i in 20 50 100; do
  for j in original kmeans pca; do
    echo "Running instances with $i customers using $j mode"
    uv run python ai4.py --config "ALNS_ai4_"$j"_"$i".json"
    rsync -av results/ $PROJECT_DIR/paper_reproduction/tsp/results/
  done
done

uv run python analysis_ai4.py
cp AI4_results_* $PROJECT_DIR/paper_reproduction/tsp/
