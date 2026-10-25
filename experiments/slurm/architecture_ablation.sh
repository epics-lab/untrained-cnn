#!/bin/bash
#SBATCH --job-name=uncnn_architecture_ablation
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=architecture_ablation_%j.out

# From experiments/: sbatch --account=<account> slurm/architecture_ablation.sh
# The ablation runtime column depends on the number of CPUs.
set -eo pipefail
[[ -n "$VENV" ]] && source "$VENV/bin/activate"
export MPLBACKEND=Agg

jupyter nbconvert --to notebook --execute 05_architecture_ablation.ipynb --ExecutePreprocessor.timeout=-1 \
    --output-dir outputs/executed --output 05_architecture_ablation
