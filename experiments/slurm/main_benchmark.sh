#!/bin/bash
#SBATCH --job-name=uncnn_main_benchmark
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=main_benchmark_%A_%a.out

# From experiments/, with the Python environment active or VENV set: sbatch --account=<account> slurm/main_benchmark.sh
# One array task per dataset: 0 PPMI, 1 NKI, 2 HBN. Then run 02_main_benchmark_results.ipynb.
set -eo pipefail
[[ -n "$VENV" ]] && source "$VENV/bin/activate"
export MPLBACKEND=Agg  # a submission from Jupyter would pass its inline backend

DATASETS=(PPMI NKI HBN)
export DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
jupyter nbconvert --to notebook --execute 01_main_benchmark.ipynb --ExecutePreprocessor.timeout=-1 \
    --output-dir outputs/executed --output "01_main_benchmark_${DATASET}"
