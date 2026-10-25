#!/bin/bash
#SBATCH --job-name=uncnn_head_timing
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=head_timing_%j.out

# From experiments/, after the NKI run of 01_main_benchmark.ipynb: sbatch --account=<account> slurm/head_timing.sh
set -eo pipefail
[[ -n "$VENV" ]] && source "$VENV/bin/activate"
export MPLBACKEND=Agg

jupyter nbconvert --to notebook --execute 04_head_timing.ipynb --ExecutePreprocessor.timeout=-1 \
    --output-dir outputs/executed --output 04_head_timing
