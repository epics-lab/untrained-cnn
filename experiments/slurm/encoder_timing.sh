#!/bin/bash
#SBATCH --job-name=uncnn_encoder_timing
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=encoder_timing_%j.out

# From experiments/: sbatch --account=<account> slurm/encoder_timing.sh
# Timing runs on one thread; the second CPU keeps Python and OS housekeeping off the timed core.
set -eo pipefail
[[ -n "$VENV" ]] && source "$VENV/bin/activate"
export MPLBACKEND=Agg

jupyter nbconvert --to notebook --execute 03_encoder_timing.ipynb --ExecutePreprocessor.timeout=-1 \
    --output-dir outputs/executed --output 03_encoder_timing
