#!/bin/bash
#SBATCH --job-name=tcr-h-paper
#SBATCH --output=logs/tcr-h_%j.out
#SBATCH --error=logs/tcr-h_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu6348
#SBATCH --qos=52cores

set -euo pipefail

echo "=== Job started: $(date) ==="
hostname
echo "=== Training started: $(date) ==="

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

conda run -n tcr-h python scripts/train_final.py

echo ""
echo "=== Job finished: $(date) ==="
echo "Results in results/final_hpc/"
