#!/bin/bash
#SBATCH --job-name=lou_aggregate
#SBATCH -p general
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
#SBATCH --time=00:15:00               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1             
#SBATCH --mem=128G               
#SBATCH -A r00939         

# --- CONDA ACTIVATION ---
module load conda 
module load gcc

# This hooks Conda into the non-interactive Slurm shell
eval "$(conda shell.bash hook)"
conda activate ou
# ------------------------

echo "Starting aggregation process..."
python aggregate_results.py
echo "Aggregation complete!"