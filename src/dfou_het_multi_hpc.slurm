#!/bin/bash
#SBATCH --job-name=dfouls_het_multi
#SBATCH --output=logs/dfouls_het_multi_task_%a.out
#SBATCH --error=logs/dfouls_het_multi_task_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --array=0-119             
#SBATCH --time=4-00:00:00           # Healthy buffer for CPU execution
#SBATCH --partition=general       # Big Red 200 standard partition (NO GPU requested)
#SBATCH --nodes=1                 
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=8         # Restricts to 8 EPYC cores per job
#SBATCH --mem=16G                 # Optimized memory footprint
#SBATCH -A r00939

# Load the necessary Python module
module load conda 
module load gcc
module load cuda

# This hooks Conda into the non-interactive Slurm shell
eval "$(conda shell.bash hook)"
conda activate dfou
# ------------------------

# Prevent C-level threading over-subscription (Critical for Big Red 200)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Setup directories
mkdir -p logs
mkdir -p dfouls_het_multi_results

# Execute the job
python dfou_het_multi_hpc.py --task_id $SLURM_ARRAY_TASK_ID --n_runs 5 --out_dir dfouls_het_multi_results

# python dfou_het_multi_hpc.py --merge --out_dir dfouls_het_multi_results