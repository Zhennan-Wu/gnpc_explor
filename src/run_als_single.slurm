#!/bin/bash
#SBATCH --job-name=dahlou
#SBATCH --output=logs/als_%A_%a.out
#SBATCH --error=logs/als_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=3-00:00:00               
#SBATCH --nodes=1                       # OPTIMIZED: 1 node per job
#SBATCH --ntasks=1                      # OPTIMIZED: 1 task
#SBATCH --cpus-per-task=4               # OPTIMIZED: 4 cores total (1 for each MCMC chain)
#SBATCH --mem=32G                       # OPTIMIZED: 32GB is plenty for one Stan empirical run                      
#SBATCH --array=1-4                     # KEEPS: Iterates through the 4 lines in als_exp.txt           

# --- VENV ACTIVATION ---
module load python/3.12.4 
module load gcc
module load scipy-stack
source /home/dzhao268/envs/emlou/bin/activate
# ------------------------

export CXX=g++
export CC=gcc

CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" als_exp.txt)
SCENARIO=$(echo $CONFIG_LINE | awk '{print $1}')
MODEL=$(echo $CONFIG_LINE | awk '{print $2}')

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Empirical Data: Scenario $SCENARIO with Model $MODEL"

# Pre-compile the model sequentially to avoid race conditions
python dahlou_hpc_mp.py --scenario "$SCENARIO" --models "$MODEL" --compile_only

# Run the single empirical execution (using 4 chains and 2500/2500 iterations for final inference)
python dahlou_hpc_mp.py --scenario "$SCENARIO" --models "$MODEL" --chains 4 --warmup 2500 --sampling 2500

echo "Job complete."