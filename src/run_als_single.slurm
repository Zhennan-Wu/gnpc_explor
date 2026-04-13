#!/bin/bash
#SBATCH --job-name=dahlou
#SBATCH --output=logs/als_%A_%a.out
#SBATCH --error=logs/als_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=4-00:00:00               
#SBATCH --nodes=1                       # OPTIMIZED: 1 node per job
#SBATCH --ntasks=1                      # OPTIMIZED: 1 task
#SBATCH --cpus-per-task=4               # OPTIMIZED: 4 cores total (1 for each MCMC chain)
#SBATCH --mem=32G                       # OPTIMIZED: 32GB is plenty for one Stan empirical run                      
#SBATCH --array=1-4                     # KEEPS: Iterates through the 4 lines in als_exp.txt    
#SBATCH -A r00939       

# --- VENV ACTIVATION ---
module load conda

eval "$(conda shell.bash hook)"
conda activate emlou
# ------------------------

export CXX=g++
export CC=gcc

# Create a unique temp dir for THIS specific array task
export TMPDIR=/N/scratch/zwu1/tmp/job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR


CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" als_exp.txt)
SCENARIO=$(echo $CONFIG_LINE | awk '{print $1}')
MODEL=$(echo $CONFIG_LINE | awk '{print $2}')

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Empirical Data: Scenario $SCENARIO with Model $MODEL"

# Pre-compile the model sequentially to avoid race conditions

# --- ARVIZ RACE CONDITION FIX ---
# Pre-create the cache folder and stamp it with today's date 
# so the parallel Python processes don't fight over file locks.
mkdir -p /N/u/zwu1/Quartz/.cache/arviz
date +%Y-%m-%d > /N/u/zwu1/Quartz/.cache/arviz/daily_warning
# --------------------------------

# Run the single empirical execution (using 4 chains and 2500/2500 iterations for final inference)
python dahlou_hpc_mp.py --scenario "$SCENARIO" --models "$MODEL" --chains 4 --warmup 2500 --sampling 2500


wait

# Clean up the temp directory when the node finishes to save inodes
rm -rf $TMPDIR

echo "Job complete."