#!/bin/bash
#SBATCH --job-name=dahlou_hd
#SBATCH -p general
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=4-00:00:00               
#SBATCH --nodes=5                       
#SBATCH --ntasks=5                      
#SBATCH --cpus-per-task=128             
#SBATCH --mem=0                      
#SBATCH --array=13-48                   
#SBATCH -A r00939           

# --- CONDA ACTIVATION ---
module load conda 
module load gcc

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


CHAINS=3

CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments_hd.txt)
SCENARIO=$(echo $CONFIG_LINE | awk '{print $1}')
MODEL=$(echo $CONFIG_LINE | awk '{print $2}')

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Scenario: $SCENARIO with Model: $MODEL using 5 Nodes"

# FIX: Pre-compile the model sequentially to avoid race conditions!
# echo "Pre-compiling Stan Model..."
# python lou_simulation_hpc_mp_hd.py --scenario "$SCENARIO" --models "$MODEL" --compile_only

# echo "Compilation step done. Starting parallel instances..."
# --- ARVIZ RACE CONDITION FIX ---
# Pre-create the cache folder and stamp it with today's date 
# so the parallel Python processes don't fight over file locks.
mkdir -p /N/u/zwu1/BigRed200/.cache/arviz
date +%Y-%m-%d > /N/u/zwu1/BigRed200/.cache/arviz/daily_warning

# We split 200 simulations across 5 nodes (40 sims per node).
# The '&' puts the process in the background. '--exclusive' ensures they don't fight for the same resources.
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp_hd.py --scenario "$SCENARIO" --models "$MODEL" --start_run 1 --end_run 40 &
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp_hd.py --scenario "$SCENARIO" --models "$MODEL" --start_run 41 --end_run 80 &
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp_hd.py --scenario "$SCENARIO" --models "$MODEL" --start_run 81 --end_run 120 &
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp_hd.py --scenario "$SCENARIO" --models "$MODEL" --start_run 121 --end_run 160 &
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp_hd.py --scenario "$SCENARIO" --models "$MODEL" --start_run 161 --end_run 200 &

# Wait for all 5 background nodes to finish their chunks
wait

echo "All nodes finished calculating. Aggregating results..."

# Run the aggregation step on the primary node
python lou_simulation_hpc_mp_hd.py --scenario "$SCENARIO" --models "$MODEL" --aggregate_only 

echo "Job complete."