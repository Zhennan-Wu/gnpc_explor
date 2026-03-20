#!/bin/bash
#SBATCH --job-name=lou_simulations_mp
#SBATCH -p general
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=4-00:00:00               
#SBATCH --nodes=4                       # NEW: Request 4 full nodes
#SBATCH --ntasks=4                      # NEW: 4 tasks total (1 per node)
#SBATCH --cpus-per-task=128             # KEEP: 128 cores per task/node
#SBATCH --mem=0                      
#SBATCH --array=1-64       
#SBATCH -A r00939           

# --- CONDA ACTIVATION ---
module load conda 
module load gcc

eval "$(conda shell.bash hook)"
conda activate ou
# ------------------------

export CXX=g++
export CC=gcc

CHAINS=3

CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)
SCENARIO=$(echo $CONFIG_LINE | awk '{print $1}')
MODEL=$(echo $CONFIG_LINE | awk '{print $2}')

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Scenario: $SCENARIO with Model: $MODEL using 4 Nodes"

# We split 200 simulations across 4 nodes (50 sims per node).
# The '&' puts the process in the background. '--exclusive' ensures they don't fight for the same resources.
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp.py --scenario "$SCENARIO" --model "$MODEL" --start_run 1 --end_run 50 &
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp.py --scenario "$SCENARIO" --model "$MODEL" --start_run 51 --end_run 100 &
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp.py --scenario "$SCENARIO" --model "$MODEL" --start_run 101 --end_run 150 &
srun -N 1 -n 1 --exclusive python lou_simulation_hpc_mp.py --scenario "$SCENARIO" --model "$MODEL" --start_run 151 --end_run 200 &

# Wait for all 4 background nodes to finish their chunks
wait

echo "All nodes finished calculating. Aggregating results..."

# Run the aggregation step on the primary node
python lou_simulation_hpc_mp.py --scenario "$SCENARIO" --model "$MODEL" --aggregate_only 

echo "Job complete."