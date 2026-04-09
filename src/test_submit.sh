#!/bin/bash
#SBATCH --job-name=test_dahlou
#SBATCH -p general
#SBATCH --output=logs/test_als_%j.out
#SBATCH --error=logs/test_als_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=00:15:00               # ONLY 15 MINUTES for the test
#SBATCH --nodes=1                       # ONLY 1 NODE
#SBATCH --ntasks=1                      
#SBATCH --cpus-per-task=8               # Just enough cores for a couple of workers
#SBATCH --mem=16G                       # Modest memory request
#SBATCH -A r00939         

# --- CONDA ACTIVATION ---
module load conda 
module load gcc

eval "$(conda shell.bash hook)"
conda activate emlou
# ------------------------

export CXX=g++
export CC=gcc

# --- HARDCODE TEST VARIABLES ---
# Instead of reading from als_exp.txt via an array, we test one known combination
SCENARIO="S1"
MODEL="dahlou_ncp_full.stan" # Adjust if your txt file drops the .stan extension

echo "Starting HPC Smoke Test..."
echo "Running Scenario: $SCENARIO with Model: $MODEL on 1 Node"

# # 1. TEST COMPILATION
# echo "Step 1: Testing Stan Model Compilation..."
# python dahlou_hpc_mp.py --scenario "$SCENARIO" --models "$MODEL" --compile_only

# 2. TEST EXECUTION (The "Micro-Run")
# We use only 2 runs, 50 patients, and 50 iterations so it finishes in a minute or two.
# Note: I am assuming your Python script accepts --models (with an 's') based on your earlier code. 
# If you changed it to --model in Python, update the flag below!
echo "Step 2: Running 2 micro-iterations to test data loading, HMC sampling, and ArviZ..."
srun python dahlou_hpc_mp.py \
    --scenario "$SCENARIO" \
    --models "$MODEL" \
    --start_run 1 \
    --end_run 2 \
    --data_size 50 \
    --warmup 50 \
    --sampling 50 \
    --chains 1 

# 3. TEST AGGREGATION
echo "Step 3: Testing Pandas Aggregation..."
python dahlou_hpc_mp.py --scenario "$SCENARIO" --models "$MODEL" --aggregate_only 

echo "Smoke test complete! Check logs/test_als_${SLURM_JOB_ID}.out for success messages."