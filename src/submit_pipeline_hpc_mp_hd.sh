#!/bin/bash

# 0. Compile the Stan models first to avoid multiple nodes trying to compile at the same time (which can cause file access conflicts).
module load conda 
module load gcc

eval "$(conda shell.bash hook)"
conda activate ou
# ------------------------

export CXX=g++
export CC=gcc

echo "Pre-compiling Stan Model..."
python compile_stan_models.py
echo "Compilation step done. Starting parallel instances..."

# 1. Submit the main job array and capture the output message
# (sbatch usually outputs: "Submitted batch job 123456")
ARRAY_OUTPUT=$(sbatch run_experiments_hpc_mp_hd.slurm)
echo "Main Array: $ARRAY_OUTPUT"

# 2. Extract just the Job ID number from that output string
JOB_ID=$(echo $ARRAY_OUTPUT | awk '{print $4}')