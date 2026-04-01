#!/bin/bash

# 1. Submit the main job array and capture the output message
# (sbatch usually outputs: "Submitted batch job 123456")
ARRAY_OUTPUT=$(sbatch run_experiments_hpc_mp_hd.slurm)
echo "Main Array: $ARRAY_OUTPUT"

# 2. Extract just the Job ID number from that output string
JOB_ID=$(echo $ARRAY_OUTPUT | awk '{print $4}')