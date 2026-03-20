#!/bin/bash

# 1. Submit the main job array and capture the output message
# (sbatch usually outputs: "Submitted batch job 123456")
ARRAY_OUTPUT=$(sbatch run_experiments_hpc_mp.slurm)
echo "Main Array: $ARRAY_OUTPUT"

# 2. Extract just the Job ID number from that output string
JOB_ID=$(echo $ARRAY_OUTPUT | awk '{print $4}')

# 3. Submit the aggregator script, telling it to wait for the Job ID
# 'afterok' means it will only run if the array finishes WITHOUT failing.
# (If you want it to run even if some tasks fail, use 'afterany' instead).
sbatch --dependency=afterok:$JOB_ID aggregate.slurm

echo "Aggregator queued. It will run automatically when Job $JOB_ID succeeds."