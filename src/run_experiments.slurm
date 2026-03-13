#!/bin/bash
#SBATCH --job-name=lou_simulations
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --time=04:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3             # <-- Hardcoded here! Matches your chains
#SBATCH --mem=8G                      
#SBATCH --array=1-10                  

# Slurm automatically populates $SLURM_CPUS_PER_TASK based on the header above.
# We include a fallback of 3 just in case you ever run this script locally outside of Slurm.
CHAINS=${SLURM_CPUS_PER_TASK:-3}

# Extract the specific scenario and model for this task ID from the text file
CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)
SCENARIO=$(echo $CONFIG_LINE | awk '{print $1}')
MODEL=$(echo $CONFIG_LINE | awk '{print $2}')

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Scenario: $SCENARIO with Model: $MODEL using $CHAINS CPUs/Chains"

# Execute the Python script with the extracted arguments
python your_script_name.py --scenario "$SCENARIO" --model "$MODEL" --chains "$CHAINS"