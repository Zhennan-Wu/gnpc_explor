#!/bin/bash
#SBATCH --job-name=lou_jls_simulations
#SBATCH -p general
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=4-00:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128             
#SBATCH --mem=0                      
#SBATCH --array=1-64       
#SBATCH -A r00939           

# --- CONDA ACTIVATION ---
module load conda 
module load gcc

# This hooks Conda into the non-interactive Slurm shell
eval "$(conda shell.bash hook)"
conda activate ou
# ------------------------

export CXX=g++
export CC=gcc

# 3. Force a clean reinstall of CmdStan
# install_cmdstan --overwrite

# Slurm automatically populates $SLURM_CPUS_PER_TASK based on the header above.
# We include a fallback of 3 just in case you ever run this script locally outside of Slurm.
CHAINS=3

# Extract the specific scenario and model for this task ID from the text file
CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments.txt)
SCENARIO=$(echo $CONFIG_LINE | awk '{print $1}')
MODEL=$(echo $CONFIG_LINE | awk '{print $2}')

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Scenario: $SCENARIO with Model: $MODEL using $CHAINS CPUs/Chains"
echo "Job started on $(date)"
echo "Allocated cores: $SLURM_CPUS_PER_TASK"

# Execute the Python script with the extracted arguments
python lou_simulation_hpc.py --scenario "$SCENARIO" --model "$MODEL" --sims 200 --chains "$CHAINS"
