#!/bin/bash
#SBATCH --job-name=DFOULS_Exp
#SBATCH --output=logs/dfou_%a.out
#SBATCH --error=logs/dfou_%a.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=4-00:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10             
#SBATCH --mem=128G                      
#SBATCH --array=1-4       
#SBATCH -A r00939   

# --- CONDA ACTIVATION ---
module load conda 
module load gcc

# This hooks Conda into the non-interactive Slurm shell
eval "$(conda shell.bash hook)"
conda activate dfou
# ------------------------

## 2. Define Experiment Files
FILES=(
    "dfou_id_multi.py"
    "dfou_id_fit.py"
    "dfou_het_multi.py"
    "dfou_het_fit.py"
)

# Get the specific file for this array task
CURRENT_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

## 3. Execute
echo "Starting Experiment: $CURRENT_FILE"
echo "Running on Host: $HOSTNAME"

python "$CURRENT_FILE"

echo "Finished Experiment: $CURRENT_FILE"