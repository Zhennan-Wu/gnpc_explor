#!/bin/bash
#SBATCH --job-name=DFOULS_Exp_CPU
#SBATCH --output=logs/dfou_%a.out
#SBATCH --error=logs/dfou_%a.err
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zwu1@iu.edu
#SBATCH --time=4-00:00:00               # Dropped to 2 days (GPU will make this fast)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6               # Dropped to 4 cores
#SBATCH --mem=64G                       # Dropped to 32GB RAM for faster queue times
#SBATCH --array=1-4       
#SBATCH -A r00939

# --- CONDA ACTIVATION ---
module load conda 
module load gcc
module load cuda

# This hooks Conda into the non-interactive Slurm shell
eval "$(conda shell.bash hook)"
conda activate dfou
# ------------------------

# Move to submission directory
cd $SLURM_SUBMIT_DIR

FILES=(
    "dfou_id_multi.py"
    "dfou_id_fit.py"
    "dfou_het_multi.py"
    "dfou_het_fit.py"
)

CURRENT_FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}

echo "Starting Experiment: $CURRENT_FILE"
echo "Running on Host: $HOSTNAME"
echo "Working directory: $(pwd)"

python "$CURRENT_FILE"

echo "Finished Experiment: $CURRENT_FILE"