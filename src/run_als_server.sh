#!/bin/bash

# --- VENV ACTIVATION ---
# Activating the virtual environment created earlier
# source /home/dzhao268/envs/emlou/bin/activate

# module load conda 
# module load gcc

eval "$(conda shell.bash hook)"
conda activate ou

# Ensure Stan uses the correct C++ compilers
# export CXX=g++
# export CC=gcc

CONFIG_FILE="als_exp.txt"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi

echo "Starting empirical runs on standalone server..."
echo "==============================================="

echo "Compiling all Stan models before running empirical data..."
python compile_stan_models.py

# Read the configuration file line by line
while read -r SCENARIO MODEL || [ -n "$SCENARIO" ]; do
    # Skip empty lines
    if [[ -z "$SCENARIO" ]]; then
        continue
    fi

    echo "-----------------------------------------------"
    echo "Running Empirical Data: Scenario $SCENARIO with Model $MODEL"
    
    # # Pre-compile the model sequentially
    # python dahlou_server.py --scenario "$SCENARIO" --models "$MODEL" --compile_only
    
    # Run the single empirical execution (4 chains, 2500/2500 iterations)
    python dahlou_server.py --scenario "$SCENARIO" --models "$MODEL" --chains 4 --warmup 2500 --sampling 2500 > "log_${SCENARIO}_${MODEL}.txt" 2>&1 &
    
    echo "Finished Scenario $SCENARIO with Model $MODEL"
    
done < "$CONFIG_FILE"

# STEP 3: Wait for everything to finish
echo "All jobs dispatched to the background. Waiting for them to complete..."

# The 'wait' command pauses this script until all background '&' jobs are done.
wait

echo "======================================================"
echo "✅ All parallel server jobs are successfully complete!"