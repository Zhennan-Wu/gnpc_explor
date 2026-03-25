#!/bin/bash

# Declare an associative array to group models by scenario
declare -A scenario_models

# 1. Read the text file and group the models
while read -r SCENARIO MODEL; do
    # Skip empty lines to prevent errors
    [[ -z "$SCENARIO" ]] && continue
    
    # Append the model to the scenario's list, separated by a space
    if [[ -z "${scenario_models[$SCENARIO]}" ]]; then
        scenario_models[$SCENARIO]="$MODEL"
    else
        scenario_models[$SCENARIO]+=" $MODEL"
    fi
done < aggregate.txt

# 2. Loop through the grouped scenarios and execute the Python script
for SCENARIO in "${!scenario_models[@]}"; do
    MODELS="${scenario_models[$SCENARIO]}"
    
    echo "========================================"
    echo "Cross-Aggregating Scenario: $SCENARIO"
    echo "Models: $MODELS"
    echo "========================================"

    python lou_simulation_hpc_mp_old.py \
        --scenario "$SCENARIO" \
        --models $MODELS \
        --cross_aggregate

done