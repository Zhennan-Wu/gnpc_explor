#!/bin/bash

while read -r SCENARIO MODEL; do
    echo "Aggregate Scenario: $SCENARIO with Model: $MODEL"

    python lou_simulation_hpc_mp_old.py \
        --scenario "$SCENARIO" \
        --model "$MODEL" \
        --aggregate_only

done < aggregate.txt