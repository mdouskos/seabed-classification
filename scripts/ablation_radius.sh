#!/bin/bash

EXPERIMENT="Radius"

REGIONS=(BB16.json BB17.json BB18.json PB.json LP.json)
# REGIONS=(PB.json)
MODES=(bs)
MODELS=(nn)
RADIUS=(8 12 14 16 20 30)

counter=1
mode=${MODES[0]}
model=${MODELS[0]}

for radius in "${RADIUS[@]}"; do
    for region in "${REGIONS[@]}"; do
        echo "Experiment $((counter++))"
        if [ $mode = "pe" ]; then
            mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P mode="bs" -P pe_dim=6 -P model=$model -P dil_radius=$radius
        else
            mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P model=$model -P mode=$mode -P dil_radius=$radius
        fi
    done
done