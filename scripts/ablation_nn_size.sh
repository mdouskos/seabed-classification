#!/bin/bash

EXPERIMENT="Size"

# REGIONS=(BB16.json BB17.json BB18.json PB.json LP.json)
REGIONS=(PB.json)
MODES=(bs)
MODELS=(nn)
NNSIZE=(64 128 256 512)

counter=1
mode=${MODES[0]}
model=${MODELS[0]}

for nnsize in "${NNSIZE[@]}"; do
    for region in "${REGIONS[@]}"; do
        echo "Experiment $((counter++))"
        if [ $mode = "pe" ]; then
            mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P mode="bs" -P pe_dim=6 -P model=$model -P nn_size=$nnsize
        else
            mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P model=$model -P mode=$mode -P nn_size=$nnsize
        fi
    done
done