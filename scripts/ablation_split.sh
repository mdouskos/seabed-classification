#!/bin/bash

EXPERIMENT="Split"

REGIONS=(BB16.json BB17.json BB18.json PB.json LP.json)
# REGIONS=(PB.json)
MODES=(bs)
MODELS=(nn)
SPLITS=(.3 .5 .7)

counter=1
mode=${MODES[0]}
model=${MODELS[0]}

for split in "${SPLITS[@]}"; do
    for region in "${REGIONS[@]}"; do
        echo "Experiment $((counter++))"
        if [ $mode = "pe" ]; then
            mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P mode="bs" -P pe_dim=6 -P model=$model -P split=$split
        else
            mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P model=$model -P mode=$mode -P split=$split
        fi
    done
done