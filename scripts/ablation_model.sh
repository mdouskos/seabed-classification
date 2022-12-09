#!/bin/bash

EXPERIMENT="Ablations"

REGIONS=(BB16.json BB17.json BB18.json PB.json LP.json)
# REGIONS=(PB.json)
MODES=(pe)
MODELS=(rf nn svm)
SIGMA=(5e-6 5e-5 5e-4 5e-3 5e-2)
counter=1

for model in "${MODELS[@]}"; do
    for region in "${REGIONS[@]}"; do
            for mode in "${MODES[@]}"; do
            echo "Experiment $((counter++))"
            if [ $mode = "pe" ]; then
                for sigma in "${SIGMA[@]}"; do
                    mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P mode="bs" -P pe_dim=6 -P pe_sigma=$sigma -P model=$model 
                done
            else
                mlflow run . --no-conda --experiment-name $EXPERIMENT -P region=$region -P model=$model -P mode=$mode
            fi
        done
    done
done