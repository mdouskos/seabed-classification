#!/bin/bash

EXPERIMENT="Split"

REGIONS=(BB16.json BB17.json BB18.json PB.json LP.json)
# REGIONS=(PB.json)
MODES=(bs)
MODELS=(nn)
DATASET="./Datasets/paper/aligned"

counter=1
mode=${MODES[0]}
model=${MODELS[0]}

for region in "${REGIONS[@]}"; do
    echo "Region $region"
    python seabed_classification.py --dataset-dir $DATASET --inputs $region --runs 1 --taxonomy "folk 5" --model $model  \
                --normalize std --dilation-radius 16 --epochs 6 --batch 256000 --xls-file cm_${region::-5} --class-map
done