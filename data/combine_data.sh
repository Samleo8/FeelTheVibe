#!/bin/bash

DATASETS=("CREMAD" "SAVEE" "TESS")
EMOTIONS=("anger" "disgust" "fear" "happiness" "neutral" "sadness" "surprise")

# Create the data folder
mkdir -p combined

for EMOTION in "${EMOTIONS[@]}"; do
    # Create the emotion folder
    mkdir -p combined/${EMOTION}

    for DATASET in "${DATASETS[@]}"; do
        # Symlink the files
        FILES=$(ls ./${DATASET}/data/${EMOTION})
        for FILE in ${FILES}; do
            ln -s ${DATASET}/data/${EMOTION}/${FILE} combined/${EMOTION}/
        done
    done
done
