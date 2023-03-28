#!/bin/bash

DATASETS=("CREMAD" "SAVEE" "TESS" "RAVDESS")
EMOTIONS=("anger" "disgust" "fear" "happiness" "neutral" "sadness" "surprise")

# Create the data folder
mkdir -p combined

for EMOTION in "${EMOTIONS[@]}"; do
    # Create the emotion folder
    mkdir -p combined/${EMOTION}

    for DATASET in "${DATASETS[@]}"; do
        cp -r ${DATASET}/data/${EMOTION} combined/

        # Symlink the files
        # FILES=$(ls ./${DATASET}/data/${EMOTION})
        # for FILE in ${FILES}; do
        #     cp ${DATASET}/data/${EMOTION}/${FILE} combined/${EMOTION}/
        # done
    done
done
