#!/bin/bash

DATA_PATH=${1:-"CREMAD"}
EXT=${2:-"wav"}

PROCESSED_PATH="data/processed"

echo "Processing data from $DATA_PATH"

mkdir -p ${PROCESSED_PATH}

for FILE in ./data/${DATA_PATH}/*.$EXT; do
    echo "Processing ${FILE}"
    
    # Extract the filename
    FILENAME=$(basename ${FILE} .$EXT)
    FILENAME=${FILENAME%.*}

    # Extract the informations
    ARR_INFO=(${FILENAME//_/ })
    
    # Extract the emotion
    EMOTION=${ARR_INFO[2]}
    EXTENT=${ARR_INFO[3]}

    # Reorganize the data
    mkdir -p ${PROCESSED_PATH}/${EMOTION}/${EXTENT}

    # Copy the file
    cp ${FILE} ${PROCESSED_PATH}/${EMOTION}/${EXTENT}/${FILENAME}.$EXT
done
