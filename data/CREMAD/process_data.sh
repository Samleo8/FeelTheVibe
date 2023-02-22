#!/bin/bash

emotion_to_str(){
    case $1 in
        "ANG") echo "anger";;
        "DIS") echo "disgust";;
        "FEA") echo "fear";;
        "HAP") echo "happiness";;
        "SAD") echo "sadness";;
        *) echo "neutral";;
    esac
}

DATASET_NAME="CREMAD"
EXT="wav"

mkdir data
for FILE in ./*.wav; do
    echo "Processing ${FILE}"
    
    # Extract the filename
    FILENAME=$(basename ${FILE} .$EXT)
    FILENAME=${FILENAME%.*}

    # Extract the informations
    ARR_INFO=(${FILENAME//_/ })
    
    # Extract the emotion
    ID=${ARR_INFO[0]}
    ACTOR=${ARR_INFO[1]}
    EMOTION=$(emotion_to_str ${ARR_INFO[2]})
    EXTENT=${ARR_INFO[3]}

    # Copy the file
    mkdir -p ./data/${EMOTION}
    echo  "./data/$EMOTION/${EMOTION}_${ACTOR}_${ID}_${DATASET_NAME}.wav"
    # mv ${FILE} ./data/${EMOTION}/${FILENAME}.$EXT
done

