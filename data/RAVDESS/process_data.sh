#!/bin/bash

emotion_to_str(){
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    case $1 in
        "05") echo "anger";;
        "07") echo "disgust";;
        "06") echo "fear";;
        "03") echo "happiness";;
        "04") echo "sadness";;
        "08") echo "surprise";;
        *) echo "neutral";;
    esac
}

modality_to_str(){
    case $1 in
        "01") echo "SPEECH";;
        "02") echo "SONG";;
    esac
}

intensity_to_str(){
    case $1 in
        "01") echo "LO";;
        "02") echo "HI";;
    esac
}

DATASET_NAME="RAVDESS"
EXT="wav"

mkdir data
for FILE in ./speech/*.wav; do
    echo "Processing ${FILE}"
    
    # Extract the filename
    FILENAME=$(basename ${FILE} .$EXT)
    FILENAME=${FILENAME%.*}

    # Extract the informations
    ARR_INFO=(${FILENAME//-/ })
    
    # Extract the emotion
    # Actor
    MODALITY=${ARR_INFO[0]}
    VOCAL_CHANNEL=$(modality_to_str ${ARR_INFO[1]})
    EMOTION=$(emotion_to_str ${ARR_INFO[2]})
    INTENSITY=$(intensity_to_str ${ARR_INFO[3]})
    STATEMENT=${ARR_INFO[4]}
    REPETITION=${ARR_INFO[5]}
    ACTOR=${ARR_INFO[6]}

    # Copy the file
    mkdir -p ./data/${EMOTION}
    NEWFILE="./data/$EMOTION/${EMOTION}_${INTENSITY}_${VOCAL_CHANNEL}_${ACTOR}_${STATEMENT}_${REPETITION}_${DATASET_NAME}.${EXT}"
    echo  "Saving to ${NEWFILE}"
    # mv ${FILE} ${NEWFILE}
done

