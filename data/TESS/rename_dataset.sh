#!/bin/bash

emotion_to_str(){
    case $1 in
        "angry") echo "anger";;
        "disgust") echo "disgust";;
        "fear") echo "fear";;
        "happy") echo "happiness";;
        "sad") echo "sadness";;
        "ps") echo "surprise";;
        *) echo "neutral";;
    esac
}

DATASET_NAME="TESS"
EXT="wav"

FOLDERS=$(ls -d data/*)

for FOLDER in $FOLDERS; do
    cd $FOLDER
    mmv '*_CREMAD.wav' '#1_TESS.wav'
    cd ../..
done
