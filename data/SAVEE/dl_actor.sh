#!/bin/bash

emotion_to_str(){
    case $1 in
        "a") echo "anger";;
        "d") echo "disgust";;
        "f") echo "fear";;
        "h") echo "happiness";;
        "sa") echo "sadness";;
        "su") echo "surprise";;
        *) echo "neutral";;
    esac
}

USER="guest2savee"
PASS="welcome!"

ACTOR=${1:-"DC"}

BASE_URL="http://kahlan.eps.surrey.ac.uk/savee/Data/AudioData/$ACTOR/"

# Get the list of files
FILES=$(curl -s -u $USER:$PASS $BASE_URL | grep -o -E "href=\"[^\"]+\"*.wav" | cut -d '"' -f 2)

# Create the directory
mkdir -p $ACTOR

# Download the files
for FILE in $FILES; do
    echo "Downloading $FILE"
    FILENAME=$(basename $FILE)
    FILENAME="${FILENAME%.*}"

    # Extract the emotion
    EMOTE_PREFIX=${FILENAME%??}
    EMOTION=`emotion_to_str ${EMOTE_PREFIX}`
    ID=${FILENAME#"$EMOTE_PREFIX"}

    echo "Emotion: $EMOTION | ID: $ID"

    wget --user $USER --password $PASS $ACTOR ${BASE_URL}${FILE} -O $ACTOR/SAVEE_${ACTOR}_${EMOTION}_${ID}.wav

    echo -e "Done\n"
done