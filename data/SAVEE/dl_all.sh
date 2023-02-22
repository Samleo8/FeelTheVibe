#!/bin/bash

ACTORS="DC JE JK KL"
PID_LIST=""
for ACTOR in $ACTORS; do
    echo "Downloading $ACTOR"
    ./dl_actor.sh $ACTOR &
    
    PID_LIST="$PID_LIST $!"
done

for PID in $PID_LIST; do
    wait $PID
done

echo "Done!"