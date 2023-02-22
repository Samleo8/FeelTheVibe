#!/bin/bash

ACTORS="DC JE JK KL"

for ACTOR in $ACTORS; do
    echo "Downloading $ACTOR"
    ./dl_actor.sh $ACTOR &

    echo -e "Done\n"
done
