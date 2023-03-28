#!/bin/bash

# Ports
STARTING_PORT=4446
PORT=$STARTING_PORT

# Blocks
# BLOCKS=( "chroma" "zcr-rms")
BLOCKS=("chroma")
BLOCK_POSTFIX="-processing-block-py"

# Number of servers
NUM_SERVERS=${#BLOCKS[@]}

# Start forwarding
for BLOCK in "${BLOCKS[@]}"
do
    # Start server
    SERVER_PATH="${BLOCK}${BLOCK_POSTFIX}"
    cd $SERVER_PATH
    PORT=$PORT python dsp-server.py &
    cd ..
done
