#!/bin/bash

# Ports
STARTING_PORT=4446
PORT=$STARTING_PORT

# Blocks
BLOCKS=( "chroma" "zcr-rms")
BLOCK_POSTFIX="-processing-block-py"

# Number of servers
NUM_SERVERS=${#BLOCKS[@]}

# Start forwarding
for BLOCK in "${BLOCKS[@]}"
do
    # Start server
    SERVER_FILE="${BLOCK}${BLOCK_POSTFIX}/dsp_server.py"
    PORT=$PORT python $SERVER_FILE &
done
