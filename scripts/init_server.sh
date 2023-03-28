#!/bin/bash

# Kill old processses
kill_old_procs(){
    killall ngrok
    fuser -k $PORT/tcp

    if [ -f server.pid ]; then
        kill $(cat server.pid)
    fi
}

# Ports
PORT=${2:-4446}

# Block Name
BLOCK_NAME=${1:-"non-mfcc"}
BLOCK_POSTFIX="-processing-block-py"

BLOCK="${BLOCK_NAME}${BLOCK_POSTFIX}"

# Kill any old processes
kill_old_procs

# Start server
cd $BLOCK
PORT=$PORT python dsp-server.py &> dsp_server.log &
PID=$!
cd ..

# Save PID to file to kill later
echo $PID > server.pid

# Start ngrok
ngrok http $PORT

# Kill old processes
trap kill_old_procs SIGINT