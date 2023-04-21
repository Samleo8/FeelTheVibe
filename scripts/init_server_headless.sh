#!/bin/bash

# Kill old processses
kill_old_procs(){
    if [ -f server.pid ]; then
        kill $(cat server.pid)
    fi
}

# Ports
PORT=${2:-4446}

# Block Name
BLOCK_NAME=${1:-"lpcc"}
BLOCK_POSTFIX="-processing-block-py"

BLOCK="${BLOCK_NAME}${BLOCK_POSTFIX}"

# Kill any old processes
git pull
kill_old_procs

# Start server
cd $BLOCK
PORT=$PORT python3 dsp-server.py &> dsp_server.log &
PID=$!
cd ..

echo "Server started at localhost:$PORT -> https://vibe.strasserver.com"

# Save PID to file to kill later
echo $PID > server.pid

# Tail the log file
tail -F $BLOCK/dsp_server.log