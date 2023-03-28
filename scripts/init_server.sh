#!/bin/bash

# Ports
PORT=${2:-4446}

# Block Name
BLOCK_NAME=${1:-"non-mfcc"}
BLOCK_POSTFIX="-processing-block-py"

BLOCK="${BLOCK_NAME}${BLOCK_POSTFIX}"

# Kill any old processes
fuser -k $PORT/tcp
killall ngrok

if [ -f server.pid ]; then
    kill $(cat server.pid)
fi

# Start server
cd $BLOCK
PORT=$PORT python dsp-server.py &
PID=$!
cd ..

# Save PID to file to kill later
echo $PID > server.pid

# Start ngrok
ngrok http $PORT