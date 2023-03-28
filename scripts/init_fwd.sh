#!/bin/bash

# Ports
STARTING_PORT=4446

# Number of servers
NUM_SERVERS=1

# Start forwarding
for i in $(seq 1 $NUM_SERVERS)
do
    PORT=$(($STARTING_PORT + $i - 1))
    ngrok http $PORT
done
