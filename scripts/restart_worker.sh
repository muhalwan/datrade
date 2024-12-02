#!/bin/bash

# Stop the worker
heroku ps:scale worker=0 --app crypto-collector

# Wait for 30 seconds
sleep 30

# Start the worker
heroku ps:scale worker=1 --app crypto-collector