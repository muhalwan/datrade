#!/bin/bash

# Stop the worker
heroku ps:scale worker=0 --app YOUR_APP_NAME

# Wait for 30 seconds
sleep 30

# Start the worker
heroku ps:scale worker=1 --app YOUR_APP_NAME