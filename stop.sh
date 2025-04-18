#!/bin/bash
# Script to stop the Document Intelligence Search System (DISS)

echo "Stopping Document Intelligence Search System (DISS)..."

# Check if the PID file exists
if [ -f .diss.pid ]; then
    PID=$(cat .diss.pid)

    # Check if the process is running
    if ps -p $PID > /dev/null; then
        # Kill the process
        kill $PID
        echo "Application stopped."
    else
        echo "Application is not running with PID $PID."
    fi

    # Remove the PID file
    rm .diss.pid
else
    # Try to find and kill the process by name
    PIDS=$(pgrep -f "python3 run.py")

    if [ -n "$PIDS" ]; then
        echo "Killing processes: $PIDS"
        kill $PIDS
        echo "Application stopped."
    else
        echo "Application is not running."
    fi
fi

# Kill any uvicorn processes
UVICORN_PIDS=$(pgrep -f "uvicorn")
if [ -n "$UVICORN_PIDS" ]; then
    echo "Killing uvicorn processes: $UVICORN_PIDS"
    kill $UVICORN_PIDS
fi

# Check if we should stop Ollama as well
if [ "$1" == "--all" ] || [ "$1" == "-a" ]; then
    echo "Stopping Ollama..."
    pkill -f "ollama"
fi

echo "Cleanup complete."
