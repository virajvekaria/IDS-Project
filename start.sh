#!/bin/bash
# Script to start the Document Intelligence Search System (DISS)

# Check if the application is already running
if pgrep -f "python3 run.py" > /dev/null; then
    echo "The application is already running."
    echo "To stop it, run ./stop.sh"
    exit 1
fi

# Create necessary directories
mkdir -p data/documents data/processed data/indexes

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Check if the llama3 model is available
if ! ollama list | grep -q "llama3"; then
    echo "The llama3 model is not available. Please run 'ollama pull llama3' first."
    exit 1
fi

# Initialize documents if needed
if [ "$1" == "--init" ] || [ "$1" == "-i" ]; then
    echo "Initializing documents from PDFs folder..."
    python3 init_documents.py
fi

# Build React frontend if needed
if [ ! -d "app/frontend/react/dist" ] || [ "$1" == "--build-frontend" ] || [ "$1" == "-b" ]; then
    echo "Building React frontend..."
    ./build_react.sh
fi

# Start the application
echo "Starting Document Intelligence Search System (DISS)..."
echo "The application will automatically process documents from the PDFs folder."
echo "Access the application at http://localhost:8000"
echo "To stop the application, run ./stop.sh"

# Run in the background and save the PID
nohup python3 run.py > diss.log 2>&1 &
echo $! > .diss.pid

# Wait a moment for the application to start
sleep 3

# Check if the application is running
if pgrep -f "python3 run.py" > /dev/null; then
    echo "Application started successfully!"
    echo "Opening browser..."
    python3 -m webbrowser http://localhost:8000
else
    echo "Failed to start the application. Check diss.log for details."
    exit 1
fi
