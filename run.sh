#!/bin/bash
# Script to run the Document Intelligence Search System (DISS)

# Kill any existing uvicorn processes
pkill -f uvicorn || true

# Install dependencies if needed
pip install -r requirements.txt

# Ensure Ollama is installed and the model is available
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it from https://ollama.ai/"
    exit 1
fi

# Pull the deepseek model if not already available
ollama list | grep -q "deepseek-r1:7b" || ollama pull deepseek-r1:7b

# Create necessary directories
mkdir -p data/documents data/processed data/indexes

# Initialize documents if needed
if [ "$1" == "--init" ] || [ "$1" == "-i" ]; then
    echo "Initializing documents from PDFs folder..."
    python init_documents.py
fi

# Run the application
echo "Starting Document Intelligence Search System (DISS)..."
echo "The application will automatically process documents from the PDFs folder."
echo "Access the application at http://localhost:8000"
python run.py