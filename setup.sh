#!/bin/bash
# Setup script for the Document Intelligence Search System (DISS)

echo "Setting up Document Intelligence Search System (DISS)..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo "Python 3.10 or higher is required. You have Python $python_version."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it from https://ollama.ai/"
    echo "After installing Ollama, run this script again."
    exit 1
fi

# Pull the deepseek model
echo "Pulling the deepseek-r1:7b model (this may take a while)..."
ollama pull deepseek-r1:7b

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/documents data/processed data/indexes

# Make scripts executable
chmod +x start.sh stop.sh

echo "Setup complete! You can now run the application with ./start.sh"
