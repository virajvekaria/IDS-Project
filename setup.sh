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

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. It's required for the React frontend."
    echo "Please install Node.js 16 or higher from https://nodejs.org/"
    echo "After installing Node.js, run this script again."
    exit 1
fi

# Check Node.js version
node_version=$(node --version | cut -c 2-)
node_major=$(echo $node_version | cut -d. -f1)

if [ "$node_major" -lt 16 ]; then
    echo "Node.js 16 or higher is required. You have Node.js $node_version."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. It's required for the React frontend."
    echo "Please install npm along with Node.js from https://nodejs.org/"
    echo "After installing npm, run this script again."
    exit 1
fi

# Make scripts executable
chmod +x start.sh stop.sh build_react.sh

# Build the React frontend
echo "Building the React frontend..."
./build_react.sh

echo "Setup complete! You can now run the application with ./start.sh"
