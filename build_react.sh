#!/bin/bash

# Build React frontend for DISS

echo "Building React frontend..."

# Navigate to the React directory
cd app/frontend/react

# Install dependencies
echo "Installing dependencies..."
npm install

# Build the application
echo "Building application..."
npm run build

echo "React frontend built successfully!"
