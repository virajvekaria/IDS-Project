#!/bin/bash

# Stop the application if it's running
if [ -f .diss.pid ]; then
    echo "Stopping the application..."
    ./stop.sh
    sleep 2
fi

echo "Clearing all data for a fresh start..."

# Remove database file
echo "Removing database..."
rm -f data/diss.db

# Clear processed documents
echo "Clearing processed documents..."
rm -rf data/processed/*
touch data/processed/.gitkeep

# Clear document indexes
echo "Clearing document indexes..."
rm -rf data/indexes/*
touch data/indexes/.gitkeep

# Clear uploaded documents (but keep the sample document)
echo "Clearing uploaded documents..."
find data/documents -type f -not -name "attentionisallyouneed.pdf" -not -name ".gitkeep" -delete

# Remove log files
echo "Removing log files..."
rm -f diss.log document_processing.log

# Reset conversations
echo "Resetting conversations..."
# This will be done automatically when the database is recreated

echo "All data has been cleared. You can now start the application with a fresh state."
echo "Run ./start.sh to start the application."
