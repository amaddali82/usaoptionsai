#!/bin/bash
# Stop USA Options AI System

echo "Stopping USA Options AI System..."

# Stop Python processes
if [ -f .pids ]; then
    echo "Stopping background processes..."
    kill $(cat .pids) 2>/dev/null
    rm .pids
    echo "  Background processes stopped"
fi

# Stop Docker services
echo "Stopping Docker services..."
docker-compose down

echo "System stopped successfully"
