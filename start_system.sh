#!/bin/bash
# USA Options AI System - Startup Script (Linux/Mac)

set -e

echo "========================================"
echo "USA Options AI System Startup"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Docker is running
echo -e "${YELLOW}[1/6] Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running. Please start Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}  Docker is running${NC}"

# Start Docker Compose services
echo -e "\n${YELLOW}[2/6] Starting Docker services...${NC}"
echo -e "${NC}  This may take a few minutes on first run...${NC}"

docker-compose up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to start Docker services${NC}"
    exit 1
fi

echo -e "${GREEN}  Docker services started successfully${NC}"

# Wait for services to be ready
echo -e "\n  Waiting for services to be ready..."
sleep 30

# Check service health
echo -e "\n  Checking service health:"

services=("Kafka:9092" "InfluxDB:8086" "TimescaleDB:5432" "Redis:6379" "Grafana:3000")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if nc -z localhost $port 2>/dev/null; then
        echo -e "    ${GREEN}✓ $name (port $port)${NC}"
    else
        echo -e "    ${YELLOW}✗ $name (port $port) - NOT READY${NC}"
    fi
done

# Initialize Kafka topics
echo -e "\n${YELLOW}[3/6] Initializing Kafka topics...${NC}"
python3 scripts/init_kafka_topics.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}  Kafka topics initialized${NC}"
else
    echo -e "${YELLOW}  WARNING: Failed to initialize Kafka topics${NC}"
fi

# Start data ingestion
echo -e "\n${YELLOW}[4/6] Starting data ingestion service...${NC}"
python3 data_ingestion/main.py > logs/ingestion.log 2>&1 &
INGESTION_PID=$!
echo -e "${GREEN}  Data ingestion service started (PID: $INGESTION_PID)${NC}"

# Start stream processing
echo -e "\n${YELLOW}[5/6] Starting stream processing jobs...${NC}"

# Submit Spark jobs
echo -e "  Submitting feature extraction job..."
spark-submit \
    --master spark://localhost:7077 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
    stream_processing/spark_jobs/feature_extraction.py > logs/spark_features.log 2>&1 &
FEATURE_PID=$!

sleep 5

echo -e "  Submitting Greeks calculator job..."
spark-submit \
    --master spark://localhost:7077 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
    stream_processing/spark_jobs/greeks_calculator.py > logs/spark_greeks.log 2>&1 &
GREEKS_PID=$!

echo -e "${GREEN}  Stream processing jobs started${NC}"
echo -e "    Feature extraction (PID: $FEATURE_PID)"
echo -e "    Greeks calculator (PID: $GREEKS_PID)"

# Start prediction service
echo -e "\n${YELLOW}[6/6] Starting ML prediction service...${NC}"
python3 models/realtime_prediction.py > logs/predictions.log 2>&1 &
PREDICTION_PID=$!
echo -e "${GREEN}  Prediction service started (PID: $PREDICTION_PID)${NC}"

# Start dashboard
echo -e "\n${YELLOW}[OPTIONAL] Starting Dash visualization dashboard...${NC}"
python3 visualization/dash_app.py > logs/dashboard.log 2>&1 &
DASH_PID=$!
echo -e "${GREEN}  Dash dashboard started (PID: $DASH_PID)${NC}"
echo -e "${CYAN}  Access at: http://localhost:8050${NC}"

# Save PIDs
echo "$INGESTION_PID" > .pids
echo "$FEATURE_PID" >> .pids
echo "$GREEKS_PID" >> .pids
echo "$PREDICTION_PID" >> .pids
echo "$DASH_PID" >> .pids

# Summary
echo -e "\n========================================"
echo -e "${CYAN}System Startup Complete!${NC}"
echo -e "========================================\n"

echo -e "${YELLOW}Service URLs:${NC}"
echo -e "  Grafana:        http://localhost:3000 (admin/admin)"
echo -e "  Dash Dashboard: http://localhost:8050"
echo -e "  Jupyter:        http://localhost:8888"
echo -e "  Spark UI:       http://localhost:8080"
echo -e "  InfluxDB:       http://localhost:8086"

echo -e "\n${YELLOW}Running Processes:${NC}"
ps -p $INGESTION_PID,$FEATURE_PID,$GREEKS_PID,$PREDICTION_PID,$DASH_PID -o pid,comm,stat 2>/dev/null || echo "  Process information not available"

echo -e "\n${YELLOW}Useful Commands:${NC}"
echo -e "  View logs:       tail -f logs/<service>.log"
echo -e "  Stop all:        ./stop_system.sh"
echo -e "  Docker logs:     docker-compose logs -f <service-name>"

echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}Shutting down...${NC}"; kill $(cat .pids 2>/dev/null) 2>/dev/null; rm .pids 2>/dev/null; echo -e "${GREEN}Cleanup complete${NC}"; exit 0' INT

# Keep script running
tail -f /dev/null
