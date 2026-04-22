#!/bin/bash
PYTHONPATH=. ./venv/bin/python3 server/app.py > validator_test.log 2>&1 &
PID=$!
sleep 5
echo "--- RESET TEST ---"
curl -s http://127.0.0.1:7860/reset -X POST -H "Content-Type: application/json" -d '{"task":"easy"}'
echo -e "\n--- STEP TEST ---"
curl -s http://127.0.0.1:7860/step -X POST -H "Content-Type: application/json" -d '{"actions":[0,1]}'
kill $PID
