#!/bin/bash
# Stop MLflow tracking server
#
# Usage:
#   ./scripts/stop_mlflow.sh

MLFLOW_ROOT="${MLFLOW_ROOT:-$HOME/mlflow}"
PID_FILE="$MLFLOW_ROOT/mlflow.pid"

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping MLflow server (PID: $PID)..."
        kill "$PID"
        rm "$PID_FILE"
        echo "MLflow server stopped."
    else
        echo "MLflow server is not running (stale PID file)."
        rm "$PID_FILE"
    fi
else
    echo "No PID file found. MLflow server may not be running."
    echo "Attempting to find and kill mlflow server process..."
    pkill -f "mlflow server" && echo "MLflow server stopped." || echo "No mlflow server process found."
fi
