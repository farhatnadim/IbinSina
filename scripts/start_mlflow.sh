#!/bin/bash
# Start MLflow tracking server
#
# Usage:
#   ./scripts/start_mlflow.sh                    # Foreground mode
#   ./scripts/start_mlflow.sh --background       # Background mode with nohup
#
# Environment:
#   MLFLOW_ROOT: Directory for MLflow data (default: ~/mlflow)
#
# The server will be available at http://$(hostname):5000
# Set MLFLOW_TRACKING_URI="http://$(hostname):5000" in your environment

set -e

MLFLOW_ROOT="${MLFLOW_ROOT:-$HOME/mlflow}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"

# Create directory structure
mkdir -p "$MLFLOW_ROOT"/{mlruns,mlartifacts}

echo "MLflow Configuration:"
echo "  Root:      $MLFLOW_ROOT"
echo "  Host:      $MLFLOW_HOST"
echo "  Port:      $MLFLOW_PORT"
echo "  DB:        $MLFLOW_ROOT/mlflow.db"
echo "  Artifacts: $MLFLOW_ROOT/mlartifacts"
echo ""

if [[ "$1" == "--background" ]]; then
    echo "Starting MLflow server in background..."
    nohup mlflow server \
        --backend-store-uri "sqlite:///$MLFLOW_ROOT/mlflow.db" \
        --default-artifact-root "$MLFLOW_ROOT/mlartifacts" \
        --host "$MLFLOW_HOST" \
        --port "$MLFLOW_PORT" \
        > "$MLFLOW_ROOT/mlflow.log" 2>&1 &

    echo $! > "$MLFLOW_ROOT/mlflow.pid"
    echo "MLflow server started (PID: $(cat $MLFLOW_ROOT/mlflow.pid))"
    echo "Log file: $MLFLOW_ROOT/mlflow.log"
    echo ""
    echo "To stop: kill \$(cat $MLFLOW_ROOT/mlflow.pid)"
else
    echo "Starting MLflow server in foreground (Ctrl+C to stop)..."
    echo ""
    mlflow server \
        --backend-store-uri "sqlite:///$MLFLOW_ROOT/mlflow.db" \
        --default-artifact-root "$MLFLOW_ROOT/mlartifacts" \
        --host "$MLFLOW_HOST" \
        --port "$MLFLOW_PORT"
fi
