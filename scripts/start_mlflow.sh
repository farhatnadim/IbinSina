#!/bin/bash
# Start MLflow tracking server
#
# Usage:
#   ./scripts/start_mlflow.sh                    # Foreground mode
#   ./scripts/start_mlflow.sh --background       # Background mode with nohup
#
# Environment variables:
#   MLFLOW_ROOT:  Directory for MLflow data (default: ~/mlflow)
#   MLFLOW_PORT:  Server port (default: 5000)
#   MLFLOW_HOST:  Server host (default: 0.0.0.0)
#   VENV_PATH:    Path to virtual environment (default: .venv in project root)
#
# Examples:
#   # Use default .venv
#   ./scripts/start_mlflow.sh --background
#
#   # Use custom venv location
#   VENV_PATH=/path/to/my/venv ./scripts/start_mlflow.sh --background
#
#   # Use globally installed mlflow (no venv)
#   VENV_PATH="" ./scripts/start_mlflow.sh --background
#
# The server will be available at http://$(hostname):5000
# Set MLFLOW_TRACKING_URI="http://$(hostname):5000" in your environment

set -e

# Find project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Find mlflow executable with the following priority:
#   1. VENV_PATH environment variable (if set)
#   2. .venv in project root (default convention)
#   3. Currently active virtualenv ($VIRTUAL_ENV)
#   4. System mlflow in PATH
find_mlflow() {
    # User-specified venv path
    if [[ -n "${VENV_PATH+x}" ]]; then
        if [[ -z "$VENV_PATH" ]]; then
            # Empty VENV_PATH means use system mlflow
            if command -v mlflow &> /dev/null; then
                echo "mlflow"
                return 0
            fi
        elif [[ -x "$VENV_PATH/bin/mlflow" ]]; then
            echo "$VENV_PATH/bin/mlflow"
            return 0
        else
            echo "Error: mlflow not found in VENV_PATH=$VENV_PATH" >&2
            return 1
        fi
    fi

    # Project .venv (default convention)
    if [[ -x "$PROJECT_ROOT/.venv/bin/mlflow" ]]; then
        echo "$PROJECT_ROOT/.venv/bin/mlflow"
        return 0
    fi

    # Active virtualenv
    if [[ -n "$VIRTUAL_ENV" && -x "$VIRTUAL_ENV/bin/mlflow" ]]; then
        echo "$VIRTUAL_ENV/bin/mlflow"
        return 0
    fi

    # System mlflow
    if command -v mlflow &> /dev/null; then
        echo "mlflow"
        return 0
    fi

    echo "Error: mlflow not found." >&2
    echo "Install with: uv pip install mlflow" >&2
    echo "Or set VENV_PATH to your virtual environment" >&2
    return 1
}

MLFLOW_CMD=$(find_mlflow) || exit 1

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
echo "  Executable: $MLFLOW_CMD"
echo ""

if [[ "$1" == "--background" ]]; then
    echo "Starting MLflow server in background..."
    nohup "$MLFLOW_CMD" server \
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
    "$MLFLOW_CMD" server \
        --backend-store-uri "sqlite:///$MLFLOW_ROOT/mlflow.db" \
        --default-artifact-root "$MLFLOW_ROOT/mlartifacts" \
        --host "$MLFLOW_HOST" \
        --port "$MLFLOW_PORT"
fi
