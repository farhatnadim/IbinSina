# MLflow Experiment Tracking Guide

This guide covers how to use MLflow tracking with the MIL training pipeline.

---

## Option 1: Training Without MLflow Server (Offline Mode)

When the MLflow server isn't running, training automatically saves tracking data locally.

### Single Training Run

```bash
# No setup needed - just run training normally
python train_mil.py --config configs/my_config.json

# You'll see this message:
# "MLflow unavailable (...), using offline mode"
```

Data is saved to `mlflow_offline/run_<name>_<timestamp>.json`

### Cross-Validation Run

```bash
# Same thing - just run CV normally
python train_mil_cv.py --config configs/my_config.json --num-folds 5

# Offline data includes parent run + nested fold runs
```

### Syncing to Server Later

Once the MLflow server is available:

```bash
# 1. Start the server
./scripts/start_mlflow.sh --background

# 2. Set the tracking URI
export MLFLOW_TRACKING_URI="http://localhost:5000"

# 3. Preview what will be synced (optional)
python scripts/sync_mlflow_offline.py --dry-run

# 4. Sync offline runs to server
python scripts/sync_mlflow_offline.py

# Output:
# Found 2 offline run(s) to sync
# Processing: run_abmil_base_uni_v2_20240401_153022.json
#   Synced and deleted: run_abmil_base_uni_v2_20240401_153022.json
# Processing: run_CV_abmil_base_uni_v2_20240401_160045.json
#   Synced and deleted: run_CV_abmil_base_uni_v2_20240401_160045.json
# Synced: 2
# Failed: 0
```

To keep offline files after syncing (as backup):

```bash
python scripts/sync_mlflow_offline.py --keep
```

---

## Option 2: Training With MLflow Server

### Step 1: Start the MLflow Server

```bash
# Set where MLflow stores data (optional, defaults to ~/mlflow)
export MLFLOW_ROOT="/path/to/shared/mlflow"

# Start in foreground (see logs directly)
./scripts/start_mlflow.sh

# Or start in background
./scripts/start_mlflow.sh --background

# Check it's running
curl http://localhost:5000/health
# Returns: {"status": "OK"}
```

### Step 2: Set Tracking URI

```bash
# For local server
export MLFLOW_TRACKING_URI="http://localhost:5000"

# For remote server (e.g., on enclave)
export MLFLOW_TRACKING_URI="http://enclave-hostname:5000"

# Add to ~/.bashrc to make permanent
echo 'export MLFLOW_TRACKING_URI="http://localhost:5000"' >> ~/.bashrc
```

### Step 3: Run Training

```bash
# Single training - automatically tracked
python train_mil.py --config configs/my_config.json

# You'll see:
# "MLflow connected: http://localhost:5000"

# Cross-validation - parent run + nested fold runs
python train_mil_cv.py --config configs/my_config.json --num-folds 5
```

### Step 4: View Results in MLflow UI

Open in browser: `http://localhost:5000`

You'll see:
- List of all runs
- Parameters (model, learning rate, epochs, etc.)
- Metrics (loss curves, accuracy, kappa)
- Artifacts (config.json, results.json)

---

## Disabling MLflow Tracking

If you don't want any tracking (no server, no offline files):

### Option A: In config JSON

```json
{
  "data": { ... },
  "train": { ... },
  "model_name": "abmil.base.uni_v2",
  "num_classes": 6,
  "mlflow": {
    "enabled": false
  }
}
```

### Option B: Disable offline fallback (will error if server unavailable)

```json
{
  "mlflow": {
    "enabled": true,
    "offline_fallback": false
  }
}
```

---

## MLflow Configuration Options

Full mlflow config in your JSON:

```json
{
  "mlflow": {
    "enabled": true,
    "tracking_uri": "http://localhost:5000",
    "experiment_name": "my-experiment",
    "offline_fallback": true,
    "offline_dir": "mlflow_offline"
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable/disable tracking entirely |
| `tracking_uri` | `null` | MLflow server URI (uses `MLFLOW_TRACKING_URI` env if null) |
| `experiment_name` | `"mil-training"` | Experiment name in MLflow UI |
| `offline_fallback` | `true` | Save locally if server unavailable |
| `offline_dir` | `"mlflow_offline"` | Directory for offline JSON files |

### Naming: `run_name` vs `mlflow.experiment_name`

- **`run_name`** (top-level): The name for individual training runs (e.g., "abmil_lr1e4_epochs20")
- **`mlflow.experiment_name`**: The MLflow experiment folder that groups related runs (e.g., "panda-classification")

Example config:
```json
{
  "model_name": "abmil.base.uni_v2.none",
  "run_name": "abmil_lr1e4_epochs20",
  "mlflow": {
    "experiment_name": "panda-classification"
  }
}
```

This creates a run called "abmil_lr1e4_epochs20" inside the "panda-classification" experiment folder.

---

## Managing the MLflow Server

```bash
# Start (foreground)
./scripts/start_mlflow.sh

# Start (background)
./scripts/start_mlflow.sh --background

# Stop
./scripts/stop_mlflow.sh

# View logs (if running in background)
tail -f ~/mlflow/mlflow.log

# Check if running
curl http://localhost:5000/health
```

### Custom Port or Host

```bash
export MLFLOW_PORT=5001
export MLFLOW_HOST=0.0.0.0
./scripts/start_mlflow.sh
```

---

## Quick Reference

| Scenario | Command |
|----------|---------|
| Train (server running) | `export MLFLOW_TRACKING_URI=http://localhost:5000 && python train_mil.py --config config.json` |
| Train (no server) | `python train_mil.py --config config.json` (auto offline mode) |
| CV (server running) | `python train_mil_cv.py --config config.json --num-folds 5` |
| CV (no server) | Same command (auto offline mode) |
| Sync offline to server | `python scripts/sync_mlflow_offline.py` |
| Preview sync | `python scripts/sync_mlflow_offline.py --dry-run` |
| Start server | `./scripts/start_mlflow.sh --background` |
| Stop server | `./scripts/stop_mlflow.sh` |
| View UI | Open `http://localhost:5000` in browser |

---

## What Gets Tracked

| Category | Single Training | Cross-Validation |
|----------|-----------------|------------------|
| **Parameters** | model_name, learning_rate, epochs, batch_size, etc. | Same + num_folds, test_frac, cv_seed |
| **Per-Epoch Metrics** | train_loss, val_loss, val_accuracy, val_kappa, val_auc | Same (per fold in nested runs) |
| **Final Metrics** | test_accuracy, test_balanced_accuracy, test_quadratic_kappa | cv_mean_*, cv_std_*, test_* |
| **Artifacts** | config.json, results.json | cv_config.json, cv_results.json |
| **Run Structure** | Single run | Parent run + nested run per fold |
