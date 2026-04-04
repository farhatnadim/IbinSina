# CLAUDE.md

This file provides context for Claude when working with this codebase.

## Project Overview

MIL-Lab is a Multiple Instance Learning (MIL) library for computational pathology. It provides standardized implementations of MIL models and pretrained weights (FEATHER models) for whole-slide image classification.

## Directory Structure

```
MIL-Lab/
├── src/                    # Core library code
│   ├── models/             # MIL model implementations (ABMIL, TransMIL, CLAM, etc.)
│   ├── components/         # Shared layers and attention mechanisms
│   ├── builders/           # Model registry and factory
│   └── builder.py          # create_model() entry point
├── training/               # Training infrastructure
│   ├── config.py           # ExperimentConfig, DataConfig, TrainConfig dataclasses
│   ├── trainer.py          # Training loop
│   ├── evaluator.py        # Evaluation metrics
│   └── mlflow_tracking.py  # MLflow wrapper with offline fallback
├── train_mil.py            # Single training run script
├── train_mil_cv.py         # Cross-validation training script
├── configs/                # JSON experiment configurations
├── scripts/                # Utility scripts (MLflow server, sync)
├── data_loading/           # Data loading utilities
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## Key Commands

### Training

```bash
# Single training run
python train_mil.py --config configs/panda_multiclass_config.json

# Cross-validation
python train_mil_cv.py --config configs/panda_multiclass_config.json --num-folds 5
```

### MLflow

```bash
# Start MLflow server
./scripts/start_mlflow.sh --background

# Set tracking URI
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Sync offline runs to server
python scripts/sync_mlflow_offline.py
```

## Configuration

Experiment configs are JSON files with this structure:

```json
{
  "data": {
    "labels_csv": "path/to/labels.csv",
    "features_dir": "path/to/features/",
    "split_column": "split"
  },
  "train": {
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "task_type": "multiclass",
    "early_stopping_metric": "kappa"
  },
  "model_name": "abmil.base.uni_v2.none",
  "run_name": "experiment_name_here",
  "num_classes": 6,
  "output_dir": "experiments/my_experiment",
  "mlflow": {
    "enabled": true,
    "experiment_name": "my-experiment-group"
  }
}
```

### Key Config Fields

- `run_name`: Name for individual training runs (shown in MLflow UI)
- `mlflow.experiment_name`: Groups related runs together in MLflow
- `mlflow.enabled`: Set `false` to disable all tracking
- `mlflow.offline_fallback`: If `true`, saves locally when server unavailable

## Model Naming Convention

Models follow the pattern: `<model>.<config>.<encoder>.<pretrain>`

Examples:
- `abmil.base.uni_v2.none` - ABMIL with UNIv2 encoder, no pretraining
- `abmil.base.uni_v2.pc108-24k` - ABMIL pretrained on PC-108 task (24K slides)

## Code Style

- Use dataclasses for configuration
- Direct imports preferred over `__init__.py` re-exports
- Type hints encouraged
- Keep functions focused and small

## Testing

The project uses standard pytest. Virtual environment is in `.venv/`.

## Documentation

See `docs/` for guides:
- `docs/mlflow_guide.md` - MLflow tracking setup and usage
- `docs/FUSION_STRATEGIES.md` - Multi-slide fusion approaches
