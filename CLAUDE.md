# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IbinSina is a training infrastructure for Multiple Instance Learning (MIL) models in computational pathology. It provides training, cross-validation, and evaluation pipelines for whole-slide image (WSI) classification. The core MIL models come from the external [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) package (`src`), while this repo contains the training infrastructure.

## Key Commands

### Setup

```bash
# Create venv with Python 3.10 (required - not 3.11+)
uv venv .venv --python 3.10
source .venv/bin/activate

# Install package with dependencies (includes MIL-Lab and TRIDENT)
uv pip install -e .

# For CLAM model support
pip install git+https://github.com/oval-group/smooth-topk
```

### Training

```bash
# Single-split training
python train_mil.py --config configs/panda_multiclass_config.json

# N-fold cross-validation with held-out test set
python train_mil_cv.py --config configs/panda_multiclass_config.json --num-folds 5 --test-frac 0.2

# Evaluate trained model
python eval_mil.py --config config.json --checkpoint best_model.pth
```

### MLflow

```bash
./scripts/start_mlflow.sh --background
export MLFLOW_TRACKING_URI="http://localhost:5000"
python scripts/sync_mlflow_offline.py  # Sync offline runs to server
```

## Architecture

### Directory Structure

```
IbinSina/
├── train_mil.py              # Single-split training entry point
├── train_mil_cv.py           # Cross-validation entry point
├── eval_mil.py               # Model evaluation script
├── training/                 # Training infrastructure
│   ├── config.py             # ExperimentConfig, DataConfig, TrainConfig dataclasses
│   ├── trainer.py            # MILTrainer with AMP, early stopping, gradient clipping
│   ├── evaluator.py          # Metrics: accuracy, kappa, AUC, F1
│   └── mlflow_tracking.py    # MLflowTracker with offline fallback
├── data_loading/             # Data pipeline
│   ├── dataset.py            # MILDataset, GroupedMILDataset, HierarchicalMILDataset
│   ├── feature_loader.py     # CLAM H5 feature loading
│   └── pytorch_adapter.py    # PyTorch wrappers, collation, weighted sampling
├── configs/                  # JSON experiment configurations
├── scripts/                  # MLflow server management
└── docs/                     # mlflow_guide.md, FUSION_STRATEGIES.md
```

### Data Loading Pipeline

Three-tier abstraction for different fusion strategies:

1. **MILDataset** (slide-level): One bag per slide, returns `SlideData(slide_id, features[M,D], label, case_id)`

2. **GroupedMILDataset** (early fusion): Via `.concat_by('case_id')` - concatenates all patches from multi-slide cases into one bag

3. **HierarchicalMILDataset** (late fusion): Via `.group_by('case_id')` - preserves slide structure for hierarchical attention

Use early fusion when slides are arbitrary splits; use late fusion when slides represent distinct samples needing slide-level interpretability.

### Model Loading

Models come from the external MIL-Lab package:

```python
from src.builder import create_model

model = create_model('abmil.base.uni_v2.none', num_classes=6)  # No pretraining
model = create_model('abmil.base.uni_v2.pc108-24k', num_classes=5)  # FEATHER pretrained
```

Model naming pattern: `<model>.<config>.<encoder>.<pretrain>`
- Models: abmil, transmil, clam, dsmil, dftd, ilra, rrt, wikg

### Configuration

Configs are JSON files loaded via `ExperimentConfig.load(path)`:

```json
{
  "data": {
    "labels_csv": "path/to/labels.csv",
    "features_dir": "path/to/features/",
    "split_column": "split",
    "group_column": "case_id",
    "fusion": "early"
  },
  "train": {
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "task_type": "multiclass",
    "early_stopping_metric": "kappa"
  },
  "model_name": "abmil.base.uni_v2.none",
  "num_classes": 6,
  "output_dir": "experiments/my_experiment",
  "run_name": "experiment_name",
  "mlflow": {
    "enabled": true,
    "experiment_name": "my-experiment-group",
    "offline_fallback": true
  }
}
```

Key fields:
- `run_name`: Individual run name (shown in MLflow)
- `mlflow.experiment_name`: Groups related runs
- `train.task_type`: "binary" or "multiclass"
- `train.early_stopping_metric`: "kappa", "balanced_accuracy", "auc", "f1_macro", "accuracy", or "auto"

## Code Style

- Use dataclasses for configuration
- Direct imports preferred over `__init__.py` re-exports
- Type hints encouraged
- Keep functions focused and small

## Output Structure

Training outputs are timestamped directories:
```
experiments/run_YYYYMMDD_HHMMSS/
├── best_model.pth
├── config.json
├── results.json
└── predictions.npz
```

Cross-validation adds fold subdirectories and `cv_results.json` with aggregated metrics.
