# Computational Pathology

Training infrastructure for Multiple Instance Learning (MIL) models in computational pathology. Built on top of [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) and [TRIDENT](https://github.com/mahmoodlab/TRIDENT) from the Mahmood Lab.

## Features

- **Single-split training** (`train_mil.py`): Train MIL models with train/val/test splits
- **Cross-validation** (`train_mil_cv.py`): N-fold CV with held-out test set and ensemble evaluation
- **Evaluation** (`eval_mil.py`): Evaluate trained models on labeled test data with metrics
- **Inference** (`infer_mil.py`): Run predictions on unlabeled slides
- **MLflow integration**: Experiment tracking and model logging
- **Flexible configuration**: JSON-based configs for reproducible experiments

## Installation

### Requirements

- Python 3.10 (required, not 3.11+)
- CUDA-capable GPU (recommended)

### Quick Start

```bash
git clone https://github.com/your-org/ComputationalPathology.git
cd ComputationalPathology

# Create virtual environment with Python 3.10
uv venv .venv --python 3.10
source .venv/bin/activate

# Install with all dependencies (automatically installs MIL-Lab and TRIDENT)
uv pip install -e .

# For CLAM model support
pip install git+https://github.com/oval-group/smooth-topk
```

This will automatically install:
- **MIL-Lab** (`src` package) - MIL model implementations and pretrained weights
- **TRIDENT** (`trident` package) - Patch feature extraction
- All other required dependencies

### Manual Installation (Alternative)

If you prefer to install dependencies separately:

```bash
# Create and activate environment
uv venv .venv --python 3.10
source .venv/bin/activate

# Install MIL-Lab and TRIDENT from GitHub
uv pip install git+https://github.com/mahmoodlab/MIL-Lab.git
uv pip install git+https://github.com/mahmoodlab/TRIDENT.git

# Install this package
uv pip install -e .
```

## Usage

### Training

```bash
# Single split training
python train_mil.py --config configs/panda_multiclass_config.json

# Cross-validation (5-fold with 20% held-out test)
python train_mil_cv.py --config configs/panda_multiclass_cv_config.json \
    --num-folds 5 --test-frac 0.2 --seed 42
```

### Evaluation

Evaluate a trained model on labeled test data (computes metrics):

```bash
# With config file
python eval_mil.py --config configs/panda_multiclass_config.json \
    --checkpoint experiments/run_001/best_model.pth

# With explicit arguments
python eval_mil.py \
    --checkpoint experiments/run_001/best_model.pth \
    --model-name abmil.base.uni_v2.none \
    --num-classes 6 \
    --labels-csv /path/to/test_labels.csv \
    --features-dir /path/to/test_features/
```

### Inference

Run predictions on unlabeled slides (no ground truth needed):

```bash
# With config file - all slides in features_dir
python infer_mil.py --config configs/panda_multiclass_config.json \
    --checkpoint experiments/run_001/best_model.pth \
    --output predictions.csv

# Single slide
python infer_mil.py --config configs/panda_multiclass_config.json \
    --checkpoint experiments/run_001/best_model.pth \
    --features /path/to/slide.h5

# With explicit arguments and custom labels
python infer_mil.py \
    --checkpoint experiments/run_001/best_model.pth \
    --model-name abmil.base.uni_v2.none \
    --num-classes 6 \
    --features-dir /path/to/features/ \
    --label-names "Gleason 3+3,Gleason 3+4,Gleason 4+3,Gleason 4+4,Gleason 4+5,Gleason 5+5" \
    --output predictions.csv
```

Output format (CSV):
```csv
slide_id,predicted_label,confidence
slide_001,Gleason 3+4,0.8743
slide_002,Gleason 4+3,0.6521
```

### Configuration

Configs are JSON files specifying data paths, model settings, and training parameters:

```json
{
  "data": {
    "labels_csv": "path/to/labels.csv",
    "features_dir": "path/to/features/"
  },
  "model_name": "abmil",
  "num_classes": 6,
  "train": {
    "epochs": 50,
    "learning_rate": 1e-4,
    "task_type": "multiclass"
  }
}
```

See `configs/` for example configurations.

## Available Models

All MIL models from [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) are supported:

| Model | Description |
|-------|-------------|
| `abmil` | Attention-Based MIL |
| `transmil` | Transformer MIL |
| `clam` | Clustering-constrained Attention MIL |
| `dsmil` | Dual-Stream MIL |
| `dftd` | Double-Tier Feature Distillation |
| `ilra` | Low-Rank Attention MIL |
| `rrt` | Region-Refined Transformer |
| `wikg` | Whole-Slide Image Knowledge Graph |

### Using Pretrained Models (FEATHER)

```python
from src.builder import create_model

# Load pretrained FEATHER model
model = create_model('abmil.base.uni.pc108-24k', num_classes=5)
```

## Project Structure

```
ComputationalPathology/
├── train_mil.py          # Single-split training
├── train_mil_cv.py       # Cross-validation training
├── eval_mil.py           # Model evaluation (with metrics)
├── infer_mil.py          # Inference (predictions only)
├── training/             # Training utilities
│   ├── config.py         # Configuration dataclasses
│   ├── trainer.py        # Training loop
│   ├── evaluator.py      # Evaluation metrics
│   └── mlflow_tracking.py
├── data_loading/         # Data utilities
│   ├── dataset.py        # MIL dataset class
│   ├── feature_loader.py # CLAM H5 feature loading
│   └── pytorch_adapter.py
└── configs/              # Example configurations
```

## License

This project is for research purposes. See [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) and [TRIDENT](https://github.com/mahmoodlab/TRIDENT) for their respective licenses.

## Acknowledgements

Built on top of:
- [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) - Mahmood Lab
- [TRIDENT](https://github.com/mahmoodlab/TRIDENT) - Mahmood Lab
