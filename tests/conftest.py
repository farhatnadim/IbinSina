"""Shared test fixtures for IbinSina test suite."""

import pytest
import torch
import numpy as np
import tempfile
import h5py
import json
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_features():
    """Sample feature tensor [M, D]."""
    return torch.randn(100, 512)


@pytest.fixture
def sample_features_small():
    """Small feature tensor for quick tests."""
    return torch.randn(10, 256)


@pytest.fixture
def sample_h5_file(temp_dir, sample_features):
    """Create a sample H5 feature file."""
    path = temp_dir / "slide_001.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('features', data=sample_features.numpy())
    return path


@pytest.fixture
def sample_features_dir(temp_dir):
    """Create a directory with multiple H5 feature files."""
    features_dir = temp_dir / "features"
    features_dir.mkdir()

    # Create multiple slides with different patch counts
    # We need enough slides to support stratified CV splits (at least 4 per class)
    slides = [
        ("slide_001", 100, 512),
        ("slide_002", 150, 512),
        ("slide_003", 80, 512),
        ("slide_004", 120, 512),
        ("slide_005", 90, 512),
        ("slide_006", 110, 512),
        ("slide_007", 95, 512),
        ("slide_008", 105, 512),
    ]

    for slide_id, num_patches, embed_dim in slides:
        path = features_dir / f"{slide_id}.h5"
        with h5py.File(path, 'w') as f:
            features = np.random.randn(num_patches, embed_dim).astype(np.float32)
            f.create_dataset('features', data=features)

    return features_dir


@pytest.fixture
def sample_labels_csv(temp_dir):
    """Create sample labels CSV."""
    csv_path = temp_dir / "labels.csv"
    csv_path.write_text(
        "slide_id,label,case_id,split\n"
        "slide_001,benign,case_1,train\n"
        "slide_002,tumor,case_1,train\n"
        "slide_003,benign,case_2,train\n"
        "slide_004,tumor,case_3,val\n"
        "slide_005,benign,case_4,val\n"
        "slide_006,tumor,case_5,test\n"
        "slide_007,benign,case_6,test\n"
        "slide_008,tumor,case_7,train\n"
    )
    return csv_path


@pytest.fixture
def sample_labels_csv_multiclass(temp_dir):
    """Create sample labels CSV with multiclass labels for CV testing."""
    csv_path = temp_dir / "labels.csv"
    csv_path.write_text(
        "slide_id,label,case_id\n"
        "slide_001,grade_0,case_1\n"
        "slide_002,grade_1,case_2\n"
        "slide_003,grade_0,case_3\n"
        "slide_004,grade_1,case_4\n"
        "slide_005,grade_0,case_5\n"
        "slide_006,grade_1,case_6\n"
        "slide_007,grade_0,case_7\n"
        "slide_008,grade_1,case_8\n"
    )
    return csv_path


@pytest.fixture
def sample_mil_dataset(sample_labels_csv, sample_features_dir):
    """Create a complete MIL dataset for testing."""
    from data_loading.dataset import MILDataset
    return MILDataset(sample_labels_csv, sample_features_dir)


@pytest.fixture
def mock_model():
    """Mock MIL model for testing."""
    class MockMILModel(torch.nn.Module):
        def __init__(self, embed_dim=512, num_classes=3):
            super().__init__()
            self.embed_dim = embed_dim
            self.fc = torch.nn.Linear(embed_dim, num_classes)
            self.num_classes = num_classes

        def forward(self, x, loss_fn=None, label=None):
            # x is [B, M, D] - pool over patches
            if isinstance(x, list):
                # Hierarchical input
                pooled = torch.stack([f.mean(dim=0) for f in x])
            else:
                pooled = x.mean(dim=1)  # [B, D]

            logits = self.fc(pooled)

            loss = torch.tensor(0.0)
            if loss_fn is not None and label is not None:
                loss = loss_fn(logits, label)

            return {'logits': logits, 'loss': loss}, None

    return MockMILModel()


@pytest.fixture
def sample_experiment_config(temp_dir, sample_labels_csv, sample_features_dir):
    """Create a sample experiment config dict."""
    return {
        "data": {
            "labels_csv": str(sample_labels_csv),
            "features_dir": str(sample_features_dir),
            "train_frac": 0.7,
            "val_frac": 0.15,
            "seed": 42,
            "num_workers": 0,
            "hierarchical": False,
            "group_column": "case_id",
            "fusion": "early",
            "num_folds": 2,
            "test_frac": 0.2,
            "cv_seed": 42,
        },
        "train": {
            "num_epochs": 2,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "feature_dropout": 0.1,
            "model_dropout": 0.25,
            "early_stopping_patience": 5,
            "min_epochs": 1,
            "max_grad_norm": 1.0,
            "use_amp": False,
            "weighted_sampling": False,
            "seed": 42,
            "task_type": "multiclass",
            "early_stopping_metric": "kappa",
        },
        "model_name": "mock_model",
        "num_classes": 3,
        "output_dir": str(temp_dir / "experiments"),
        "run_name": "test_run",
        "num_heads": 1,
        "tracking": {
            "backend": "none",
            "enabled": False,
            "experiment_name": "test",
        },
    }


@pytest.fixture
def sample_split_json(temp_dir):
    """Create a sample split JSON file."""
    split_path = temp_dir / "split.json"
    split_data = {
        "train": ["case_1", "case_2", "case_3", "case_7"],  # Updated for new fixture
        "val": ["case_4", "case_5"],
        "test": ["case_6"],
    }
    with open(split_path, 'w') as f:
        json.dump(split_data, f)
    return split_path
