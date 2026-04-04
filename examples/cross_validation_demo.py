#!/usr/bin/env python3
"""
Cross-Validation Demo

Demonstrates how to create reproducible CV splits with a held-out test set.

Usage:
    python examples/cross_validation_demo.py --mock
    python examples/cross_validation_demo.py --labels /path/to/labels.csv --features /path/to/features/
"""

import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loading.dataset import MILDataset
from data_loading.pytorch_adapter import create_dataloader


def create_mock_data(tmpdir: str):
    """Create mock data for demonstration."""
    import h5py

    features_dir = Path(tmpdir) / 'features'
    features_dir.mkdir()

    slides, labels, case_ids = [], [], []
    slide_counts = [1, 2, 3, 2, 1, 3, 2, 4, 2, 3, 1, 2, 3, 2, 1, 4, 2, 3, 1, 2,
                    3, 2, 1, 4, 2, 3, 1, 2, 3, 2]
    patient_labels = ['0', '0', '1', '1', '0', '1', '2', '2', '1', '0',
                      '0', '1', '2', '0', '1', '2', '0', '1', '2', '0',
                      '1', '2', '0', '1', '2', '0', '1', '2', '0', '1']

    slide_idx = 0
    for patient_idx, (n_slides, label) in enumerate(zip(slide_counts, patient_labels)):
        for _ in range(n_slides):
            slides.append(f'slide_{slide_idx:03d}')
            labels.append(label)
            case_ids.append(f'patient_{patient_idx:03d}')
            slide_idx += 1

    labels_csv = Path(tmpdir) / 'labels.csv'
    pd.DataFrame({'slide_id': slides, 'case_id': case_ids, 'label': labels}).to_csv(labels_csv, index=False)

    for slide_id in slides:
        with h5py.File(features_dir / f'{slide_id}.h5', 'w') as f:
            f.create_dataset('features', data=np.random.randn(np.random.randint(50, 200), 1024).astype(np.float32))

    return str(labels_csv), str(features_dir)


def demo_cv_splits(labels_csv: str, features_dir: str):
    """Demonstrate cross-validation split creation."""
    print("=" * 70)
    print("CROSS-VALIDATION SPLITS")
    print("=" * 70)

    # Load and group by patient
    dataset = MILDataset(labels_csv, features_dir)
    grouped = dataset.concat_by('case_id')
    print(f"\nDataset: {len(grouped)} patients ({len(dataset)} slides)")

    # Create CV splits
    print("\n" + "-" * 70)
    print("Creating 5-fold CV with 20% held-out test set")
    print("-" * 70)

    test_dataset, folds = grouped.create_cv_splits(num_folds=5, test_frac=0.2, seed=42)

    # Show fold details
    print("\nFold composition:")
    for i, (train_ds, val_ds) in enumerate(folds):
        train_labels = [grouped[gid].label for gid in train_ds.group_ids]
        val_labels = [grouped[gid].label for gid in val_ds.group_ids]
        print(f"  Fold {i}: train={len(train_ds)} {dict(pd.Series(train_labels).value_counts())}, "
              f"val={len(val_ds)} {dict(pd.Series(val_labels).value_counts())}")

    # Show test set
    test_labels = [grouped[gid].label for gid in test_dataset.group_ids]
    print(f"\nTest set: {len(test_dataset)} patients {dict(pd.Series(test_labels).value_counts())}")

    # DataLoader creation example
    print("\n" + "-" * 70)
    print("DataLoader Creation")
    print("-" * 70)

    train_ds, val_ds = folds[0]
    train_loader, adapter = create_dataloader(train_ds, batch_size=4, num_workers=0)
    val_loader, _ = create_dataloader(val_ds, batch_size=1, label_map=adapter.label_map, num_workers=0)
    test_loader, _ = create_dataloader(test_dataset, batch_size=1, label_map=adapter.label_map, num_workers=0)

    print(f"\nFold 0 loaders:")
    print(f"  train: {len(train_loader)} batches")
    print(f"  val:   {len(val_loader)} batches")
    print(f"  test:  {len(test_loader)} batches")
    print(f"  label_map: {adapter.label_map}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Validation Demo')
    parser.add_argument('--labels', type=str, help='Path to labels CSV')
    parser.add_argument('--features', type=str, help='Path to features directory')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    args = parser.parse_args()

    if args.mock or (args.labels is None and args.features is None):
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_csv, features_dir = create_mock_data(tmpdir)
            demo_cv_splits(labels_csv, features_dir)
    else:
        demo_cv_splits(args.labels, args.features)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
