#!/usr/bin/env python3
"""
Data Loading Demo for Design Review

Demonstrates how the data_loading module prepares data for MIL training.

Usage:
    # Mock data - run all scenarios
    python examples/data_loading_demo.py --mock
    python examples/data_loading_demo.py --mock --scenario single
    python examples/data_loading_demo.py --mock --scenario multi

    # From config - runs only the config's setting
    python examples/data_loading_demo.py --config configs/panda_config.json
"""

import sys
import json
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loading.dataset import MILDataset
from data_loading.pytorch_adapter import create_dataloader


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_mock_data(tmpdir: str, scenario: str = 'single'):
    """Create mock data for demonstration."""
    import h5py

    features_dir = Path(tmpdir) / f'features_{scenario}'
    features_dir.mkdir(exist_ok=True)

    if scenario == 'single':
        data = {
            'slide_id': [f'slide_{i:03d}' for i in range(20)],
            'label': ['0'] * 8 + ['1'] * 7 + ['2'] * 5,
        }
    else:
        slides, labels, case_ids = [], [], []
        slide_counts = [1, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 1, 3, 2, 1, 4, 2, 3, 1, 2,
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

        data = {'slide_id': slides, 'case_id': case_ids, 'label': labels}

    labels_csv = Path(tmpdir) / f'labels_{scenario}.csv'
    pd.DataFrame(data).to_csv(labels_csv, index=False)

    for slide_id in data['slide_id']:
        n_patches = np.random.randint(50, 500)
        with h5py.File(features_dir / f'{slide_id}.h5', 'w') as f:
            f.create_dataset('features', data=np.random.randn(n_patches, 1024).astype(np.float32))

    return str(labels_csv), str(features_dir)


def demo_slide_level(labels_csv: str, features_dir: str):
    """Demo: Slide-level dataset (no grouping)."""
    print("=" * 70)
    print("SLIDE-LEVEL DATASET")
    print("=" * 70)

    dataset = MILDataset(labels_csv, features_dir)
    print(f"\nDataset: {len(dataset)} slides, embed_dim={dataset.embed_dim}, classes={dataset.num_classes}")

    sample = dataset[0]
    print(f"\nSample: {type(sample).__name__}")
    print(f"  slide_id: {sample.slide_id}, label: {sample.label}")
    print(f"  features: {sample.features.shape}  # [M, D]")

    train_loader, adapter = create_dataloader(dataset, batch_size=4, num_workers=0)
    print(f"\nDataLoader: {len(train_loader)} batches, label_map={adapter.label_map}")

    features, labels, mask = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  features: {features.shape}  # [B, M_max, D]")
    print(f"  labels:   {labels.shape}    # [B]")
    print(f"  mask:     {mask.shape}      # [B, M_max] (1=real, 0=pad)")


def demo_early_fusion(labels_csv: str, features_dir: str, group_column: str):
    """Demo: Early fusion (concat_by)."""
    print("=" * 70)
    print(f"EARLY FUSION: concat_by('{group_column}')")
    print("=" * 70)

    base_dataset = MILDataset(labels_csv, features_dir)
    print(f"\nBase dataset: {len(base_dataset)} slides")

    if group_column not in base_dataset.df.columns:
        print(f"\nERROR: '{group_column}' column not found. Available: {list(base_dataset.df.columns)}")
        return

    slides_per_group = base_dataset.df.groupby(group_column).size()
    print(f"Slides/group: min={slides_per_group.min()}, max={slides_per_group.max()}, mean={slides_per_group.mean():.1f}")

    grouped = base_dataset.concat_by(group_column, label_voting='max')
    print(f"\nGrouped: {len(grouped)} groups")

    sample = grouped[0]
    print(f"\nSample: {type(sample).__name__}")
    print(f"  group_id: {sample.group_id}, label: {sample.label}")
    print(f"  features: {sample.features.shape}  # [M_total, D] - all slides concatenated")
    print(f"  item_ids: {sample.item_ids}")

    train_loader, adapter = create_dataloader(grouped, batch_size=2, num_workers=0)
    print(f"\nDataLoader: {len(train_loader)} batches, label_map={adapter.label_map}")

    features, labels, mask = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  features: {features.shape}  # [B, M_max, D]")
    print(f"  labels:   {labels.shape}    # [B]")
    print(f"  mask:     {mask.shape}      # [B, M_max]")


def demo_late_fusion(labels_csv: str, features_dir: str, group_column: str):
    """Demo: Late fusion (group_by)."""
    print("=" * 70)
    print(f"LATE FUSION: group_by('{group_column}')")
    print("=" * 70)

    base_dataset = MILDataset(labels_csv, features_dir)
    print(f"\nBase dataset: {len(base_dataset)} slides")

    if group_column not in base_dataset.df.columns:
        print(f"\nERROR: '{group_column}' column not found. Available: {list(base_dataset.df.columns)}")
        return

    slides_per_group = base_dataset.df.groupby(group_column).size()
    print(f"Slides/group: min={slides_per_group.min()}, max={slides_per_group.max()}, mean={slides_per_group.mean():.1f}")

    hier = base_dataset.group_by(group_column, label_voting='max')
    print(f"\nHierarchical: {len(hier)} groups")

    sample = hier[0]
    print(f"\nSample: {type(sample).__name__}")
    print(f"  group_id: {sample.group_id}, label: {sample.label}")
    print(f"  features: List of {len(sample.features)} tensors (one per slide)")
    for i, feat in enumerate(sample.features):
        print(f"    Slide {i}: {feat.shape}")

    train_loader, adapter = create_dataloader(hier, batch_size=2, num_workers=0)
    print(f"\nDataLoader: {len(train_loader)} batches, label_map={adapter.label_map}")

    features_list, labels, _ = next(iter(train_loader))
    print(f"\nBatch: {len(features_list)} groups, labels={labels.shape}")
    for i, patient in enumerate(features_list[:2]):
        print(f"  Group {i}: {len(patient)} slides, shapes={[f.shape for f in patient]}")


def demo_from_config(config_path: str):
    """Run demo based on config settings."""
    print(f"Loading config: {config_path}\n")
    config = load_config(config_path)

    labels_csv = config['data']['labels_csv']
    features_dir = config['data']['features_dir']
    group_column = config['data'].get('group_column', 'slide_id')
    fusion = config['data'].get('fusion', 'early')
    hierarchical = config['data'].get('hierarchical', False)

    print(f"  labels_csv:   {labels_csv}")
    print(f"  features_dir: {features_dir}")
    print(f"  group_column: {group_column}")
    print(f"  fusion:       {fusion}")
    print(f"  hierarchical: {hierarchical}")
    print()

    # Determine if grouping is needed
    df = pd.read_csv(labels_csv)
    has_grouping = (
        group_column != 'slide_id' and
        group_column in df.columns and
        (df.groupby(group_column).size() > 1).any()
    )

    if not has_grouping:
        demo_slide_level(labels_csv, features_dir)
    elif hierarchical or fusion == 'late':
        demo_late_fusion(labels_csv, features_dir, group_column)
    else:
        demo_early_fusion(labels_csv, features_dir, group_column)


def main():
    parser = argparse.ArgumentParser(description='Data Loading Demo')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--scenario', choices=['single', 'multi', 'both'], default='both',
                        help='Scenario for mock data')
    args = parser.parse_args()

    if args.config:
        demo_from_config(args.config)
    elif args.mock:
        with tempfile.TemporaryDirectory() as tmpdir:
            if args.scenario in ['single', 'both']:
                labels_csv, features_dir = create_mock_data(tmpdir, 'single')
                demo_slide_level(labels_csv, features_dir)
            if args.scenario in ['multi', 'both']:
                labels_csv, features_dir = create_mock_data(tmpdir, 'multi')
                print("\n")
                demo_early_fusion(labels_csv, features_dir, 'case_id')
                print("\n")
                demo_late_fusion(labels_csv, features_dir, 'case_id')
    else:
        parser.print_help()
        return

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
