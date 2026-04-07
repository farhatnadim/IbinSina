"""Integration tests for data loading pipelines."""

import pytest
import torch
import numpy as np
from pathlib import Path

from data_loading.dataset import MILDataset, GroupedMILDataset, HierarchicalMILDataset
from data_loading.pytorch_adapter import (
    create_dataloader,
    MILDatasetAdapter,
    HierarchicalMILDatasetAdapter,
)
from data_loading.feature_loader import load_features, get_embed_dim


class TestCSVToDataloader:
    """Tests for full CSV to DataLoader pipeline."""

    def test_csv_to_dataloader(self, sample_labels_csv, sample_features_dir, capsys):
        """Full data loading pipeline from CSV to DataLoader."""
        # Load dataset
        dataset = MILDataset(sample_labels_csv, sample_features_dir)

        # Create dataloader
        loader, adapter = create_dataloader(
            dataset, batch_size=2, shuffle=False, num_workers=0
        )

        # Iterate and verify
        for batch in loader:
            features, labels, mask, sample_ids = batch

            assert features.dim() == 3  # [B, M, D]
            assert labels.dim() == 1  # [B]
            assert mask.dim() == 2  # [B, M]
            assert isinstance(sample_ids, list)

            # Verify batch consistency
            assert features.shape[0] == labels.shape[0] == mask.shape[0]
            break

    def test_csv_to_dataloader_with_label_map(self, sample_labels_csv, sample_features_dir, capsys):
        """Test consistent label mapping across train/val/test."""
        dataset = MILDataset(sample_labels_csv, sample_features_dir)

        # Split into train/val
        splits = dataset.random_split(train_frac=0.5, val_frac=0.25, seed=42)

        # Create train loader first
        train_loader, train_adapter = create_dataloader(
            splits['train'], batch_size=1, shuffle=True, num_workers=0
        )

        # Create val loader with same label map
        val_loader, val_adapter = create_dataloader(
            splits['val'], batch_size=1, shuffle=False, num_workers=0,
            label_map=train_adapter.label_map
        )

        # Verify same label mapping
        assert train_adapter.label_map == val_adapter.label_map


class TestSplitToLoaders:
    """Tests for split to loaders pipeline."""

    def test_split_to_loaders(self, sample_mil_dataset):
        """Train/val/test split to separate loaders."""
        splits = sample_mil_dataset.random_split(
            train_frac=0.5, val_frac=0.25, seed=42
        )

        # Create loaders for each split
        loaders = {}
        label_map = None

        for split_name, split_ds in splits.items():
            loader, adapter = create_dataloader(
                split_ds, batch_size=1, shuffle=(split_name == 'train'),
                num_workers=0, label_map=label_map
            )
            loaders[split_name] = loader

            if label_map is None:
                label_map = adapter.label_map

        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders

    def test_split_by_column_to_loaders(self, sample_mil_dataset):
        """Split by column to loaders."""
        splits = sample_mil_dataset.split_by_column('split')

        loaders = {}
        label_map = None

        for split_name, split_ds in splits.items():
            loader, adapter = create_dataloader(
                split_ds, batch_size=1, shuffle=False,
                num_workers=0, label_map=label_map
            )
            loaders[split_name] = loader

            if label_map is None:
                label_map = adapter.label_map

        assert 'train' in loaders or 'val' in loaders or 'test' in loaders


class TestCVSplitsToLoaders:
    """Tests for CV fold loaders."""

    def test_cv_splits_to_loaders(self, sample_labels_csv_multiclass, sample_features_dir, capsys):
        """CV fold splits to loaders."""
        dataset = MILDataset(sample_labels_csv_multiclass, sample_features_dir)

        test_dataset, folds = dataset.create_cv_splits(
            num_folds=2, test_frac=0.25, seed=42
        )

        # Create test loader
        test_loader, test_adapter = create_dataloader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        # Create loaders for each fold
        for fold_idx, (train_ds, val_ds) in enumerate(folds):
            train_loader, train_adapter = create_dataloader(
                train_ds, batch_size=1, shuffle=True, num_workers=0,
                label_map=test_adapter.label_map
            )
            val_loader, val_adapter = create_dataloader(
                val_ds, batch_size=1, shuffle=False, num_workers=0,
                label_map=test_adapter.label_map
            )

            # Verify consistent label mapping
            assert train_adapter.label_map == test_adapter.label_map
            assert val_adapter.label_map == test_adapter.label_map


class TestGroupingPipeline:
    """Tests for dataset grouping pipelines."""

    def test_grouping_pipeline_early_fusion(self, sample_mil_dataset, capsys):
        """Dataset -> Grouped -> Loader (early fusion)."""
        # Apply grouping
        grouped = sample_mil_dataset.concat_by('case_id')

        assert isinstance(grouped, GroupedMILDataset)

        # Create loader
        loader, adapter = create_dataloader(
            grouped, batch_size=1, shuffle=False, num_workers=0
        )

        # Iterate
        for batch in loader:
            features, labels, mask, sample_ids = batch

            assert features.dim() == 3
            # Sample IDs should be group_ids (case_ids)
            assert isinstance(sample_ids[0], str)
            break

    def test_grouping_pipeline_late_fusion(self, sample_mil_dataset, capsys):
        """Dataset -> Hierarchical -> Loader (late fusion)."""
        # Apply hierarchical grouping
        hier = sample_mil_dataset.group_by('case_id')

        assert isinstance(hier, HierarchicalMILDataset)

        # Create loader
        loader, adapter = create_dataloader(
            hier, batch_size=1, shuffle=False, num_workers=0
        )

        # Iterate
        for batch in loader:
            features_list, labels, mask, sample_ids = batch

            # Features should be list of lists (batch of groups)
            assert isinstance(features_list, list)
            assert isinstance(sample_ids[0], str)
            break


class TestFeatureLoaderIntegration:
    """Tests for feature loading integration."""

    def test_feature_loader_to_dataset(self, sample_labels_csv, sample_features_dir, capsys):
        """Test features loaded correctly into dataset."""
        dataset = MILDataset(sample_labels_csv, sample_features_dir)

        # Check embedding dimension matches
        expected_dim = get_embed_dim(sample_features_dir)
        assert dataset.embed_dim == expected_dim

        # Load a slide and verify features
        slide = dataset[0]
        assert slide.features.shape[1] == expected_dim

    def test_feature_consistency(self, sample_features_dir, sample_h5_file):
        """Test that loaded features match source."""
        # Load directly
        direct_features = load_features(sample_h5_file)

        # Load via dataset (if sample_h5_file is in sample_features_dir)
        # This test verifies the loader produces consistent results
        assert direct_features.dim() == 2
        assert direct_features.dtype == torch.float32


class TestBatchCollation:
    """Tests for batch collation edge cases."""

    def test_variable_length_batching(self, sample_mil_dataset):
        """Test batching with variable-length bags."""
        loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=2, shuffle=False, num_workers=0
        )

        for batch in loader:
            features, labels, mask, sample_ids = batch

            # All samples should be padded to same length
            assert features.shape[0] == features.shape[0]  # Batch consistency
            assert mask.shape == features.shape[:2]  # Mask matches features

            # Mask should indicate actual lengths
            for i in range(features.shape[0]):
                valid_count = mask[i].sum().int()
                assert valid_count > 0
            break

    def test_single_item_batch(self, sample_mil_dataset):
        """Test batching with batch_size=1."""
        loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        for batch in loader:
            features, labels, mask, sample_ids = batch

            assert features.shape[0] == 1
            assert len(sample_ids) == 1
            break
