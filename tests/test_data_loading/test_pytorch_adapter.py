"""Tests for data_loading/pytorch_adapter.py - Adapters, collate functions, dataloader."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import (
    MILDatasetAdapter,
    HierarchicalMILDatasetAdapter,
    mil_collate_fn,
    hierarchical_collate_fn,
    create_dataloader,
)


class TestMILDatasetAdapter:
    """Tests for MILDatasetAdapter class."""

    def test_mil_adapter_getitem(self, sample_mil_dataset):
        """Test that adapter returns (features, label, sample_id) tuple."""
        adapter = MILDatasetAdapter(sample_mil_dataset)

        features, label, sample_id = adapter[0]

        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2  # [M, D]
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
        assert isinstance(sample_id, str)

    def test_mil_adapter_label_map(self, sample_mil_dataset):
        """Test label encoding with custom label map."""
        custom_map = {'benign': 0, 'tumor': 1}
        adapter = MILDatasetAdapter(sample_mil_dataset, label_map=custom_map)

        # Get a sample and verify label encoding
        _, label, _ = adapter[0]
        slide = sample_mil_dataset[0]

        assert label.item() == custom_map[slide.label]

    def test_mil_adapter_auto_label_map(self, sample_mil_dataset):
        """Test automatic label map creation."""
        adapter = MILDatasetAdapter(sample_mil_dataset)

        # Should create sorted label map automatically
        assert 'benign' in adapter.label_map
        assert 'tumor' in adapter.label_map
        assert adapter.label_map['benign'] == 0  # Alphabetically first
        assert adapter.label_map['tumor'] == 1

    def test_mil_adapter_inverse_label_map(self, sample_mil_dataset):
        """Test inverse label map."""
        adapter = MILDatasetAdapter(sample_mil_dataset)

        for label_str, label_int in adapter.label_map.items():
            assert adapter.inverse_label_map[label_int] == label_str

    def test_mil_adapter_num_classes(self, sample_mil_dataset):
        """Test num_classes property."""
        adapter = MILDatasetAdapter(sample_mil_dataset)
        assert adapter.num_classes == 2  # benign and tumor

    def test_mil_adapter_embed_dim(self, sample_mil_dataset):
        """Test embed_dim property."""
        adapter = MILDatasetAdapter(sample_mil_dataset)
        assert adapter.embed_dim == 512

    def test_mil_adapter_get_labels(self, sample_mil_dataset):
        """Test get_labels method for weighted sampling."""
        adapter = MILDatasetAdapter(sample_mil_dataset)
        labels = adapter.get_labels()

        assert len(labels) == len(sample_mil_dataset)
        assert all(isinstance(l, int) for l in labels)

    def test_mil_adapter_len(self, sample_mil_dataset):
        """Test __len__ method."""
        adapter = MILDatasetAdapter(sample_mil_dataset)
        assert len(adapter) == len(sample_mil_dataset)


class TestHierarchicalMILDatasetAdapter:
    """Tests for HierarchicalMILDatasetAdapter class."""

    def test_hierarchical_adapter_getitem(self, sample_mil_dataset):
        """Test that hierarchical adapter returns correct format."""
        hier_dataset = sample_mil_dataset.group_by('case_id')
        adapter = HierarchicalMILDatasetAdapter(hier_dataset)

        features_list, label, group_id = adapter[0]

        assert isinstance(features_list, list)
        assert all(isinstance(f, torch.Tensor) for f in features_list)
        assert isinstance(label, torch.Tensor)
        assert isinstance(group_id, str)

    def test_hierarchical_adapter_label_encoding(self, sample_mil_dataset):
        """Test label encoding in hierarchical adapter."""
        hier_dataset = sample_mil_dataset.group_by('case_id')
        adapter = HierarchicalMILDatasetAdapter(hier_dataset)

        _, label, _ = adapter[0]
        assert label.dtype == torch.long

    def test_hierarchical_adapter_get_labels(self, sample_mil_dataset):
        """Test get_labels for hierarchical adapter."""
        hier_dataset = sample_mil_dataset.group_by('case_id')
        adapter = HierarchicalMILDatasetAdapter(hier_dataset)
        labels = adapter.get_labels()

        assert len(labels) == len(hier_dataset)


class TestMILCollateFn:
    """Tests for mil_collate_fn function."""

    def test_mil_collate_fn_padding(self):
        """Test that collate function pads features correctly."""
        batch = [
            (torch.randn(10, 64), torch.tensor(0), 'slide_001'),
            (torch.randn(20, 64), torch.tensor(1), 'slide_002'),
            (torch.randn(15, 64), torch.tensor(0), 'slide_003'),
        ]

        padded, labels, mask, sample_ids = mil_collate_fn(batch)

        # Check shapes
        assert padded.shape == (3, 20, 64)  # Padded to max length (20)
        assert labels.shape == (3,)
        assert mask.shape == (3, 20)

    def test_mil_collate_fn_mask(self):
        """Test that mask correctly indicates valid positions."""
        batch = [
            (torch.randn(5, 32), torch.tensor(0), 'slide_001'),
            (torch.randn(10, 32), torch.tensor(1), 'slide_002'),
        ]

        _, _, mask, _ = mil_collate_fn(batch)

        # First sample: 5 valid patches
        assert mask[0, :5].sum() == 5
        assert mask[0, 5:].sum() == 0

        # Second sample: 10 valid patches (all)
        assert mask[1, :10].sum() == 10

    def test_mil_collate_fn_sample_ids(self):
        """Test that sample_ids are correctly returned."""
        batch = [
            (torch.randn(10, 64), torch.tensor(0), 'sample_a'),
            (torch.randn(10, 64), torch.tensor(1), 'sample_b'),
        ]

        _, _, _, sample_ids = mil_collate_fn(batch)

        assert sample_ids == ['sample_a', 'sample_b']

    def test_mil_collate_fn_single_item(self):
        """Test collate with single item batch."""
        batch = [
            (torch.randn(10, 64), torch.tensor(0), 'slide_001'),
        ]

        padded, labels, mask, sample_ids = mil_collate_fn(batch)

        assert padded.shape == (1, 10, 64)
        assert labels.shape == (1,)
        assert mask.shape == (1, 10)
        assert sample_ids == ['slide_001']


class TestHierarchicalCollateFn:
    """Tests for hierarchical_collate_fn function."""

    def test_hierarchical_collate_fn(self):
        """Test hierarchical collate function."""
        batch = [
            # Each item: (list of tensors, label, group_id)
            ([torch.randn(10, 64), torch.randn(15, 64)], torch.tensor(0), 'case_1'),
            ([torch.randn(20, 64)], torch.tensor(1), 'case_2'),
        ]

        features_lists, labels, masks, sample_ids = hierarchical_collate_fn(batch)

        assert len(features_lists) == 2
        assert labels.shape == (2,)
        assert sample_ids == ['case_1', 'case_2']

    def test_hierarchical_collate_preserves_structure(self):
        """Test that hierarchical collate preserves list structure."""
        batch = [
            ([torch.randn(10, 64), torch.randn(15, 64)], torch.tensor(0), 'case_1'),
        ]

        features_lists, _, _, _ = hierarchical_collate_fn(batch)

        # Should preserve the list of tensors
        assert isinstance(features_lists[0], list)
        assert len(features_lists[0]) == 2


class TestCreateDataloader:
    """Tests for create_dataloader function."""

    def test_create_dataloader_basic(self, sample_mil_dataset):
        """Test basic dataloader creation."""
        loader, adapter = create_dataloader(
            sample_mil_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        assert len(loader) > 0
        assert isinstance(adapter, MILDatasetAdapter)

    def test_create_dataloader_with_label_map(self, sample_mil_dataset):
        """Test dataloader with custom label map."""
        label_map = {'benign': 0, 'tumor': 1}
        loader, adapter = create_dataloader(
            sample_mil_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            label_map=label_map,
        )

        assert adapter.label_map == label_map

    def test_create_dataloader_weighted_sampling(self, sample_mil_dataset):
        """Test dataloader with weighted sampling."""
        loader, adapter = create_dataloader(
            sample_mil_dataset,
            batch_size=1,
            shuffle=False,  # Should be overridden by weighted_sampling
            num_workers=0,
            weighted_sampling=True,
        )

        # Weighted sampler should be present
        assert loader.sampler is not None

    def test_create_dataloader_hierarchical(self, sample_mil_dataset):
        """Test dataloader with hierarchical dataset."""
        hier_dataset = sample_mil_dataset.group_by('case_id')

        loader, adapter = create_dataloader(
            hier_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        assert isinstance(adapter, HierarchicalMILDatasetAdapter)

    def test_create_dataloader_iteration(self, sample_mil_dataset):
        """Test that dataloader can be iterated."""
        loader, _ = create_dataloader(
            sample_mil_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        for batch in loader:
            features, labels, mask, sample_ids = batch
            assert features.dim() == 3  # [B, M, D]
            assert labels.dim() == 1  # [B]
            assert mask.dim() == 2  # [B, M]
            assert isinstance(sample_ids, list)
            break  # Just test first batch
