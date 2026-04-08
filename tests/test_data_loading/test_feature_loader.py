"""Tests for data_loading/feature_loader.py - H5 loading functionality."""

import pytest
import torch
import numpy as np
import h5py
from pathlib import Path

from downstream.classification.multiple_instance_learning.data_loading.feature_loader import (
    load_features,
    load_features_batch,
    get_slide_ids,
    get_embed_dim,
    prepare_for_mil,
    batch_for_mil,
    CLAMFeatureLoader,
)


class TestLoadFeatures:
    """Tests for load_features function."""

    def test_load_features_basic(self, sample_h5_file, sample_features):
        """Test loading features from H5 file."""
        features = load_features(sample_h5_file)

        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert features.shape == sample_features.shape

    def test_load_features_2d_format(self, temp_dir):
        """Test loading 2D features [M, D]."""
        path = temp_dir / "slide_2d.h5"
        data = np.random.randn(50, 256).astype(np.float32)

        with h5py.File(path, 'w') as f:
            f.create_dataset('features', data=data)

        features = load_features(path)

        assert features.shape == (50, 256)
        assert features.dim() == 2

    def test_load_features_3d_format_squeeze(self, temp_dir):
        """Test loading 3D features [1, M, D] gets squeezed."""
        path = temp_dir / "slide_3d.h5"
        data = np.random.randn(1, 50, 256).astype(np.float32)

        with h5py.File(path, 'w') as f:
            f.create_dataset('features', data=data)

        features = load_features(path)

        # Should be squeezed to 2D
        assert features.shape == (50, 256)
        assert features.dim() == 2

    def test_load_features_single_patch(self, temp_dir):
        """Test loading single patch (edge case)."""
        path = temp_dir / "slide_single.h5"
        data = np.random.randn(256,).astype(np.float32)  # 1D

        with h5py.File(path, 'w') as f:
            f.create_dataset('features', data=data)

        features = load_features(path)

        # Should be expanded to 2D [1, D]
        assert features.shape == (1, 256)
        assert features.dim() == 2


class TestLoadFeaturesBatch:
    """Tests for load_features_batch function."""

    def test_load_features_batch(self, sample_features_dir):
        """Test batch loading multiple slides."""
        slide_ids = ['slide_001', 'slide_002']
        features_dict = load_features_batch(sample_features_dir, slide_ids)

        assert len(features_dict) == 2
        assert 'slide_001' in features_dict
        assert 'slide_002' in features_dict

        for slide_id, features in features_dict.items():
            assert isinstance(features, torch.Tensor)
            assert features.dim() == 2

    def test_load_features_batch_missing_slide(self, sample_features_dir):
        """Test batch loading with missing slide."""
        slide_ids = ['slide_001', 'nonexistent_slide']
        features_dict = load_features_batch(sample_features_dir, slide_ids)

        # Only existing slide should be returned
        assert len(features_dict) == 1
        assert 'slide_001' in features_dict
        assert 'nonexistent_slide' not in features_dict


class TestGetSlideIds:
    """Tests for get_slide_ids function."""

    def test_get_slide_ids(self, sample_features_dir):
        """Test getting all slide IDs from directory."""
        slide_ids = get_slide_ids(sample_features_dir)

        assert len(slide_ids) == 8  # Updated fixture has 8 slides
        assert 'slide_001' in slide_ids
        assert 'slide_002' in slide_ids

    def test_get_slide_ids_sorted(self, sample_features_dir):
        """Test that slide IDs are sorted."""
        slide_ids = get_slide_ids(sample_features_dir)

        assert slide_ids == sorted(slide_ids)

    def test_get_slide_ids_empty_dir(self, temp_dir):
        """Test getting slide IDs from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        slide_ids = get_slide_ids(empty_dir)

        assert slide_ids == []


class TestGetEmbedDim:
    """Tests for get_embed_dim function."""

    def test_get_embed_dim(self, sample_features_dir):
        """Test inferring embedding dimension."""
        embed_dim = get_embed_dim(sample_features_dir)

        assert embed_dim == 512

    def test_get_embed_dim_empty_dir(self, temp_dir):
        """Test error with empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No .h5 files"):
            get_embed_dim(empty_dir)


class TestPrepareForMIL:
    """Tests for prepare_for_mil function."""

    def test_prepare_for_mil_2d(self):
        """Test preparing 2D tensor for MIL."""
        features = torch.randn(100, 512)

        prepared = prepare_for_mil(features)

        assert prepared.shape == (1, 100, 512)
        assert prepared.dim() == 3

    def test_prepare_for_mil_already_3d(self):
        """Test that 3D tensor is unchanged."""
        features = torch.randn(1, 100, 512)

        prepared = prepare_for_mil(features)

        assert prepared.shape == (1, 100, 512)


class TestBatchForMIL:
    """Tests for batch_for_mil function."""

    def test_batch_for_mil_padding(self):
        """Test batching with padding."""
        features_list = [
            torch.randn(10, 64),
            torch.randn(20, 64),
            torch.randn(15, 64),
        ]

        padded, mask = batch_for_mil(features_list)

        # Should pad to max length (20)
        assert padded.shape == (3, 20, 64)
        assert mask.shape == (3, 20)

    def test_batch_for_mil_mask(self):
        """Test that mask is correct."""
        features_list = [
            torch.randn(5, 32),
            torch.randn(10, 32),
        ]

        padded, mask = batch_for_mil(features_list)

        # First sample: 5 valid patches
        assert mask[0, :5].sum() == 5
        assert mask[0, 5:].sum() == 0

        # Second sample: all 10 are valid
        assert mask[1].sum() == 10

    def test_batch_for_mil_single_item(self):
        """Test batching single item."""
        features_list = [torch.randn(10, 64)]

        padded, mask = batch_for_mil(features_list)

        assert padded.shape == (1, 10, 64)
        assert mask.shape == (1, 10)
        assert mask.sum() == 10


class TestCLAMFeatureLoader:
    """Tests for CLAMFeatureLoader class."""

    def test_clam_loader_init(self, sample_features_dir):
        """Test CLAMFeatureLoader initialization."""
        loader = CLAMFeatureLoader(sample_features_dir)

        assert len(loader) == 8  # Updated fixture has 8 slides
        assert loader.embed_dim == 512

    def test_clam_loader_with_subset(self, sample_features_dir):
        """Test loader with subset of slides."""
        loader = CLAMFeatureLoader(
            sample_features_dir,
            slide_ids=['slide_001', 'slide_002'],
        )

        assert len(loader) == 2

    def test_clam_loader_iteration(self, sample_features_dir):
        """Test iterating over loader."""
        loader = CLAMFeatureLoader(sample_features_dir)

        slides = list(loader)

        assert len(slides) == 8  # Updated fixture has 8 slides
        for slide_id, features in slides:
            assert isinstance(slide_id, str)
            assert isinstance(features, torch.Tensor)
            assert features.dim() == 2

    def test_clam_loader_getitem(self, sample_features_dir):
        """Test getitem access."""
        loader = CLAMFeatureLoader(sample_features_dir)

        features = loader['slide_001']

        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
