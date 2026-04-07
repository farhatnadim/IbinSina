"""Tests for training/utils.py - Utility functions."""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.utils import apply_grouping, save_predictions, save_results_summary


class TestApplyGrouping:
    """Tests for apply_grouping function."""

    def test_apply_grouping_early_fusion(self, sample_mil_dataset, capsys):
        """Test that early fusion calls concat_by."""
        mock_config = MagicMock()
        mock_config.data.group_column = 'case_id'
        mock_config.data.fusion = 'early'
        mock_config.data.hierarchical = False

        result = apply_grouping(sample_mil_dataset, mock_config)

        # Should return GroupedMILDataset (concatenated)
        from data_loading.dataset import GroupedMILDataset
        assert isinstance(result, GroupedMILDataset)

    def test_apply_grouping_late_fusion(self, sample_mil_dataset, capsys):
        """Test that late fusion calls group_by."""
        mock_config = MagicMock()
        mock_config.data.group_column = 'case_id'
        mock_config.data.fusion = 'late'
        mock_config.data.hierarchical = True

        result = apply_grouping(sample_mil_dataset, mock_config)

        # Should return HierarchicalMILDataset
        from data_loading.dataset import HierarchicalMILDataset
        assert isinstance(result, HierarchicalMILDataset)

    def test_apply_grouping_no_groups(self, sample_mil_dataset, capsys):
        """Test no grouping when group_column is None."""
        mock_config = MagicMock()
        mock_config.data.group_column = None

        result = apply_grouping(sample_mil_dataset, mock_config)

        # Should return original dataset unchanged
        assert result is sample_mil_dataset

    def test_apply_grouping_missing_column(self, sample_mil_dataset):
        """Test error when group column doesn't exist."""
        mock_config = MagicMock()
        mock_config.data.group_column = 'nonexistent_column'
        mock_config.data.fusion = 'early'
        mock_config.data.hierarchical = False

        with pytest.raises(ValueError, match="not found in dataset"):
            apply_grouping(sample_mil_dataset, mock_config)

    def test_apply_grouping_single_slide_cases(self, temp_dir, capsys):
        """Test behavior when all cases have single slides."""
        # Create dataset where each case has only one slide
        csv_path = temp_dir / "labels.csv"
        csv_path.write_text(
            "slide_id,label,case_id\n"
            "slide_001,a,case_1\n"
            "slide_002,b,case_2\n"
            "slide_003,c,case_3\n"
        )

        features_dir = temp_dir / "features"
        features_dir.mkdir()
        import h5py
        for slide_id in ['slide_001', 'slide_002', 'slide_003']:
            with h5py.File(features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.zeros((10, 64)))

        from data_loading.dataset import MILDataset
        dataset = MILDataset(csv_path, features_dir)

        mock_config = MagicMock()
        mock_config.data.group_column = 'case_id'
        mock_config.data.fusion = 'early'
        mock_config.data.hierarchical = False

        result = apply_grouping(dataset, mock_config)

        # With all single-slide cases and no hierarchical flag, should return original
        assert result is dataset


class TestSavePredictions:
    """Tests for save_predictions function."""

    def test_save_predictions_with_ids(self, temp_dir):
        """Test saving predictions with sample_ids."""
        path = temp_dir / "predictions.npz"

        labels = [0, 1, 2, 0, 1]
        predictions = [0, 1, 1, 0, 2]
        class_labels = ['class_0', 'class_1', 'class_2']
        sample_ids = ['s1', 's2', 's3', 's4', 's5']

        save_predictions(path, labels, predictions, class_labels, sample_ids)

        assert path.exists()

        # Load and verify
        data = np.load(path, allow_pickle=True)
        assert 'labels' in data
        assert 'predictions' in data
        assert 'class_labels' in data
        assert 'sample_ids' in data

        np.testing.assert_array_equal(data['labels'], labels)
        np.testing.assert_array_equal(data['predictions'], predictions)
        np.testing.assert_array_equal(data['sample_ids'], sample_ids)

    def test_save_predictions_without_ids(self, temp_dir):
        """Test saving predictions without sample_ids (backward compatible)."""
        path = temp_dir / "predictions.npz"

        labels = [0, 1, 2]
        predictions = [0, 1, 2]
        class_labels = ['a', 'b', 'c']

        save_predictions(path, labels, predictions, class_labels)

        assert path.exists()

        data = np.load(path, allow_pickle=True)
        assert 'labels' in data
        assert 'predictions' in data
        assert 'class_labels' in data
        assert 'sample_ids' not in data


class TestSaveResultsSummary:
    """Tests for save_results_summary function."""

    def test_save_results_summary(self, temp_dir):
        """Test saving results summary to JSON."""
        path = temp_dir / "results.json"

        summary = {
            'accuracy': 0.95,
            'kappa': 0.89,
            'model_name': 'test_model',
            'num_epochs': 10,
        }

        save_results_summary(path, summary)

        assert path.exists()

        # Load and verify
        with open(path) as f:
            loaded = json.load(f)

        assert loaded == summary

    def test_save_results_summary_nested(self, temp_dir):
        """Test saving nested summary structure."""
        path = temp_dir / "results.json"

        summary = {
            'metrics': {
                'accuracy': 0.95,
                'kappa': 0.89,
            },
            'config': {
                'learning_rate': 1e-4,
                'epochs': 10,
            },
        }

        save_results_summary(path, summary)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded['metrics']['accuracy'] == 0.95
        assert loaded['config']['learning_rate'] == 1e-4
