"""Tests for training/evaluator.py - Core metrics logic."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from training.evaluator import calculate_metrics, evaluate, _compute_auc_safe


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_multiclass(self):
        """Test multiclass metrics computation."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 2]  # 2 errors
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],  # Wrong
            [0.2, 0.6, 0.2],  # Wrong
            [0.9, 0.05, 0.05],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
        ])

        metrics = calculate_metrics(
            y_true, y_pred, y_prob=y_prob, task_type='multiclass', num_classes=3
        )

        # Check all expected metrics are present
        assert 'accuracy' in metrics
        assert 'balanced_accuracy' in metrics
        assert 'f1_macro' in metrics
        assert 'precision_macro' in metrics
        assert 'quadratic_kappa' in metrics
        assert 'confusion_matrix' in metrics
        assert 'auc' in metrics

        # Verify accuracy (7/9 correct)
        assert metrics['accuracy'] == pytest.approx(7 / 9, rel=1e-3)

        # Verify confusion matrix shape
        assert metrics['confusion_matrix'].shape == (3, 3)

    def test_calculate_metrics_binary(self):
        """Test binary classification metrics."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]  # 2 errors
        y_prob = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],  # Wrong
            [0.3, 0.7],
            [0.8, 0.2],
            [0.6, 0.4],  # Wrong
        ])

        metrics = calculate_metrics(
            y_true, y_pred, y_prob=y_prob, task_type='binary', num_classes=2
        )

        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert metrics['accuracy'] == pytest.approx(4 / 6, rel=1e-3)
        # AUC should be between 0 and 1
        assert 0.0 <= metrics['auc'] <= 1.0

    def test_calculate_metrics_without_probs(self):
        """Test that metrics work without probability predictions."""
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]

        metrics = calculate_metrics(y_true, y_pred, task_type='multiclass')

        assert 'accuracy' in metrics
        assert 'auc' not in metrics  # No AUC without probabilities
        assert metrics['accuracy'] == 1.0

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        y_prob = np.eye(3)[[0, 1, 2, 0, 1, 2]]  # One-hot encoding

        metrics = calculate_metrics(
            y_true, y_pred, y_prob=y_prob, task_type='multiclass', num_classes=3
        )

        assert metrics['accuracy'] == 1.0
        assert metrics['balanced_accuracy'] == 1.0
        assert metrics['quadratic_kappa'] == 1.0


class TestComputeAucSafe:
    """Tests for _compute_auc_safe function."""

    def test_auc_missing_classes(self):
        """Test AUC computation when validation set is missing some classes."""
        # Only classes 0 and 1 present, but model predicts 3 classes
        y_true = [0, 0, 1, 1]
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
        ])

        auc = _compute_auc_safe(y_true, y_prob, 'multiclass', num_classes=3)

        # Should compute AUC for classes 0 and 1 only
        assert 0.0 <= auc <= 1.0

    def test_auc_single_class(self):
        """Test AUC returns 0 gracefully when only one class present."""
        y_true = [0, 0, 0, 0]  # Only one class
        y_prob = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ])

        auc = _compute_auc_safe(y_true, y_prob, 'binary', num_classes=2)

        assert auc == 0.0

    def test_auc_binary_both_classes(self):
        """Test binary AUC with both classes present."""
        y_true = [0, 1, 0, 1]
        y_prob = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7],
        ])

        auc = _compute_auc_safe(y_true, y_prob, 'binary', num_classes=2)

        assert 0.0 < auc <= 1.0

    def test_auc_multiclass_ovr(self):
        """Test multiclass OvR AUC computation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ])

        auc = _compute_auc_safe(y_true, y_prob, 'multiclass', num_classes=3)

        assert 0.0 <= auc <= 1.0


class TestEvaluate:
    """Tests for evaluate function."""

    def test_evaluate_with_sample_ids(self, mock_model):
        """Test that sample_ids are returned when provided in batch."""
        # Create mock dataloader that returns sample_ids
        mock_batch = (
            torch.randn(2, 10, 512),  # features
            torch.tensor([0, 1]),  # labels
            torch.ones(2, 10),  # mask
            ['sample_1', 'sample_2'],  # sample_ids
        )

        mock_loader = [mock_batch]

        device = torch.device('cpu')
        results = evaluate(
            mock_model,
            mock_loader,
            device,
            use_amp=False,
            task_type='multiclass',
            num_classes=3,
        )

        assert 'sample_ids' in results
        assert results['sample_ids'] == ['sample_1', 'sample_2']
        assert 'predictions' in results
        assert 'labels' in results

    def test_evaluate_backward_compat(self, mock_model):
        """Test evaluation with old batch format (without sample_ids)."""
        # Old format: (features, labels, mask) - no sample_ids
        mock_batch = (
            torch.randn(2, 10, 512),
            torch.tensor([0, 1]),
            torch.ones(2, 10),
        )

        mock_loader = [mock_batch]

        device = torch.device('cpu')
        results = evaluate(
            mock_model,
            mock_loader,
            device,
            use_amp=False,
            task_type='binary',
            num_classes=2,
        )

        # Should work without sample_ids
        assert 'predictions' in results
        assert 'labels' in results
        assert 'sample_ids' not in results or results['sample_ids'] == []

    def test_evaluate_returns_all_metrics(self, mock_model):
        """Test that evaluate returns all expected metrics."""
        mock_batch = (
            torch.randn(4, 10, 512),
            torch.tensor([0, 1, 2, 0]),
            torch.ones(4, 10),
            ['s1', 's2', 's3', 's4'],
        )

        mock_loader = [mock_batch]

        device = torch.device('cpu')
        results = evaluate(
            mock_model,
            mock_loader,
            device,
            use_amp=False,
            task_type='multiclass',
            num_classes=3,
        )

        # Check all expected keys
        assert 'accuracy' in results
        assert 'balanced_accuracy' in results
        assert 'quadratic_kappa' in results
        assert 'auc' in results
        assert 'predictions' in results
        assert 'labels' in results
