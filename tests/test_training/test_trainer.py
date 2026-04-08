"""Tests for training/trainer.py - MILTrainer class."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from downstream.classification.multiple_instance_learning.training.trainer import MILTrainer
from downstream.classification.multiple_instance_learning.training.config import TrainConfig, TaskType


class TestMILTrainerInit:
    """Tests for MILTrainer initialization."""

    def test_trainer_init(self, mock_model, sample_mil_dataset):
        """Test basic trainer initialization."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=2,
            use_amp=False,
        )

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
        )

        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.config == config

    def test_trainer_init_with_checkpoint_dir(self, mock_model, sample_mil_dataset, temp_dir):
        """Test trainer with checkpoint directory."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(num_epochs=2, use_amp=False)
        checkpoint_dir = temp_dir / "checkpoints"

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
            checkpoint_dir=str(checkpoint_dir),
        )

        assert trainer.checkpoint_dir == checkpoint_dir
        assert checkpoint_dir.exists()

    def test_trainer_init_with_tracker(self, mock_model, sample_mil_dataset):
        """Test trainer with mock tracker."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(num_epochs=2, use_amp=False)

        mock_tracker = MagicMock()

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
            tracker=mock_tracker,
        )

        assert trainer.tracker is mock_tracker


class TestMILTrainerFit:
    """Tests for MILTrainer fit method."""

    def test_trainer_fit_one_epoch(self, mock_model, sample_mil_dataset):
        """Test that single epoch training completes."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=1,
            use_amp=False,
            early_stopping_patience=10,
        )

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
        )

        history = trainer.fit()

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1

    def test_trainer_early_stopping(self, sample_mil_dataset, temp_dir):
        """Test that early stopping triggers correctly."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        # Create a model that always produces same output (no improvement)
        # Must have at least one parameter for optimizer and proper gradient flow
        class ConstantModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Need at least one parameter for optimizer
                self.fc = nn.Linear(512, 2)

            def forward(self, x, loss_fn=None, label=None):
                batch_size = x.shape[0] if not isinstance(x, list) else len(x)
                # Use the fc layer to ensure gradients flow, but output constant-ish values
                pooled = x.mean(dim=1)  # [B, D]
                logits = self.fc(pooled) * 0.0  # Always zero logits
                loss = torch.tensor(1.0, requires_grad=True)
                if loss_fn and label is not None:
                    loss = loss_fn(logits, label)
                return {'logits': logits, 'loss': loss}, None

        model = ConstantModel()

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=10,
            use_amp=False,
            early_stopping_patience=2,
            min_epochs=1,
        )

        trainer = MILTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
            checkpoint_dir=str(temp_dir / "checkpoints"),
        )

        history = trainer.fit()

        # Should stop before 10 epochs due to patience
        assert len(history['train_loss']) < 10

    def test_trainer_with_tracker_logs_metrics(self, mock_model, sample_mil_dataset):
        """Test that trainer logs metrics to tracker."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=1,
            use_amp=False,
        )

        mock_tracker = MagicMock()

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
            tracker=mock_tracker,
        )

        trainer.fit()

        # Verify tracker.log_metrics was called
        assert mock_tracker.log_metrics.called

    def test_trainer_without_tracker(self, mock_model, sample_mil_dataset):
        """Test trainer works with tracker=None."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=1,
            use_amp=False,
        )

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
            tracker=None,
        )

        history = trainer.fit()

        assert 'train_loss' in history


class TestMILTrainerCheckpoint:
    """Tests for checkpoint save/load functionality."""

    def test_trainer_checkpoint_save_load(self, mock_model, sample_mil_dataset, temp_dir):
        """Test checkpoint round-trip."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(num_epochs=2, use_amp=False)

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
            checkpoint_dir=str(temp_dir),
        )

        # Run one epoch
        trainer.fit()

        # Save checkpoint manually
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Create a new trainer and load checkpoint
        new_trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
        )

        new_trainer.load_checkpoint(str(checkpoint_path), weights_only=False)

        assert new_trainer.best_val_metric == trainer.best_val_metric

    def test_trainer_load_best_model(self, mock_model, sample_mil_dataset, temp_dir):
        """Test loading best model from checkpoint."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(num_epochs=2, use_amp=False)

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
            checkpoint_dir=str(temp_dir),
        )

        trainer.fit()

        # Load best model
        trainer.load_best_model()

        # Should not raise


class TestMILTrainerMetricResolution:
    """Tests for early stopping metric resolution."""

    def test_resolve_metric_binary_auto(self, mock_model, sample_mil_dataset):
        """Test auto metric resolution for binary task."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=1,
            use_amp=False,
            task_type=TaskType.BINARY,
            early_stopping_metric="auto",
        )

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
        )

        # For binary, auto should resolve to AUC
        assert trainer._early_stopping_metric_name == "auc"

    def test_resolve_metric_multiclass_auto(self, mock_model, sample_mil_dataset):
        """Test auto metric resolution for multiclass task."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=1,
            use_amp=False,
            task_type=TaskType.MULTICLASS,
            early_stopping_metric="auto",
        )

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
        )

        # For multiclass, auto should resolve to kappa
        assert trainer._early_stopping_metric_name == "kappa"

    def test_resolve_metric_explicit(self, mock_model, sample_mil_dataset):
        """Test explicit metric specification."""
        from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader

        train_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        val_loader, _ = create_dataloader(
            sample_mil_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        config = TrainConfig(
            num_epochs=1,
            use_amp=False,
            early_stopping_metric="balanced_accuracy",
        )

        trainer = MILTrainer(
            model=mock_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device('cpu'),
        )

        assert trainer._early_stopping_metric_name == "balanced_accuracy"
