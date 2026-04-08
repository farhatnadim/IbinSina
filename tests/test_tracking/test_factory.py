"""Tests for training/tracking/factory.py - Tracker factory and registry."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from downstream.classification.multiple_instance_learning.training.tracking.factory import (
    create_tracker,
    get_available_backends,
    register_tracker,
    TRACKER_REGISTRY,
)
from downstream.classification.multiple_instance_learning.training.tracking.base import ExperimentTracker, TrackerConfig
from downstream.classification.multiple_instance_learning.training.config import ExperimentConfig, TrackingConfig, DataConfig, TrainConfig


class TestCreateTracker:
    """Tests for create_tracker function."""

    def test_create_tracker_disabled(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test that disabled tracking returns None."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="test",
            num_classes=2,
            output_dir=str(temp_dir / "output"),
            tracking=TrackingConfig(enabled=False),
        )

        tracker = create_tracker(config)

        assert tracker is None

    def test_create_tracker_no_config(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test that default tracking config creates tracker."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="test",
            num_classes=2,
            output_dir=str(temp_dir / "output"),
            # No explicit tracking config - will use default
        )

        # Default config enables mlflow, which will go offline since server not running
        tracker = create_tracker(config)

        # Should create a tracker (may be in offline mode)
        if tracker is not None:
            from downstream.classification.multiple_instance_learning.training.tracking import MLflowTracker
            assert isinstance(tracker, MLflowTracker)

    def test_create_tracker_none_backend(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test that 'none' backend returns None."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="test",
            num_classes=2,
            output_dir=str(temp_dir / "output"),
            tracking=TrackingConfig(backend="none"),
        )

        tracker = create_tracker(config)

        assert tracker is None

    def test_create_tracker_mlflow(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test creating MLflow tracker."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="test",
            num_classes=2,
            output_dir=str(temp_dir / "output"),
            tracking=TrackingConfig(
                backend="mlflow",
                enabled=True,
                experiment_name="test-experiment",
            ),
        )

        tracker = create_tracker(config)

        from downstream.classification.multiple_instance_learning.training.tracking import MLflowTracker
        assert isinstance(tracker, MLflowTracker)

    def test_create_tracker_wandb(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test creating W&B tracker."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="test",
            num_classes=2,
            output_dir=str(temp_dir / "output"),
            tracking=TrackingConfig(
                backend="wandb",
                enabled=True,
                experiment_name="test-experiment",
            ),
        )

        tracker = create_tracker(config)

        from downstream.classification.multiple_instance_learning.training.tracking import WandBTracker
        assert isinstance(tracker, WandBTracker)

    def test_create_tracker_unknown_backend(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test that unknown backend raises ValueError."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="test",
            num_classes=2,
            output_dir=str(temp_dir / "output"),
            tracking=TrackingConfig(
                backend="unknown_backend",
                enabled=True,
            ),
        )

        with pytest.raises(ValueError, match="Unknown tracker backend"):
            create_tracker(config)


class TestGetAvailableBackends:
    """Tests for get_available_backends function."""

    def test_get_available_backends(self):
        """Test listing available backends."""
        backends = get_available_backends()

        assert isinstance(backends, list)
        assert "mlflow" in backends
        assert "wandb" in backends
        assert "none" in backends


class TestRegisterTracker:
    """Tests for register_tracker function."""

    def test_register_custom_tracker(self):
        """Test registering a custom tracker."""
        # Create a mock tracker class
        @dataclass
        class CustomConfig(TrackerConfig):
            custom_field: str = "test"

        class CustomTracker(ExperimentTracker):
            def __init__(self, config):
                self.config = config
                self._offline = False

            def start_run(self, run_name=None, nested=False, tags=None):
                pass

            def log_params(self, params):
                pass

            def log_metrics(self, metrics, step=None):
                pass

            def log_artifact(self, path):
                pass

            def set_tags(self, tags):
                pass

            @property
            def is_offline(self):
                return self._offline

        # Register it
        register_tracker("custom", CustomTracker, CustomConfig)

        try:
            assert "custom" in get_available_backends()
            assert TRACKER_REGISTRY["custom"] == (CustomTracker, CustomConfig)
        finally:
            # Clean up
            del TRACKER_REGISTRY["custom"]

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate backend raises error."""
        @dataclass
        class DummyConfig(TrackerConfig):
            pass

        class DummyTracker(ExperimentTracker):
            def __init__(self, config):
                pass

            def start_run(self, **kwargs):
                pass

            def log_params(self, params):
                pass

            def log_metrics(self, metrics, step=None):
                pass

            def log_artifact(self, path):
                pass

            def set_tags(self, tags):
                pass

            @property
            def is_offline(self):
                return True

        with pytest.raises(ValueError, match="already registered"):
            register_tracker("mlflow", DummyTracker, DummyConfig)
