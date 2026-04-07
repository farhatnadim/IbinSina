"""Tests for tracking implementations - MLflow and W&B trackers."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.tracking.mlflow_tracker import MLflowTracker, MLflowConfig
from training.tracking.wandb_tracker import WandBTracker, WandBConfig


class TestMLflowTracker:
    """Tests for MLflowTracker class."""

    def test_mlflow_tracker_offline_fallback(self, temp_dir):
        """Test that tracker falls back to offline mode when MLflow unavailable."""
        config = MLflowConfig(
            enabled=True,
            experiment_name="test-experiment",
            offline_fallback=True,
            offline_dir=str(temp_dir / "mlflow_offline"),
        )

        # Should not raise even if MLflow isn't available/connected
        tracker = MLflowTracker(config)

        # Will be offline since no server running
        assert tracker.is_offline == True or tracker._mlflow is not None

    def test_mlflow_tracker_disabled(self):
        """Test disabled tracker goes to offline mode."""
        config = MLflowConfig(enabled=False)

        tracker = MLflowTracker(config)

        assert tracker.is_offline == True

    def test_mlflow_tracker_log_metrics_offline(self, temp_dir):
        """Test logging metrics in offline mode."""
        config = MLflowConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "mlflow_offline"),
        )

        tracker = MLflowTracker(config)
        # Force offline mode for testing
        tracker._offline_mode = True
        tracker._offline_data = {
            "run_name": "test",
            "params": {},
            "metrics": [],
            "artifacts": [],
            "tags": {},
        }

        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=0)

        assert len(tracker._offline_data["metrics"]) == 1
        assert tracker._offline_data["metrics"][0]["values"]["loss"] == 0.5
        assert tracker._offline_data["metrics"][0]["step"] == 0

    def test_mlflow_tracker_log_params_offline(self, temp_dir):
        """Test logging params in offline mode."""
        config = MLflowConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "mlflow_offline"),
        )

        tracker = MLflowTracker(config)
        tracker._offline_mode = True
        tracker._offline_data = {
            "run_name": "test",
            "params": {},
            "metrics": [],
            "artifacts": [],
            "tags": {},
        }

        tracker.log_params({"learning_rate": 1e-4, "epochs": 10})

        assert tracker._offline_data["params"]["learning_rate"] == 1e-4
        assert tracker._offline_data["params"]["epochs"] == 10

    def test_mlflow_tracker_context_manager_offline(self, temp_dir):
        """Test start_run context manager in offline mode."""
        config = MLflowConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "mlflow_offline"),
        )

        tracker = MLflowTracker(config)
        tracker._offline_mode = True

        with tracker.start_run(run_name="test_run") as t:
            assert t is tracker
            t.log_params({"param1": "value1"})
            t.log_metrics({"metric1": 0.5}, step=0)

        # Verify offline data was saved
        offline_dir = temp_dir / "mlflow_offline"
        assert offline_dir.exists()
        json_files = list(offline_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_mlflow_tracker_nested_runs_offline(self, temp_dir):
        """Test nested runs in offline mode."""
        config = MLflowConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "mlflow_offline"),
        )

        tracker = MLflowTracker(config)
        tracker._offline_mode = True

        with tracker.start_run(run_name="parent_run") as parent:
            parent.log_params({"parent_param": "value"})

            with tracker.start_run(run_name="nested_run", nested=True) as nested:
                nested.log_params({"nested_param": "nested_value"})

    def test_mlflow_tracker_log_artifact_offline(self, temp_dir):
        """Test logging artifacts in offline mode."""
        config = MLflowConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "mlflow_offline"),
        )

        tracker = MLflowTracker(config)
        tracker._offline_mode = True
        tracker._offline_data = {
            "run_name": "test",
            "params": {},
            "metrics": [],
            "artifacts": [],
            "tags": {},
        }

        artifact_path = temp_dir / "test_artifact.txt"
        artifact_path.write_text("test content")

        tracker.log_artifact(artifact_path)

        assert str(artifact_path) in tracker._offline_data["artifacts"]


class TestWandBTracker:
    """Tests for WandBTracker class."""

    def test_wandb_tracker_offline_fallback(self, temp_dir):
        """Test that tracker falls back to offline mode when W&B unavailable."""
        config = WandBConfig(
            enabled=True,
            project="test-project",
            offline_fallback=True,
            offline_dir=str(temp_dir / "wandb_offline"),
        )

        # Should not raise even if W&B isn't available/logged in
        tracker = WandBTracker(config)

        # Will be offline since not logged in
        assert tracker.is_offline == True or tracker._wandb is not None

    def test_wandb_tracker_disabled(self):
        """Test disabled tracker goes to offline mode."""
        config = WandBConfig(enabled=False)

        tracker = WandBTracker(config)

        assert tracker.is_offline == True

    def test_wandb_tracker_log_metrics_offline(self, temp_dir):
        """Test logging metrics in offline mode."""
        config = WandBConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "wandb_offline"),
        )

        tracker = WandBTracker(config)
        tracker._offline_mode = True
        tracker._offline_data = {
            "run_name": "test",
            "params": {},
            "metrics": [],
            "artifacts": [],
            "tags": {},
        }

        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=0)

        assert len(tracker._offline_data["metrics"]) == 1
        assert tracker._offline_data["metrics"][0]["values"]["loss"] == 0.5

    def test_wandb_tracker_log_params_offline(self, temp_dir):
        """Test logging params in offline mode."""
        config = WandBConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "wandb_offline"),
        )

        tracker = WandBTracker(config)
        tracker._offline_mode = True
        tracker._offline_data = {
            "run_name": "test",
            "params": {},
            "metrics": [],
            "artifacts": [],
            "tags": {},
        }

        tracker.log_params({"learning_rate": 1e-4})

        assert tracker._offline_data["params"]["learning_rate"] == 1e-4

    def test_wandb_tracker_context_manager_offline(self, temp_dir):
        """Test start_run context manager in offline mode."""
        config = WandBConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "wandb_offline"),
        )

        tracker = WandBTracker(config)
        tracker._offline_mode = True

        with tracker.start_run(run_name="test_run") as t:
            assert t is tracker
            t.log_params({"param1": "value1"})

        # Verify offline data was saved
        offline_dir = temp_dir / "wandb_offline"
        assert offline_dir.exists()

    def test_wandb_tracker_set_tags_offline(self, temp_dir):
        """Test setting tags in offline mode."""
        config = WandBConfig(
            enabled=True,
            offline_fallback=True,
            offline_dir=str(temp_dir / "wandb_offline"),
        )

        tracker = WandBTracker(config)
        tracker._offline_mode = True
        tracker._offline_data = {
            "run_name": "test",
            "params": {},
            "metrics": [],
            "artifacts": [],
            "tags": {},
        }

        tracker.set_tags({"experiment": "test", "version": "v1"})

        assert tracker._offline_data["tags"]["experiment"] == "test"
        assert tracker._offline_data["tags"]["version"] == "v1"


class TestTrackerIsOfflineProperty:
    """Tests for is_offline property across trackers."""

    def test_mlflow_is_offline_property(self):
        """Test is_offline property for MLflow."""
        config = MLflowConfig(enabled=False)
        tracker = MLflowTracker(config)

        assert tracker.is_offline == True

    def test_wandb_is_offline_property(self):
        """Test is_offline property for W&B."""
        config = WandBConfig(enabled=False)
        tracker = WandBTracker(config)

        assert tracker.is_offline == True
