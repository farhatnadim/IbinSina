"""Tests for training/config.py - Configuration dataclasses."""

import pytest
import json
import tempfile
from pathlib import Path

from downstream.classification.multiple_instance_learning.training.config import (
    DataConfig,
    TrainConfig,
    ExperimentConfig,
    TrackingConfig,
    TaskType,
)


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_train_config_defaults(self):
        """Test default values."""
        config = TrainConfig()

        assert config.num_epochs == 20
        assert config.batch_size == 1
        assert config.learning_rate == 1e-4
        assert config.task_type == TaskType.MULTICLASS
        assert config.early_stopping_metric == "auto"

    def test_train_config_task_type_enum(self):
        """Test string to TaskType enum conversion."""
        config = TrainConfig(task_type="binary")

        assert config.task_type == TaskType.BINARY

    def test_train_config_task_type_enum_multiclass(self):
        """Test multiclass task type conversion."""
        config = TrainConfig(task_type="multiclass")

        assert config.task_type == TaskType.MULTICLASS

    def test_train_config_custom_values(self):
        """Test custom configuration values."""
        config = TrainConfig(
            num_epochs=50,
            learning_rate=1e-3,
            early_stopping_patience=20,
        )

        assert config.num_epochs == 50
        assert config.learning_rate == 1e-3
        assert config.early_stopping_patience == 20


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_data_config_validation(self, sample_labels_csv, sample_features_dir):
        """Test that paths are validated on init."""
        config = DataConfig(
            labels_csv=str(sample_labels_csv),
            features_dir=str(sample_features_dir),
        )

        assert config.labels_csv == str(sample_labels_csv)
        assert config.features_dir == str(sample_features_dir)

    def test_data_config_missing_labels_csv(self, sample_features_dir):
        """Test error when labels CSV doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Labels CSV not found"):
            DataConfig(
                labels_csv="/nonexistent/path/labels.csv",
                features_dir=str(sample_features_dir),
            )

    def test_data_config_missing_features_dir(self, sample_labels_csv):
        """Test error when features dir doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Features dir not found"):
            DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir="/nonexistent/path/features",
            )

    def test_data_config_defaults(self, sample_labels_csv, sample_features_dir):
        """Test default values."""
        config = DataConfig(
            labels_csv=str(sample_labels_csv),
            features_dir=str(sample_features_dir),
        )

        assert config.train_frac == 0.7
        assert config.val_frac == 0.15
        assert config.fusion == 'early'
        assert config.num_folds == 5


class TestTrackingConfig:
    """Tests for TrackingConfig dataclass."""

    def test_tracking_config_defaults(self):
        """Test default values."""
        config = TrackingConfig()

        assert config.backend == "mlflow"
        assert config.enabled == True
        assert config.experiment_name == "mil-training"
        assert config.offline_fallback == True

    def test_tracking_config_wandb_backend(self):
        """Test W&B backend configuration."""
        config = TrackingConfig(
            backend="wandb",
            wandb_project="my-project",
            wandb_entity="my-team",
        )

        assert config.backend == "wandb"
        assert config.wandb_project == "my-project"
        assert config.wandb_entity == "my-team"

    def test_tracking_config_disabled(self):
        """Test disabled tracking."""
        config = TrackingConfig(enabled=False)

        assert config.enabled == False

    def test_tracking_config_none_backend(self):
        """Test 'none' backend for no tracking."""
        config = TrackingConfig(backend="none")

        assert config.backend == "none"


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_experiment_config_load_save(self, sample_experiment_config, temp_dir):
        """Test JSON round-trip serialization."""
        # Write config to JSON
        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(sample_experiment_config, f, indent=2)

        # Load it back
        loaded = ExperimentConfig.load(str(config_path))

        assert loaded.model_name == sample_experiment_config["model_name"]
        assert loaded.num_classes == sample_experiment_config["num_classes"]
        assert loaded.train.num_epochs == sample_experiment_config["train"]["num_epochs"]

    def test_experiment_config_save(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test saving config to JSON."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(num_epochs=10),
            model_name="test_model",
            num_classes=3,
            output_dir=str(temp_dir / "output"),
        )

        save_path = temp_dir / "saved_config.json"
        config.save(str(save_path))

        assert save_path.exists()

        # Verify it can be read back
        with open(save_path) as f:
            data = json.load(f)

        assert data["model_name"] == "test_model"
        assert data["num_classes"] == 3

    def test_experiment_config_to_mlflow_params(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test flattening config for MLflow."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(
                num_epochs=10,
                learning_rate=1e-3,
            ),
            model_name="abmil.base.uni_v2.none",
            num_classes=5,
            output_dir=str(temp_dir / "output"),
        )

        params = config.to_mlflow_params()

        assert params["model_name"] == "abmil.base.uni_v2.none"
        assert params["num_classes"] == 5
        assert params["train.learning_rate"] == 1e-3
        assert params["train.num_epochs"] == 10
        assert "train.task_type" in params

    def test_experiment_config_auto_run_name(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test automatic run name generation."""
        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="abmil.base.uni_v2.none",
            num_classes=3,
            output_dir=str(temp_dir / "output"),
        )

        assert config.run_name == "abmil_base_uni_v2_none"

    def test_experiment_config_creates_output_dir(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test that output directory is created."""
        output_dir = temp_dir / "new_output"
        assert not output_dir.exists()

        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=str(sample_labels_csv),
                features_dir=str(sample_features_dir),
            ),
            train=TrainConfig(),
            model_name="test",
            num_classes=2,
            output_dir=str(output_dir),
        )

        assert output_dir.exists()

    def test_experiment_config_with_tracking(self, sample_labels_csv, sample_features_dir, temp_dir):
        """Test config with tracking settings."""
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
                experiment_name="test-exp",
            ),
        )

        assert config.tracking.backend == "wandb"
        assert config.tracking.experiment_name == "test-exp"


class TestTaskTypeEnum:
    """Tests for TaskType enum."""

    def test_task_type_values(self):
        """Test enum values."""
        assert TaskType.BINARY.value == "binary"
        assert TaskType.MULTICLASS.value == "multiclass"

    def test_task_type_from_string(self):
        """Test creating enum from string."""
        assert TaskType("binary") == TaskType.BINARY
        assert TaskType("multiclass") == TaskType.MULTICLASS
