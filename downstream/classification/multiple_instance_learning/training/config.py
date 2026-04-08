#!/usr/bin/env python3
"""
Configuration dataclasses for MIL training.

Simple Python dataclasses that can be easily extended to support
YAML/JSON loading or MLflow parameter logging.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from enum import Enum
import warnings

# Import for backward compatibility
from .mlflow_tracking import MLflowConfig
from .encoder_mapping import get_encoder_dim, parse_encoder_from_model_name


@dataclass
class TrackingConfig:
    """Unified tracking configuration for experiment tracking backends.

    This config supports multiple tracking backends (MLflow, W&B) through
    a single interface. Backend-specific settings are prefixed with the
    backend name (e.g., wandb_project).

    Example JSON config:
        {
            "tracking": {
                "backend": "wandb",
                "enabled": true,
                "experiment_name": "panda-grading",
                "wandb_project": "computational-pathology",
                "wandb_entity": "my-team",
                "git_tag": true
            }
        }
    """

    # General settings
    backend: str = "mlflow"  # "mlflow", "wandb", or "none"
    enabled: bool = True
    experiment_name: str = "mil-training"

    # MLflow-specific
    tracking_uri: Optional[str] = None  # From env MLFLOW_TRACKING_URI if None
    offline_fallback: bool = True
    offline_dir: str = "mlflow_offline"

    # W&B-specific
    wandb_project: Optional[str] = None  # Defaults to experiment_name
    wandb_entity: Optional[str] = None  # W&B team/user
    wandb_offline_dir: str = "wandb_offline"

    # Git versioning
    git_tag: bool = False  # Auto-tag experiments with git
    git_push: bool = False  # Auto-push tags to remote


class TaskType(Enum):
    """Classification task type."""
    BINARY = "binary"
    MULTICLASS = "multiclass"


@dataclass
class EncoderConfig:
    """Encoder configuration for metadata and validation.

    Used for logging encoder information and validating feature dimensions.
    The encoder name can be auto-parsed from model_name if not specified.

    Attributes:
        name: Encoder name (e.g., 'uni_v2', 'conch_v15')
        expected_dim: Auto-populated from ENCODER_DIM_MAPPING
    """
    name: str
    expected_dim: Optional[int] = None

    def __post_init__(self):
        # Auto-populate expected_dim from mapping
        if self.expected_dim is None:
            self.expected_dim = get_encoder_dim(self.name)


@dataclass
class DataConfig:
    """Configuration for data loading.

    Uses pre-extracted H5 features from features_dir.
    """
    labels_csv: str
    features_dir: str  # Directory containing H5 feature files
    dataset_name: Optional[str] = None  # Human-readable dataset ID (e.g., 'panda', 'camelyon16')
    # Split settings
    split_column: Optional[str] = None  # If set, use this column for splits
    split_dir: Optional[str] = None     # If set, load pre-generated JSON splits
    fold: Optional[int] = None          # Specific fold to load if split_dir is set
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42
    num_workers: int = 4
    hierarchical: bool = False
    group_column: str = 'case_id'
    fusion: str = 'early'  # 'early' or 'late'
    # Cross-validation settings
    num_folds: int = 5
    test_frac: float = 0.2
    cv_seed: int = 42

    def __post_init__(self):
        # Validate paths exist
        if not Path(self.labels_csv).exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")

        if not Path(self.features_dir).exists():
            raise FileNotFoundError(f"Features dir not found: {self.features_dir}")


@dataclass
class TrainConfig:
    """Configuration for training loop."""
    num_epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    feature_dropout: float = 0.1
    model_dropout: float = 0.25
    early_stopping_patience: int = 100  # Match original script
    min_epochs: int = 10  # Match original script
    max_grad_norm: float = 1.0
    use_amp: bool = True
    weighted_sampling: bool = True
    seed: int = 42
    # Task type and metric selection
    task_type: TaskType = TaskType.MULTICLASS
    early_stopping_metric: Literal["auto", "kappa", "balanced_accuracy", "auc", "f1_macro", "accuracy"] = "auto"

    def __post_init__(self):
        # Convert string to TaskType if needed (for JSON loading)
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    data: DataConfig
    train: TrainConfig
    model_name: str
    num_classes: int
    output_dir: str = 'experiments'
    run_name: Optional[str] = None  # Name for the run (individual training)
    num_heads: int = 1
    # Encoder configuration (for Trident integration)
    encoder: Optional[EncoderConfig] = None
    # New unified tracking config
    tracking: Optional[TrackingConfig] = None
    # Legacy mlflow config (deprecated, for backward compatibility)
    mlflow: Optional[MLflowConfig] = None

    def __post_init__(self):
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Generate run name if not provided
        if self.run_name is None:
            model_short = self.model_name.replace('.', '_')
            self.run_name = f"{model_short}"

        # Handle tracking configuration migration
        if self.tracking is None and self.mlflow is not None:
            # Migrate legacy mlflow config to new tracking config
            warnings.warn(
                "The 'mlflow' config field is deprecated. "
                "Please use 'tracking' with 'backend': 'mlflow' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.tracking = TrackingConfig(
                backend="mlflow",
                enabled=self.mlflow.enabled,
                experiment_name=self.mlflow.experiment_name,
                offline_fallback=self.mlflow.offline_fallback,
                tracking_uri=getattr(self.mlflow, "tracking_uri", None),
                offline_dir=getattr(self.mlflow, "offline_dir", "mlflow_offline"),
            )
        elif self.tracking is None:
            # Default tracking config
            self.tracking = TrackingConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary (useful for MLflow logging)."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    result[f"{key}.{sub_key}"] = sub_value
            else:
                result[key] = value
        return result

    def _parse_model_name(self) -> Dict[str, str]:
        """Parse model_name into components.

        Format: <mil_model>.<config>.<encoder>.<pretrain>
        Example: abmil.base.uni_v2.pc108-24k

        Returns:
            Dictionary with keys: mil_model, model_config, encoder, pretrain
        """
        parts = self.model_name.split('.')
        if len(parts) >= 4:
            return {
                'mil_model': parts[0],      # abmil, transmil, clam, etc.
                'model_config': parts[1],   # base, sb, etc.
                'encoder': parts[2],        # uni_v2, conch_v15, etc.
                'pretrain': parts[3],       # none, pc108-24k, etc.
            }
        return {'mil_model': self.model_name}

    def to_mlflow_params(self) -> Dict[str, Any]:
        """Flatten config to MLflow params dict (excludes mlflow config itself)."""
        params = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "num_heads": self.num_heads,
            "train.learning_rate": self.train.learning_rate,
            "train.num_epochs": self.train.num_epochs,
            "train.batch_size": self.train.batch_size,
            "train.weight_decay": self.train.weight_decay,
            "train.feature_dropout": self.train.feature_dropout,
            "train.model_dropout": self.train.model_dropout,
            "train.early_stopping_patience": self.train.early_stopping_patience,
            "train.min_epochs": self.train.min_epochs,
            "train.task_type": self.train.task_type.value,
            "train.early_stopping_metric": self.train.early_stopping_metric,
            "train.weighted_sampling": self.train.weighted_sampling,
            "data.hierarchical": self.data.hierarchical,
            "data.fusion": self.data.fusion,
            "data.num_workers": self.data.num_workers,
        }

        # Add parsed model components
        parsed = self._parse_model_name()
        params["mil_model"] = parsed.get('mil_model')
        params["pretrain"] = parsed.get('pretrain', 'none')

        # Add encoder metadata
        encoder_name = self._get_encoder_name()
        if encoder_name:
            params["encoder.name"] = encoder_name
            encoder_dim = get_encoder_dim(encoder_name)
            if encoder_dim:
                params["encoder.dim"] = encoder_dim

        # Add data traceability
        if self.data.dataset_name:
            params["dataset"] = self.data.dataset_name
        params["data.features_dir"] = self.data.features_dir
        params["data.labels_csv"] = self.data.labels_csv

        return params

    def _get_encoder_name(self) -> Optional[str]:
        """Get encoder name from config or model_name.

        Priority:
        1. Explicit encoder config
        2. Parsed from model_name
        """
        if self.encoder is not None:
            return self.encoder.name
        return parse_encoder_from_model_name(self.model_name)

    def save(self, path: str):
        """Save config to JSON file."""
        import json

        def serialize(obj):
            """Custom serializer for enums and other non-JSON types."""
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            return obj

        data = serialize(asdict(self))
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)

        # Load encoder config if present
        encoder_config = None
        if 'encoder' in data:
            encoder_config = EncoderConfig(**data['encoder'])

        # Load tracking config (new format)
        tracking_config = None
        if 'tracking' in data:
            tracking_config = TrackingConfig(**data['tracking'])

        # Load legacy mlflow config if present (for backward compatibility)
        mlflow_config = None
        if 'mlflow' in data and tracking_config is None:
            mlflow_config = MLflowConfig(**data['mlflow'])

        return cls(
            data=DataConfig(**data['data']),
            train=TrainConfig(**data['train']),
            model_name=data['model_name'],
            num_classes=data['num_classes'],
            output_dir=data.get('output_dir', 'experiments'),
            run_name=data.get('run_name'),
            num_heads=data.get('num_heads', 1),
            encoder=encoder_config,
            tracking=tracking_config,
            mlflow=mlflow_config,
        )
