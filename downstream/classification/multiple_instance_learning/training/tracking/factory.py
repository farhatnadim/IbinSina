"""Tracker factory with registry pattern."""

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

from .base import ExperimentTracker, TrackerConfig
from .mlflow_tracker import MLflowConfig, MLflowTracker
from .wandb_tracker import WandBConfig, WandBTracker

if TYPE_CHECKING:
    from ..config import ExperimentConfig, TrackingConfig

# Registry mapping backend names to (TrackerClass, ConfigClass) tuples
# None indicates tracking is disabled for that backend
TRACKER_REGISTRY: Dict[str, Tuple[Optional[Type[ExperimentTracker]], Optional[Type[TrackerConfig]]]] = {
    "mlflow": (MLflowTracker, MLflowConfig),
    "wandb": (WandBTracker, WandBConfig),
    "none": (None, None),
}


def create_tracker(config: "ExperimentConfig") -> Optional[ExperimentTracker]:
    """Create tracker based on configuration.

    This factory function creates the appropriate tracker based on the
    tracking configuration. It supports both the new TrackingConfig format
    and the legacy MLflowConfig format for backward compatibility.

    Args:
        config: ExperimentConfig with tracking settings

    Returns:
        ExperimentTracker instance or None if tracking is disabled

    Raises:
        ValueError: If backend is unknown
    """
    # Check for new tracking config
    if hasattr(config, "tracking") and config.tracking is not None:
        tracking_config = config.tracking

        if not tracking_config.enabled:
            return None

        backend = tracking_config.backend

        if backend not in TRACKER_REGISTRY:
            raise ValueError(
                f"Unknown tracker backend: '{backend}'. "
                f"Available backends: {list(TRACKER_REGISTRY.keys())}"
            )

        TrackerClass, ConfigClass = TRACKER_REGISTRY[backend]

        if TrackerClass is None:
            return None

        # Build backend-specific config from TrackingConfig
        if backend == "mlflow":
            backend_config = MLflowConfig(
                enabled=tracking_config.enabled,
                experiment_name=tracking_config.experiment_name,
                offline_fallback=tracking_config.offline_fallback,
                tracking_uri=tracking_config.tracking_uri,
                offline_dir=tracking_config.offline_dir,
            )
        elif backend == "wandb":
            backend_config = WandBConfig(
                enabled=tracking_config.enabled,
                experiment_name=tracking_config.experiment_name,
                offline_fallback=tracking_config.offline_fallback,
                project=tracking_config.wandb_project or tracking_config.experiment_name,
                entity=tracking_config.wandb_entity,
                offline_dir=tracking_config.wandb_offline_dir,
            )
        else:
            backend_config = ConfigClass(
                enabled=tracking_config.enabled,
                experiment_name=tracking_config.experiment_name,
                offline_fallback=tracking_config.offline_fallback,
            )

        return TrackerClass(backend_config)

    # Legacy support: check for old mlflow config
    if hasattr(config, "mlflow") and config.mlflow is not None:
        if config.mlflow.enabled:
            # Convert legacy MLflowConfig to new format
            legacy_config = MLflowConfig(
                enabled=config.mlflow.enabled,
                experiment_name=config.mlflow.experiment_name,
                offline_fallback=config.mlflow.offline_fallback,
                tracking_uri=getattr(config.mlflow, "tracking_uri", None),
                offline_dir=getattr(config.mlflow, "offline_dir", "mlflow_offline"),
            )
            return MLflowTracker(legacy_config)

    return None


def get_available_backends() -> list:
    """Get list of available tracker backends.

    Returns:
        List of backend names
    """
    return list(TRACKER_REGISTRY.keys())


def register_tracker(
    name: str,
    tracker_class: Type[ExperimentTracker],
    config_class: Type[TrackerConfig],
) -> None:
    """Register a custom tracker backend.

    This allows users to add their own tracking implementations.

    Args:
        name: Backend name (used in config)
        tracker_class: Class implementing ExperimentTracker
        config_class: Corresponding config dataclass

    Raises:
        ValueError: If name is already registered
    """
    if name in TRACKER_REGISTRY:
        raise ValueError(f"Tracker backend '{name}' is already registered")

    TRACKER_REGISTRY[name] = (tracker_class, config_class)
