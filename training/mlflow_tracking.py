"""MLflow tracking wrapper - DEPRECATED.

This module is deprecated. Please use `training.tracking` instead.

For backward compatibility, this module re-exports the MLflow tracker
and create_tracker function from the new tracking module.

Migration guide:
    # Old (deprecated)
    from training.mlflow_tracking import MLflowTracker, create_tracker

    # New (recommended)
    from training.tracking import MLflowTracker, create_tracker
    # Or for backend-agnostic tracking:
    from training.tracking import create_tracker  # Works with any backend
"""

import warnings
from dataclasses import dataclass
from typing import Optional

# Re-export from new location for backward compatibility
from .tracking.mlflow_tracker import MLflowConfig, MLflowTracker
from .tracking.factory import create_tracker as _create_tracker


def create_tracker(config: "ExperimentConfig") -> Optional[MLflowTracker]:
    """Create an experiment tracker from config.

    DEPRECATED: This function is deprecated. Use `training.tracking.create_tracker` instead.

    Args:
        config: ExperimentConfig with mlflow/tracking settings

    Returns:
        MLflowTracker instance or None if tracking is disabled
    """
    warnings.warn(
        "create_tracker from training.mlflow_tracking is deprecated. "
        "Use training.tracking.create_tracker instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _create_tracker(config)


__all__ = ["MLflowConfig", "MLflowTracker", "create_tracker"]
