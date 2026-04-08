"""Pluggable experiment tracking system.

This module provides a provider-agnostic interface for experiment tracking
that supports multiple backends (MLflow, Weights & Biases) and includes
git tagging for reproducibility.

Example usage:
    from downstream.classification.multiple_instance_learning.training.tracking import (
        create_tracker,
        get_git_info,
    )

    # Create tracker from config
    tracker = create_tracker(config)

    if tracker:
        with tracker.start_run(run_name="my_experiment"):
            tracker.log_params(get_git_info())
            tracker.log_params({"learning_rate": 1e-4})

            # Training loop
            for epoch in range(num_epochs):
                # ... train ...
                tracker.log_metrics({"loss": loss}, step=epoch)

            tracker.log_artifact(model_path)
"""

# Base classes
from .base import ExperimentTracker, TrackerConfig

# Concrete implementations
from .mlflow_tracker import MLflowConfig, MLflowTracker
from .wandb_tracker import WandBConfig, WandBTracker

# Factory
from .factory import create_tracker, get_available_backends, register_tracker

# Git versioning
from .git_versioning import (
    GitVersioningError,
    create_experiment_tag,
    delete_experiment_tag,
    ensure_clean_repo,
    get_git_info,
    has_uncommitted_changes,
    is_git_repo,
    list_experiment_tags,
)

__all__ = [
    # Base classes
    "ExperimentTracker",
    "TrackerConfig",
    # MLflow
    "MLflowConfig",
    "MLflowTracker",
    # W&B
    "WandBConfig",
    "WandBTracker",
    # Factory
    "create_tracker",
    "get_available_backends",
    "register_tracker",
    # Git versioning
    "GitVersioningError",
    "get_git_info",
    "has_uncommitted_changes",
    "is_git_repo",
    "ensure_clean_repo",
    "create_experiment_tag",
    "delete_experiment_tag",
    "list_experiment_tags",
]
