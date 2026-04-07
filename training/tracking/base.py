"""Abstract base class for experiment tracking."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrackerConfig:
    """Base configuration for all trackers."""

    enabled: bool = True
    experiment_name: str = "mil-training"
    offline_fallback: bool = True


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking.

    All concrete tracker implementations must inherit from this class
    and implement all abstract methods.
    """

    @abstractmethod
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for run lifecycle.

        Args:
            run_name: Optional name for the run
            nested: If True, create a nested run under the current parent
            tags: Optional tags to attach to the run

        Yields:
            Self for method chaining
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameter names to values
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics (optionally per-epoch with step).

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        pass

    @abstractmethod
    def log_artifact(self, path: Path) -> None:
        """Log artifact file.

        Args:
            path: Path to the artifact file
        """
        pass

    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set run tags.

        Args:
            tags: Dictionary of tag names to values
        """
        pass

    @property
    @abstractmethod
    def is_offline(self) -> bool:
        """Check if tracker is in offline mode."""
        pass
