"""MLflow tracking implementation with offline fallback."""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExperimentTracker, TrackerConfig


@dataclass
class MLflowConfig(TrackerConfig):
    """Configuration for MLflow tracking."""

    tracking_uri: Optional[str] = None  # From env MLFLOW_TRACKING_URI if None
    offline_dir: str = "mlflow_offline"


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker with graceful degradation to offline mode."""

    def __init__(self, config: MLflowConfig):
        self.config = config
        self._mlflow = None
        self._run = None
        self._offline_mode = False
        self._offline_data: Dict[str, Any] = {}
        self._run_name: Optional[str] = None
        self._nested_level = 0
        self._init_mlflow()

    def _init_mlflow(self):
        """Try to import and connect to MLflow."""
        if not self.config.enabled:
            self._offline_mode = True
            return

        try:
            import mlflow

            self._mlflow = mlflow

            uri = self.config.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)

            # Test connection by setting experiment
            mlflow.set_experiment(self.config.experiment_name)
            print(f"MLflow connected: {uri or 'local'}")

        except Exception as e:
            if self.config.offline_fallback:
                print(f"MLflow unavailable ({e}), using offline mode")
                self._offline_mode = True
            else:
                raise

    @property
    def is_offline(self) -> bool:
        """Check if tracker is in offline mode."""
        return self._offline_mode

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for MLflow run.

        Args:
            run_name: Optional name for the run
            nested: If True, create a nested run under the current parent
            tags: Optional tags to attach to the run
        """
        if self._offline_mode:
            # Offline mode - just track data locally
            self._run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if nested:
                self._nested_level += 1
                self._offline_data.setdefault("nested_runs", []).append(
                    {
                        "name": self._run_name,
                        "tags": tags or {},
                        "params": {},
                        "metrics": [],
                        "artifacts": [],
                    }
                )
            else:
                self._offline_data = {
                    "run_name": self._run_name,
                    "tags": tags or {},
                    "params": {},
                    "metrics": [],
                    "artifacts": [],
                    "nested_runs": [],
                    "start_time": datetime.now().isoformat(),
                }
            try:
                yield self
            finally:
                if nested:
                    self._nested_level -= 1
                else:
                    self._offline_data["end_time"] = datetime.now().isoformat()
                    self._save_offline()
        else:
            # Online MLflow mode
            try:
                with self._mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
                    self._run = run
                    yield self
            finally:
                self._run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.

        Args:
            params: Dictionary of parameter names to values
        """
        if self._offline_mode:
            target = self._get_offline_target()
            target["params"].update(params)
        else:
            # MLflow has issues with None values and certain types
            clean_params = {}
            for k, v in params.items():
                if v is None:
                    clean_params[k] = "None"
                elif isinstance(v, (list, dict)):
                    clean_params[k] = json.dumps(v)
                else:
                    clean_params[k] = v
            self._mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        if self._offline_mode:
            target = self._get_offline_target()
            target["metrics"].append(
                {
                    "step": step,
                    "values": metrics,
                }
            )
        else:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        """Log artifact file.

        Args:
            path: Path to the artifact file
        """
        path = Path(path)
        if self._offline_mode:
            target = self._get_offline_target()
            target["artifacts"].append(str(path))
        else:
            if path.exists():
                self._mlflow.log_artifact(str(path))
            else:
                print(f"Warning: Artifact not found: {path}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags on the current run.

        Args:
            tags: Dictionary of tag names to values
        """
        if self._offline_mode:
            target = self._get_offline_target()
            target["tags"].update(tags)
        else:
            self._mlflow.set_tags(tags)

    def _get_offline_target(self) -> Dict[str, Any]:
        """Get the appropriate target dict for offline logging (parent or nested run)."""
        if self._nested_level > 0 and self._offline_data.get("nested_runs"):
            return self._offline_data["nested_runs"][-1]
        return self._offline_data

    def _save_offline(self):
        """Save offline data to JSON file."""
        if not self._offline_data:
            return

        offline_dir = Path(self.config.offline_dir)
        offline_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self._offline_data.get("run_name", "unnamed")
        filename = f"run_{run_name}_{timestamp}.json"

        filepath = offline_dir / filename
        with open(filepath, "w") as f:
            json.dump(self._offline_data, f, indent=2, default=str)

        print(f"Offline tracking data saved to: {filepath}")
