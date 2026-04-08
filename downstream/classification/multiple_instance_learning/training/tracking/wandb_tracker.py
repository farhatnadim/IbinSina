"""Weights & Biases tracking implementation with offline fallback."""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExperimentTracker, TrackerConfig


@dataclass
class WandBConfig(TrackerConfig):
    """Configuration for Weights & Biases tracking."""

    project: str = "mil-training"
    entity: Optional[str] = None  # W&B team/user
    offline_dir: str = "wandb_offline"


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker with offline fallback."""

    def __init__(self, config: WandBConfig):
        self.config = config
        self._wandb = None
        self._run = None
        self._offline_mode = False
        self._offline_data: Dict[str, Any] = {}
        self._run_name: Optional[str] = None
        self._nested_runs: list = []  # Stack for nested run context
        self._init_wandb()

    def _init_wandb(self):
        """Try to import and connect to W&B."""
        if not self.config.enabled:
            self._offline_mode = True
            return

        try:
            import wandb

            self._wandb = wandb

            # Check if W&B is logged in
            if wandb.api.api_key is None:
                if self.config.offline_fallback:
                    print("W&B not logged in, using offline mode")
                    self._offline_mode = True
                else:
                    raise RuntimeError("W&B not logged in and offline_fallback is False")
            else:
                print(f"W&B ready: project={self.config.project}, entity={self.config.entity or 'default'}")

        except ImportError as e:
            if self.config.offline_fallback:
                print(f"W&B not installed ({e}), using offline mode")
                self._offline_mode = True
            else:
                raise
        except Exception as e:
            if self.config.offline_fallback:
                print(f"W&B unavailable ({e}), using offline mode")
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
        """Context manager for W&B run.

        Args:
            run_name: Optional name for the run
            nested: If True, create a nested run (W&B uses groups for this)
            tags: Optional tags to attach to the run

        Note:
            W&B doesn't support true nested runs. For CV workflows, we use
            run groups to associate fold runs with a parent CV run.
        """
        if self._offline_mode:
            # Offline mode - just track data locally
            self._run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if nested:
                # Store parent run data and create nested context
                self._nested_runs.append(self._offline_data.copy() if self._offline_data else {})
                self._offline_data = {
                    "run_name": self._run_name,
                    "tags": tags or {},
                    "params": {},
                    "metrics": [],
                    "artifacts": [],
                    "nested": True,
                    "start_time": datetime.now().isoformat(),
                }
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
                self._offline_data["end_time"] = datetime.now().isoformat()
                if nested:
                    # Save nested run and restore parent
                    nested_data = self._offline_data.copy()
                    self._offline_data = self._nested_runs.pop() if self._nested_runs else {}
                    if self._offline_data:
                        self._offline_data.setdefault("nested_runs", []).append(nested_data)
                    else:
                        # No parent, save directly
                        self._save_offline(nested_data)
                else:
                    self._save_offline()
        else:
            # Online W&B mode
            try:
                # Convert tags dict to list format for W&B
                tag_list = list(tags.keys()) if tags else None

                # For nested runs, use the parent run name as the group
                group = None
                if nested and self._run:
                    group = self._run.name

                # Initialize W&B run
                run = self._wandb.init(
                    project=self.config.project,
                    entity=self.config.entity,
                    name=run_name,
                    tags=tag_list,
                    group=group,
                    reinit=True,  # Allow multiple runs in same process
                    config=tags,  # Store tags in config for reference
                )

                previous_run = self._run
                self._run = run

                yield self

            finally:
                if self._run:
                    self._run.finish()
                self._run = previous_run if nested else None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters (config in W&B terminology).

        Args:
            params: Dictionary of parameter names to values
        """
        if self._offline_mode:
            self._offline_data["params"].update(params)
        else:
            if self._run:
                self._run.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        if self._offline_mode:
            self._offline_data["metrics"].append(
                {
                    "step": step,
                    "values": metrics,
                }
            )
        else:
            if self._run:
                log_dict = metrics.copy()
                if step is not None:
                    log_dict["epoch"] = step
                self._run.log(log_dict, step=step)

    def log_artifact(self, path: Path) -> None:
        """Log artifact file.

        Args:
            path: Path to the artifact file
        """
        path = Path(path)
        if self._offline_mode:
            self._offline_data["artifacts"].append(str(path))
        else:
            if self._run and path.exists():
                # W&B artifacts - save file directly
                self._run.save(str(path), policy="now")
            elif not path.exists():
                print(f"Warning: Artifact not found: {path}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set run tags.

        Args:
            tags: Dictionary of tag names to values

        Note:
            W&B tags are string lists, not key-value pairs.
            We convert by using the keys as tags and storing
            the full dict in the run config.
        """
        if self._offline_mode:
            self._offline_data["tags"].update(tags)
        else:
            if self._run:
                # Add tag keys to the run's tag list
                current_tags = list(self._run.tags) if self._run.tags else []
                new_tags = list(set(current_tags + list(tags.keys())))
                self._run.tags = tuple(new_tags)
                # Also store full dict in config for reference
                self._run.config.update({"_tags": tags})

    def _save_offline(self, data: Optional[Dict[str, Any]] = None):
        """Save offline data to JSON file."""
        data = data or self._offline_data
        if not data:
            return

        offline_dir = Path(self.config.offline_dir)
        offline_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = data.get("run_name", "unnamed")
        filename = f"run_{run_name}_{timestamp}.json"

        filepath = offline_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Offline W&B tracking data saved to: {filepath}")
