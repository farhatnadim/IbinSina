"""Shared utilities for training scripts."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


def apply_grouping(dataset, config):
    """
    Apply grouping/fusion based on config.

    Args:
        dataset: MILDataset instance
        config: ExperimentConfig with data.group_column, data.fusion, data.hierarchical

    Returns:
        Modified dataset (grouped or concatenated) or original if no grouping needed
    """
    if not config.data.group_column:
        return dataset

    if config.data.group_column not in dataset.df.columns:
        raise ValueError(
            f"group_column '{config.data.group_column}' not found in dataset. "
            f"Available columns: {list(dataset.df.columns)}. "
            f"Set group_column to null in config if grouping is not needed."
        )

    # Check if we actually have multiple slides for some cases
    has_multi = (dataset.df.groupby(config.data.group_column).size() > 1).any()

    if has_multi or config.data.hierarchical:
        if config.data.fusion == 'late':
            print(f"Using LATE FUSION (Hierarchical) grouping by: {config.data.group_column}")
            return dataset.group_by(config.data.group_column)
        else:
            print(f"Using EARLY FUSION (Concatenated) grouping by: {config.data.group_column}")
            return dataset.concat_by(config.data.group_column)
    else:
        print(f"No multi-slide cases found for {config.data.group_column}, using slide-level loading.")
        return dataset


def save_predictions(path: Path, labels: List[int], predictions: List[int], class_labels: List[str]):
    """
    Save predictions to npz file.

    Args:
        path: Path to save predictions
        labels: True labels
        predictions: Predicted labels
        class_labels: List of class label names
    """
    np.savez(
        path,
        labels=labels,
        predictions=predictions,
        class_labels=class_labels,
    )


def save_results_summary(path: Path, summary: Dict[str, Any]):
    """
    Save results summary to JSON file.

    Args:
        path: Path to save results
        summary: Dictionary with results summary
    """
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
