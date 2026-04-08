"""
Modular training infrastructure for MIL-Lab.

Usage:
    from training import ExperimentConfig, DataConfig, TrainConfig, MILTrainer, evaluate

    config = ExperimentConfig(
        data=DataConfig(labels_csv='...', features_dir='...'),
        train=TrainConfig(num_epochs=20),
        model_name='abmil.base.uni_v2.pc108-24k',
        num_classes=6,
    )

    trainer = MILTrainer(model, train_loader, val_loader, config.train, device)
    history = trainer.fit()
    results = evaluate(model, test_loader, device)
"""

from .config import (
    DataConfig,
    TrainConfig,
    ExperimentConfig,
    TrackingConfig,
    TaskType,
)

# New tracking module (recommended)
from .tracking import (
    ExperimentTracker,
    TrackerConfig,
    MLflowConfig,
    MLflowTracker,
    WandBConfig,
    WandBTracker,
    create_tracker,
    get_available_backends,
    register_tracker,
    # Git versioning
    GitVersioningError,
    get_git_info,
    has_uncommitted_changes,
    is_git_repo,
    ensure_clean_repo,
    create_experiment_tag,
    delete_experiment_tag,
    list_experiment_tags,
)

from .trainer import MILTrainer

from .evaluator import (
    evaluate,
    calculate_metrics,
    print_evaluation_results,
)

from .utils import apply_grouping, save_predictions, save_results_summary


__all__ = [
    # Config
    'DataConfig',
    'TrainConfig',
    'ExperimentConfig',
    'TrackingConfig',
    'TaskType',
    # Tracking (new)
    'ExperimentTracker',
    'TrackerConfig',
    'MLflowConfig',
    'MLflowTracker',
    'WandBConfig',
    'WandBTracker',
    'create_tracker',
    'get_available_backends',
    'register_tracker',
    # Git versioning
    'GitVersioningError',
    'get_git_info',
    'has_uncommitted_changes',
    'is_git_repo',
    'ensure_clean_repo',
    'create_experiment_tag',
    'delete_experiment_tag',
    'list_experiment_tags',
    # Training
    'MILTrainer',
    # Evaluation
    'evaluate',
    'calculate_metrics',
    'print_evaluation_results',
    # Utils
    'apply_grouping',
    'save_predictions',
    'save_results_summary',
]
