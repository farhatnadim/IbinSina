#!/usr/bin/env python3
"""
MIL Training Entry Point

A clean, modular training script that uses the new data_loading and training modules.

Usage:
    python train_mil.py  # Uses default config
    python train_mil.py --config path/to/config.json

Example Config:
    {
        "data": {
            "labels_csv": "/path/to/labels.csv",
            "features_dir": "/path/to/features/",
            "split_column": "split"
        },
        "train": {
            "num_epochs": 20,
            "learning_rate": 1e-4
        },
        "model_name": "abmil.base.uni_v2.pc108-24k",
        "num_classes": 6,
        "output_dir": "experiments"
    }
"""

import json
import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from downstream.classification.multiple_instance_learning.data_loading.dataset import MILDataset
from downstream.classification.multiple_instance_learning.data_loading.pytorch_adapter import create_dataloader
from downstream.classification.multiple_instance_learning.training.config import (
    ExperimentConfig,
    DataConfig,
    TrainConfig,
    TrackingConfig,
    TaskType,
)
from downstream.classification.multiple_instance_learning.training.tracking import (
    ExperimentTracker,
    create_tracker,
    get_git_info,
    ensure_clean_repo,
    create_experiment_tag,
    GitVersioningError,
)
from downstream.classification.multiple_instance_learning.training.trainer import MILTrainer
from downstream.classification.multiple_instance_learning.training.evaluator import (
    evaluate,
    print_evaluation_results,
)
from downstream.classification.multiple_instance_learning.training.utils import (
    apply_grouping,
    save_predictions,
    save_results_summary,
)
from downstream.classification.multiple_instance_learning.training.encoder_mapping import (
    validate_encoder_consistency,
)
from src.builder import create_model


def _evaluate_and_save(
    model,
    loader,
    config,
    device,
    trainer,
    history,
    run_dir,
    class_labels,
    is_test: bool = True,
):
    """
    Evaluate model and save results.

    Args:
        model: Trained model
        loader: Data loader (test or val)
        config: ExperimentConfig
        device: torch device
        trainer: MILTrainer instance
        history: Training history dict
        run_dir: Output directory
        class_labels: List of class label names
        is_test: True if evaluating on test set, False for validation

    Returns:
        Results dictionary from evaluate()
    """
    split_name = "test" if is_test else "val"
    header = "EVALUATION" if is_test else "VALIDATION EVALUATION (No test set provided)"

    print("\n" + "=" * 70)
    print(header)
    print("=" * 70 + "\n")

    results = evaluate(
        model, loader, device,
        use_amp=config.train.use_amp,
        task_type=config.train.task_type.value,
        num_classes=config.num_classes,
    )

    print_evaluation_results(results, class_labels)

    # Save predictions
    save_predictions(
        run_dir / 'predictions.npz',
        results['labels'],
        results['predictions'],
        class_labels,
        sample_ids=results.get('sample_ids'),
    )
    if is_test:
        print(f"Predictions saved to: {run_dir / 'predictions.npz'}")
        print("  (Use separate plotting script to generate confusion matrix)")

    # Save results summary
    summary = {
        'model_name': config.model_name,
        f'{split_name}_accuracy': results['accuracy'],
        f'{split_name}_balanced_accuracy': results['balanced_accuracy'],
        f'{split_name}_quadratic_kappa': results['quadratic_kappa'],
        'best_val_metric': trainer.best_val_metric,
        'early_stopping_metric': trainer._early_stopping_metric_name,
        'best_epoch': trainer.best_epoch + 1,
        'total_epochs': len(history['train_loss']),
    }
    save_results_summary(run_dir / 'results.json', summary)

    if is_test:
        print(f"\nResults saved to: {run_dir / 'results.json'}")

    # Print final summary
    completion_msg = "TRAINING COMPLETE" if is_test else "TRAINING COMPLETE (CV FOLD)"
    print("\n" + "=" * 80)
    print(completion_msg)
    print("=" * 80)
    metric_name = trainer._early_stopping_metric_name.upper()
    print(f"\nBest Val {metric_name}:      {trainer.best_val_metric:.4f} (epoch {trainer.best_epoch + 1})")
    label = "Test" if is_test else "Val"
    print(f"{label} Accuracy:         {results['accuracy']:.4f}")
    print(f"{label} Balanced Acc:     {results['balanced_accuracy']:.4f}")
    print(f"{label} Quadratic Kappa:  {results['quadratic_kappa']:.4f}")
    print(f"\nAll outputs saved to: {run_dir}")

    return results


def main(
    config: ExperimentConfig,
    checkpoint_path: str = None,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    run_dir: Path = None,
    tracker: ExperimentTracker = None,
):
    """
    Main training function.

    Args:
        config: Experiment configuration
        checkpoint_path: Optional path to local model checkpoint
        train_dataset: Optional pre-split training dataset (for CV)
        val_dataset: Optional pre-split validation dataset (for CV)
        test_dataset: Optional pre-split test dataset (for CV)
        run_dir: Optional explicit run directory (for CV fold organization)
        tracker: Optional ExperimentTracker instance (for nested CV runs)

    Returns:
        Tuple of (results dict, history dict, label_map dict)
    """
    # Check git state if tagging is enabled
    if config.tracking and config.tracking.git_tag:
        try:
            ensure_clean_repo()
        except GitVersioningError as e:
            print(f"\nGit versioning error: {e}")
            print("To disable this check, set 'git_tag: false' in your tracking config.\n")
            raise

    # Initialize tracker if not provided and enabled
    own_tracker = False
    if tracker is None:
        tracker = create_tracker(config)
        own_tracker = True

    # Determine run name
    tracking_run_name = config.run_name or config.model_name

    # Build tags for tracking
    tracking_tags = {
        "model_name": config.model_name,
        "task_type": config.train.task_type.value,
        "mil_model": config._parse_model_name().get('mil_model', 'unknown'),
    }
    # Add encoder metadata to tags
    encoder_name = config._get_encoder_name()
    if encoder_name:
        tracking_tags["encoder"] = encoder_name
    # Add dataset tag if available
    if config.data.dataset_name:
        tracking_tags["dataset"] = config.data.dataset_name

    # Use tracker context if tracker exists and we own it (not nested CV)
    tracker_context = (
        tracker.start_run(
            run_name=tracking_run_name,
            tags=tracking_tags,
        )
        if tracker and own_tracker
        else nullcontext()
    )

    with tracker_context:
        # Log parameters at start of run
        if tracker and own_tracker:
            # Log git info for reproducibility
            tracker.log_params(get_git_info())
            tracker.log_params(config.to_mlflow_params())

        results, history, label_map = _train_and_evaluate(
            config=config,
            checkpoint_path=checkpoint_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            run_dir=run_dir,
            tracker=tracker,
        )

        # Create git tag after successful training (only if we own the tracker)
        if own_tracker and config.tracking and config.tracking.git_tag:
            try:
                tag_name = create_experiment_tag(
                    run_name=config.run_name or config.model_name,
                    metrics={
                        "accuracy": results["accuracy"],
                        "balanced_accuracy": results["balanced_accuracy"],
                        "quadratic_kappa": results["quadratic_kappa"],
                    },
                    push=config.tracking.git_push,
                )
                if tracker:
                    tracker.set_tags({"git_tag": tag_name})
            except GitVersioningError as e:
                print(f"Warning: Failed to create git tag: {e}")

        return results, history, label_map


def _train_and_evaluate(
    config: ExperimentConfig,
    checkpoint_path: str = None,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    run_dir: Path = None,
    tracker: ExperimentTracker = None,
):
    """Internal function containing the actual training logic."""
    print("=" * 80)
    print("MIL TRAINING")
    print(f"Model: {config.model_name}")
    print(f"Output: {config.output_dir}")
    print("=" * 80 + "\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Set seeds
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)

    # Create output directory
    if run_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(config.output_dir) / f'run_{timestamp}'
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}\n")

    # Save config
    config.save(run_dir / 'config.json')

    # Check if datasets are provided (CV mode)
    datasets_provided = train_dataset is not None and val_dataset is not None

    if datasets_provided:
        # Use provided datasets (CV mode)
        print("=" * 70)
        print("USING PROVIDED DATASETS (CV MODE)")
        print("=" * 70 + "\n")

        splits = {
            'train': train_dataset,
            'val': val_dataset,
        }
        if test_dataset is not None:
            splits['test'] = test_dataset

        # Get embed_dim and num_classes from train dataset
        embed_dim = train_dataset.embed_dim
        num_classes = train_dataset.num_classes

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        if test_dataset is not None:
            print(f"Test samples: {len(test_dataset)}")
        print(f"Embed dim: {embed_dim}")
        print(f"Num classes: {num_classes}\n")

    else:
        # Load dataset normally
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70 + "\n")

        dataset = MILDataset(
            labels_csv=config.data.labels_csv,
            features_dir=config.data.features_dir,
        )

        dataset = apply_grouping(dataset, config)

        # Validate encoder consistency if config specifies encoder
        if config.encoder is not None:
            is_valid, msg = validate_encoder_consistency(
                config.model_name,
                config.encoder.name,
                dataset.embed_dim,
            )
            if not is_valid:
                print(f"Warning: {msg}")

        print(f"Dataset type: {type(dataset).__name__}")
        print(f"Total samples: {len(dataset)}")
        print(f"Embed dim: {dataset.embed_dim}")
        print(f"Num classes: {dataset.num_classes}\n")

        # Split dataset
        if config.data.split_column:
            print(f"Using predefined splits from column: {config.data.split_column}")
            splits = dataset.split_by_column(config.data.split_column)
        else:
            print(f"Using random splits (train={config.data.train_frac}, val={config.data.val_frac})")
            splits = dataset.random_split(
                train_frac=config.data.train_frac,
                val_frac=config.data.val_frac,
                seed=config.data.seed,
            )

    print("\nSplit sizes:")
    for name, split_dataset in splits.items():
        print(f"  {name}: {len(split_dataset)} slides")

    # Create dataloaders
    print("\nCreating dataloaders...")

    train_loader, train_adapter = create_dataloader(
        splits['train'],
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        weighted_sampling=config.train.weighted_sampling,
        seed=config.train.seed,
    )

    # Use same label_map for val and test
    label_map = train_adapter.label_map

    val_loader, _ = create_dataloader(
        splits['val'],
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        weighted_sampling=False,
        label_map=label_map,
    )

    # Test loader is optional (not used in CV mode per-fold training)
    test_loader = None
    if 'test' in splits:
        test_loader, _ = create_dataloader(
            splits['test'],
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
            weighted_sampling=False,
            label_map=label_map,
        )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70 + "\n")

    model_kwargs = {
        'num_classes': config.num_classes,
        'dropout': config.train.model_dropout,
    }
    # DFTD doesn't use the gate parameter
    if not config.model_name.lower().startswith('dftd'):
        model_kwargs['gate'] = True

    # Add num_heads for models that support it (e.g., ABMIL)
    if hasattr(config, 'num_heads') and config.num_heads is not None:
        model_kwargs['num_heads'] = config.num_heads

    # Add checkpoint_path if provided for local model loading
    if checkpoint_path:
        print(f"Loading model from local checkpoint: {checkpoint_path}")
        model_kwargs['checkpoint_path'] = checkpoint_path

    # Create the MIL aggregation model
    model = create_model(config.model_name, **model_kwargs).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")

    trainer = MILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.train,
        device=device,
        checkpoint_dir=run_dir,
        tracker=tracker,  # Real-time per-epoch logging
    )

    history = trainer.fit()

    # Load best model for evaluation
    trainer.load_best_model()

    # Get class labels for display
    inverse_label_map = {v: k for k, v in label_map.items()}
    class_labels = [inverse_label_map[i] for i in range(len(label_map))]

    # Evaluate on test set if available, otherwise validation set
    if test_loader is not None:
        results = _evaluate_and_save(
            model, test_loader, config, device, trainer, history,
            run_dir, class_labels, is_test=True,
        )
    else:
        results = _evaluate_and_save(
            model, val_loader, config, device, trainer, history,
            run_dir, class_labels, is_test=False,
        )

    # Log final metrics and artifacts to MLflow
    if tracker:
        split_prefix = 'test' if test_loader is not None else 'val'
        tracker.log_metrics({
            f'{split_prefix}_accuracy': results['accuracy'],
            f'{split_prefix}_balanced_accuracy': results['balanced_accuracy'],
            f'{split_prefix}_quadratic_kappa': results['quadratic_kappa'],
            'best_val_metric': trainer.best_val_metric,
            'best_epoch': trainer.best_epoch + 1,
        })
        # Log lightweight artifacts (config and results, not large model files)
        tracker.log_artifact(run_dir / 'config.json')
        tracker.log_artifact(run_dir / 'results.json')

    return results, history, label_map


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MIL Training Script')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file',
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default=None,
        help='Path to labels CSV file',
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default=None,
        help='Path to features directory',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='abmil.base.uni_v2.pc108-24k',
        help='Model name (default: abmil.base.uni_v2.pc108-24k)',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=6,
        help='Number of classes (default: 6)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs (default: 20)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Output directory (default: experiments)',
    )
    parser.add_argument(
        '--split-column',
        type=str,
        default=None,
        help='Column name for predefined splits',
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=100,
        help='Early stopping patience (default: 100)',
    )
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=10,
        help='Minimum epochs before early stopping (default: 10)',
    )
    parser.add_argument(
        '--hierarchical',
        action='store_true',
        help='Enable hierarchical grouping (default: False)',
    )
    parser.add_argument(
        '--group-column',
        type=str,
        default='case_id',
        help='Column to group by for hierarchical/grouped training (default: case_id)',
    )
    parser.add_argument(
        '--fusion',
        type=str,
        default='early',
        choices=['early', 'late'],
        help='Fusion strategy for multi-slide cases: early (concatenate) or late (average) (default: early)',
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to local model checkpoint (.pth, .pt, .bin, or .safetensors)',
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=1,
        help='Number of attention heads (default: 1)',
    )
    parser.add_argument(
        '--task-type',
        type=str,
        default='multiclass',
        choices=['binary', 'multiclass'],
        help='Task type for metric selection (default: multiclass)',
    )
    parser.add_argument(
        '--early-stopping-metric',
        type=str,
        default='auto',
        choices=['auto', 'kappa', 'balanced_accuracy', 'auc'],
        help='Metric for early stopping (default: auto - kappa for multiclass, auc for binary)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.config:
        # Load from config file
        try:
            config = ExperimentConfig.load(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {args.config}")
            print(f"  {e}")
            exit(1)
    else:
        # Build config from args
        if not args.labels_csv or not args.features_dir:
            print("Error: --labels-csv and --features-dir are required when not using --config")
            print("\nExample usage:")
            print("  python train_mil.py --labels-csv labels.csv --features-dir /path/to/features")
            print("  python train_mil.py --config config.json")
            exit(1)

        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=args.labels_csv,
                features_dir=args.features_dir,
                split_column=args.split_column,
                hierarchical=args.hierarchical,
                group_column=args.group_column,
                fusion=args.fusion,
            ),
            train=TrainConfig(
                num_epochs=args.epochs,
                early_stopping_patience=args.early_stopping_patience,
                min_epochs=args.min_epochs,
                task_type=TaskType(args.task_type),
                early_stopping_metric=args.early_stopping_metric,
            ),
            model_name=args.model,
            num_classes=args.num_classes,
            output_dir=args.output_dir,
            num_heads=args.num_heads,
        )

    main(config, checkpoint_path=args.checkpoint_path)
