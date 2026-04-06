#!/usr/bin/env python3
"""
Cross-Validation Training for MIL Models

Orchestrates n-fold cross-validation with a fixed held-out test set:
1. Randomly select and fix a test set (e.g., 20%)
2. Perform n-fold CV on remaining data (train/val remix each fold)
3. Train a model per fold
4. Aggregate fold metrics (mean +/- std)
5. Ensemble evaluation on held-out test set

Usage:
    python train_mil_cv.py --config configs/panda_multiclass_config.json \
        --num-folds 5 --test-frac 0.2 --seed 42

Output Structure:
    cv_run_YYYYMMDD_HHMMSS/
    |-- cv_config.json           # Full config with CV params
    |-- cv_results.json          # Aggregated metrics
    |-- fold_0/
    |   |-- best_model.pth
    |   |-- results.json
    |   +-- predictions.npz
    |-- fold_1/
    |   +-- ...
    +-- test_evaluation/
        |-- predictions.npz
        +-- metrics.json
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

from data_loading.dataset import MILDataset
from data_loading.pytorch_adapter import create_dataloader
from training.config import ExperimentConfig, DataConfig, TrainConfig, TaskType
from training.tracking import (
    ExperimentTracker,
    create_tracker,
    get_git_info,
    ensure_clean_repo,
    create_experiment_tag,
    GitVersioningError,
)
from training.evaluator import evaluate, calculate_metrics
from training.utils import apply_grouping
from src.builder import create_model
import train_mil


def ensemble_evaluate(
    fold_model_paths: List[Path],
    test_loader,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Ensemble evaluation: average predictions from all fold models.

    Args:
        fold_model_paths: List of paths to best_model.pth for each fold
        test_loader: Test data loader
        config: Experiment configuration
        device: Device to run inference on

    Returns:
        Dictionary with ensemble metrics and predictions
    """
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION")
    print("=" * 70)
    print(f"Loading {len(fold_model_paths)} fold models...")

    # Load all models
    models = []
    for path in fold_model_paths:
        model_kwargs = {
            'num_classes': config.num_classes,
            'dropout': config.train.model_dropout,
        }
        if not config.model_name.lower().startswith('dftd'):
            model_kwargs['gate'] = True
        if hasattr(config, 'num_heads') and config.num_heads is not None:
            model_kwargs['num_heads'] = config.num_heads

        model = create_model(config.model_name, **model_kwargs).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  Loaded: {path}")

    # Run inference with each model
    criterion = nn.CrossEntropyLoss()
    all_fold_logits = []
    all_sample_ids = []

    for model_idx, model in enumerate(models):
        fold_logits = []

        with torch.no_grad():
            for batch in tqdm(
                test_loader,
                desc=f'Model {model_idx + 1}/{len(models)}',
                leave=False
            ):
                # Handle both old format (features, labels, mask) and new (features, labels, mask, ids)
                if len(batch) == 4:
                    features, labels, mask, sample_ids = batch
                    # Only collect sample_ids on first model pass
                    if model_idx == 0:
                        all_sample_ids.extend(sample_ids)
                else:
                    features, labels, *rest = batch

                features = features.to(device)
                labels = labels.to(device)

                if config.train.use_amp:
                    with torch.amp.autocast('cuda'):
                        results_dict, _ = model(
                            features,
                            loss_fn=criterion,
                            label=labels,
                        )
                else:
                    results_dict, _ = model(
                        features,
                        loss_fn=criterion,
                        label=labels,
                    )

                fold_logits.append(results_dict['logits'].cpu())

        all_fold_logits.append(torch.cat(fold_logits, dim=0))

    # Stack and average logits across models
    stacked_logits = torch.stack(all_fold_logits, dim=0)  # [num_folds, N, num_classes]
    ensemble_logits = stacked_logits.mean(dim=0)  # [N, num_classes]
    ensemble_probs = torch.softmax(ensemble_logits, dim=1).numpy()
    ensemble_preds = torch.argmax(ensemble_logits, dim=1).numpy()

    # Get true labels
    all_labels = []
    for batch in test_loader:
        if len(batch) == 4:
            _, labels, _, _ = batch
        else:
            _, labels, *_ = batch
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    # Calculate metrics
    metrics = calculate_metrics(
        all_labels.tolist(),
        ensemble_preds.tolist(),
        y_prob=ensemble_probs,
        task_type=config.train.task_type.value,
        num_classes=config.num_classes,
    )

    print(f"\nEnsemble Test Results:")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
    print(f"  Quadratic Kappa:  {metrics['quadratic_kappa']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:              {metrics['auc']:.4f}")

    result = {
        **metrics,
        'predictions': ensemble_preds.tolist(),
        'labels': all_labels.tolist(),
        'ensemble_logits': ensemble_logits.numpy(),
        'ensemble_probs': ensemble_probs,
    }

    # Include sample_ids if available
    if all_sample_ids:
        result['sample_ids'] = all_sample_ids

    return result


def aggregate_fold_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across folds (mean +/- std).

    Args:
        fold_results: List of result dicts from each fold

    Returns:
        Dictionary with aggregated metrics
    """
    metric_keys = ['accuracy', 'balanced_accuracy', 'quadratic_kappa', 'f1_macro', 'auc']

    aggregated = {}
    for key in metric_keys:
        values = [r.get(key, 0.0) for r in fold_results if key in r]
        if values:
            aggregated[f'mean_{key}'] = float(np.mean(values))
            aggregated[f'std_{key}'] = float(np.std(values))

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Cross-Validation Training for MIL')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON file',
    )
    parser.add_argument(
        '--num-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)',
    )
    parser.add_argument(
        '--test-frac',
        type=float,
        default=0.2,
        help='Fraction of data for held-out test set (default: 0.2)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for CV splits (default: 42)',
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to local model checkpoint',
    )
    args = parser.parse_args()

    # Load base config
    try:
        config = ExperimentConfig.load(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {args.config}")
        print(f"  {e}")
        exit(1)

    # Override CV settings from CLI
    config.data.num_folds = args.num_folds
    config.data.test_frac = args.test_frac
    config.data.cv_seed = args.seed

    # Check git state if tagging is enabled
    if config.tracking and config.tracking.git_tag:
        try:
            ensure_clean_repo()
        except GitVersioningError as e:
            print(f"\nGit versioning error: {e}")
            print("To disable this check, set 'git_tag: false' in your tracking config.\n")
            exit(1)

    print("=" * 80)
    print("CROSS-VALIDATION TRAINING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {config.model_name}")
    print(f"Folds: {args.num_folds}")
    print(f"Test Fraction: {args.test_frac}")
    print(f"CV Seed: {args.seed}")
    print("=" * 80 + "\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Create CV run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv_run_dir = Path(config.output_dir) / f'cv_run_{timestamp}'
    cv_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"CV Run directory: {cv_run_dir}\n")

    # Load dataset
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70 + "\n")

    dataset = MILDataset(
        labels_csv=config.data.labels_csv,
        features_dir=config.data.features_dir,
    )

    dataset = apply_grouping(dataset, config)

    print(f"\nDataset type: {type(dataset).__name__}")
    print(f"Total samples: {len(dataset)}")
    print(f"Embed dim: {dataset.embed_dim}")
    print(f"Num classes: {dataset.num_classes}")

    # Create CV splits
    test_dataset, folds = dataset.create_cv_splits(
        num_folds=args.num_folds,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # Save CV config
    cv_config = {
        'base_config': args.config,
        'num_folds': args.num_folds,
        'test_frac': args.test_frac,
        'cv_seed': args.seed,
        'model_name': config.model_name,
        'num_classes': config.num_classes,
        'total_samples': len(dataset),
        'test_samples': len(test_dataset),
    }
    with open(cv_run_dir / 'cv_config.json', 'w') as f:
        json.dump(cv_config, f, indent=2)

    # Initialize MLflow tracker for CV parent run
    tracker = create_tracker(config)

    # Use MLflow context for parent CV run
    mlflow_context = (
        tracker.start_run(
            run_name=f"CV_{config.model_name}",
            tags={
                "model_name": config.model_name,
                "task_type": config.train.task_type.value,
                "cv_run": "true",
            }
        )
        if tracker
        else nullcontext()
    )

    with mlflow_context:
        # Log CV parameters on parent run
        if tracker:
            # Log git info for reproducibility
            tracker.log_params(get_git_info())
            tracker.log_params({
                **config.to_mlflow_params(),
                "num_folds": args.num_folds,
                "test_frac": args.test_frac,
                "cv_seed": args.seed,
            })

        # Train each fold
        fold_results = []
        fold_model_paths = []
        label_map = None

        for fold_idx, (train_dataset, val_dataset) in enumerate(folds):
            print("\n" + "=" * 80)
            print(f"FOLD {fold_idx + 1}/{args.num_folds}")
            print("=" * 80)

            fold_dir = cv_run_dir / f'fold_{fold_idx}'

            # Update config output_dir temporarily
            fold_config = ExperimentConfig.load(args.config)
            fold_config.data.num_folds = args.num_folds
            fold_config.data.test_frac = args.test_frac
            fold_config.data.cv_seed = args.seed

            # Use nested MLflow run for each fold
            fold_mlflow_context = (
                tracker.start_run(
                    run_name=f"fold_{fold_idx}",
                    nested=True,
                    tags={"cv_fold": str(fold_idx)}
                )
                if tracker
                else nullcontext()
            )

            with fold_mlflow_context:
                # Log fold-specific params
                if tracker:
                    tracker.log_params({"fold": fold_idx})

                # Train this fold (pass tracker for metric logging)
                results, history, label_map = train_mil.main(
                    config=fold_config,
                    checkpoint_path=args.checkpoint_path,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=None,  # Don't evaluate on test during CV
                    run_dir=fold_dir,
                    tracker=tracker,  # Pass tracker for nested logging
                )

                # Log fold metrics to nested run
                if tracker:
                    tracker.log_metrics({
                        'val_accuracy': results['accuracy'],
                        'val_balanced_accuracy': results['balanced_accuracy'],
                        'val_quadratic_kappa': results['quadratic_kappa'],
                        'val_f1_macro': results.get('f1_macro', 0.0),
                        'val_auc': results.get('auc', 0.0),
                    })

            # Store results (these are validation results since no test was provided)
            fold_results.append({
                'fold': fold_idx,
                'val_accuracy': results['accuracy'],
                'val_balanced_accuracy': results['balanced_accuracy'],
                'val_quadratic_kappa': results['quadratic_kappa'],
                'val_f1_macro': results.get('f1_macro', 0.0),
                'val_auc': results.get('auc', 0.0),
            })

            fold_model_paths.append(fold_dir / 'best_model.pth')

            print(f"\nFold {fold_idx} complete:")
            print(f"  Val Accuracy:      {results['accuracy']:.4f}")
            print(f"  Val Balanced Acc:  {results['balanced_accuracy']:.4f}")
            print(f"  Val Kappa:         {results['quadratic_kappa']:.4f}")

        # Aggregate fold metrics
        print("\n" + "=" * 80)
        print("FOLD AGGREGATION")
        print("=" * 80)

        # Calculate mean/std for validation metrics
        val_metrics = ['accuracy', 'balanced_accuracy', 'quadratic_kappa', 'f1_macro', 'auc']
        aggregated = {}

        for metric in val_metrics:
            values = [r.get(f'val_{metric}', 0.0) for r in fold_results]
            aggregated[f'mean_val_{metric}'] = float(np.mean(values))
            aggregated[f'std_val_{metric}'] = float(np.std(values))
            print(f"Val {metric}: {aggregated[f'mean_val_{metric}']:.4f} +/- {aggregated[f'std_val_{metric}']:.4f}")

        # Create test dataloader
        test_loader, _ = create_dataloader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.data.num_workers,
            weighted_sampling=False,
            label_map=label_map,
        )

        # Ensemble evaluation on test set
        ensemble_results = ensemble_evaluate(
            fold_model_paths=fold_model_paths,
            test_loader=test_loader,
            config=config,
            device=device,
        )

        # Save test evaluation
        test_eval_dir = cv_run_dir / 'test_evaluation'
        test_eval_dir.mkdir(parents=True, exist_ok=True)

        # Get class labels
        inverse_label_map = {v: k for k, v in label_map.items()}
        class_labels = [inverse_label_map[i] for i in range(len(label_map))]

        # Save predictions
        save_dict = {
            'labels': ensemble_results['labels'],
            'predictions': ensemble_results['predictions'],
            'ensemble_logits': ensemble_results['ensemble_logits'],
            'ensemble_probs': ensemble_results['ensemble_probs'],
            'class_labels': class_labels,
        }
        if 'sample_ids' in ensemble_results:
            save_dict['sample_ids'] = ensemble_results['sample_ids']

        np.savez(test_eval_dir / 'predictions.npz', **save_dict)

        # Save test metrics
        test_metrics = {
            'accuracy': ensemble_results['accuracy'],
            'balanced_accuracy': ensemble_results['balanced_accuracy'],
            'quadratic_kappa': ensemble_results['quadratic_kappa'],
            'f1_macro': ensemble_results.get('f1_macro', 0.0),
            'auc': ensemble_results.get('auc', 0.0),
        }
        with open(test_eval_dir / 'metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)

        # Log aggregated CV metrics to MLflow parent run
        if tracker:
            tracker.log_metrics({
                'cv_mean_accuracy': aggregated['mean_val_accuracy'],
                'cv_std_accuracy': aggregated['std_val_accuracy'],
                'cv_mean_balanced_accuracy': aggregated['mean_val_balanced_accuracy'],
                'cv_std_balanced_accuracy': aggregated['std_val_balanced_accuracy'],
                'cv_mean_quadratic_kappa': aggregated['mean_val_quadratic_kappa'],
                'cv_std_quadratic_kappa': aggregated['std_val_quadratic_kappa'],
                'test_accuracy': test_metrics['accuracy'],
                'test_balanced_accuracy': test_metrics['balanced_accuracy'],
                'test_quadratic_kappa': test_metrics['quadratic_kappa'],
            })
            # Log CV results artifact
            tracker.log_artifact(cv_run_dir / 'cv_config.json')

        # Compile final CV results
        cv_results = {
            'num_folds': args.num_folds,
            'test_frac': args.test_frac,
            'cv_seed': args.seed,
            'model_name': config.model_name,
            # Aggregated validation metrics
            **aggregated,
            # Per-fold results
            'per_fold': fold_results,
            # Test ensemble metrics
            'test_metrics': test_metrics,
        }

        with open(cv_run_dir / 'cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)

        # Log final CV results to MLflow
        if tracker:
            tracker.log_artifact(cv_run_dir / 'cv_results.json')

        # Create git tag after successful CV training
        if config.tracking and config.tracking.git_tag:
            try:
                cv_run_name = f"CV_{config.run_name or config.model_name}"
                tag_name = create_experiment_tag(
                    run_name=cv_run_name,
                    metrics={
                        "cv_mean_accuracy": aggregated["mean_val_accuracy"],
                        "cv_mean_balanced_accuracy": aggregated["mean_val_balanced_accuracy"],
                        "cv_mean_kappa": aggregated["mean_val_quadratic_kappa"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_kappa": test_metrics["quadratic_kappa"],
                    },
                    push=config.tracking.git_push,
                )
                if tracker:
                    tracker.set_tags({"git_tag": tag_name})
            except GitVersioningError as e:
                print(f"Warning: Failed to create git tag: {e}")

        # Print final summary
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\nValidation (Mean +/- Std across {args.num_folds} folds):")
        print(f"  Accuracy:      {aggregated['mean_val_accuracy']:.4f} +/- {aggregated['std_val_accuracy']:.4f}")
        print(f"  Balanced Acc:  {aggregated['mean_val_balanced_accuracy']:.4f} +/- {aggregated['std_val_balanced_accuracy']:.4f}")
        print(f"  Kappa:         {aggregated['mean_val_quadratic_kappa']:.4f} +/- {aggregated['std_val_quadratic_kappa']:.4f}")

        print(f"\nTest Set (Ensemble of {args.num_folds} models):")
        print(f"  Accuracy:      {test_metrics['accuracy']:.4f}")
        print(f"  Balanced Acc:  {test_metrics['balanced_accuracy']:.4f}")
        print(f"  Kappa:         {test_metrics['quadratic_kappa']:.4f}")

        print(f"\nAll outputs saved to: {cv_run_dir}")
        print(f"  - cv_config.json")
        print(f"  - cv_results.json")
        for i in range(args.num_folds):
            print(f"  - fold_{i}/")
        print(f"  - test_evaluation/")


if __name__ == '__main__':
    main()
