#!/usr/bin/env python3
"""
MIL Evaluation Script

Evaluate a trained MIL model on test data.

Usage:
    # With config file
    python eval_mil.py \
        --config configs/panda_config.json \
        --checkpoint experiments/run_001/best_model.pth

    # With explicit arguments
    python eval_mil.py \
        --checkpoint experiments/run_001/best_model.pth \
        --model-name abmil.base.uni_v2.none \
        --num-classes 6 \
        --labels-csv /path/to/test_labels.csv \
        --features-dir /path/to/test_features/
"""

import argparse
import json
from pathlib import Path

import torch

from data_loading.dataset import MILDataset
from data_loading.pytorch_adapter import create_dataloader
from training.config import ExperimentConfig
from training.evaluator import evaluate, print_evaluation_results
from training.utils import save_predictions
from src.builder import create_model


def main():
    args = parse_args()

    # Load config if provided
    config = None
    if args.config:
        try:
            config = ExperimentConfig.load(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {args.config}")
            print(f"  {e}")
            exit(1)

    # Resolve parameters: CLI args override config
    model_name = args.model_name or (config.model_name if config else None)
    num_classes = args.num_classes or (config.num_classes if config else None)
    labels_csv = args.labels_csv or (config.data.labels_csv if config else None)
    features_dir = args.features_dir or (config.data.features_dir if config else None)
    task_type = args.task_type or (config.train.task_type.value if config else 'multiclass')

    # Validate required parameters
    missing = []
    if not model_name:
        missing.append('--model-name')
    if not num_classes:
        missing.append('--num-classes')
    if not labels_csv:
        missing.append('--labels-csv')
    if not features_dir:
        missing.append('--features-dir')

    if missing:
        print(f"Error: Missing required arguments: {', '.join(missing)}")
        print("Provide these via --config or as command line arguments.")
        exit(1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70 + "\n")

    dataset = MILDataset(
        labels_csv=labels_csv,
        features_dir=features_dir,
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Embed dim: {dataset.embed_dim}")
    print(f"Num classes: {dataset.num_classes}\n")

    # Create dataloader
    loader, adapter = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        weighted_sampling=False,
    )

    label_map = adapter.label_map
    inverse_label_map = {v: k for k, v in label_map.items()}
    class_labels = [inverse_label_map[i] for i in range(len(label_map))]

    # Create model
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70 + "\n")

    model = create_model(model_name, num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Evaluate
    print("=" * 70)
    print("EVALUATING")
    print("=" * 70 + "\n")

    results = evaluate(
        model=model,
        test_loader=loader,
        device=device,
        use_amp=True,
        task_type=task_type,
        num_classes=num_classes,
    )

    print_evaluation_results(results, class_labels)

    # Save predictions
    save_predictions(
        output_dir / 'predictions.npz',
        results['labels'],
        results['predictions'],
        class_labels,
    )
    print(f"Predictions saved to: {output_dir / 'predictions.npz'}")

    # Save metrics
    metrics = {
        'accuracy': results['accuracy'],
        'balanced_accuracy': results['balanced_accuracy'],
        'quadratic_kappa': results['quadratic_kappa'],
        'f1_macro': results['f1_macro'],
        'precision_macro': results['precision_macro'],
    }
    if 'auc' in results:
        metrics['auc'] = results['auc']

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_dir / 'metrics.json'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained MIL model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file (makes other args optional)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file (provides model-name, num-classes, labels-csv, features-dir)',
    )

    # Required arguments (or from config)
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (best_model.pth)',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model architecture string (e.g., abmil.base.uni_v2.none)',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Number of output classes',
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default=None,
        help='Path to CSV with test labels',
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default=None,
        help='Path to directory with test feature files',
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./eval_output',
        help='Directory to save results (default: ./eval_output)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for evaluation (default: 1)',
    )
    parser.add_argument(
        '--task-type',
        type=str,
        default=None,
        choices=['binary', 'multiclass'],
        help='Task type: binary or multiclass (default: from config or multiclass)',
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
