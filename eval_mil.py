#!/usr/bin/env python3
"""
MIL Evaluation Script

Evaluate a trained MIL model on test data.

Usage:
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
from training.evaluator import evaluate, print_evaluation_results
from training.utils import save_predictions
from src.builder import create_model


def main():
    args = parse_args()

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
        labels_csv=args.labels_csv,
        features_dir=args.features_dir,
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

    model = create_model(args.model_name, num_classes=args.num_classes)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f"Model: {args.model_name}")
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
        task_type=args.task_type,
        num_classes=args.num_classes,
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
    parser = argparse.ArgumentParser(description='Evaluate a trained MIL model')

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (best_model.pth)',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model architecture string (e.g., abmil.base.uni_v2.none)',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        required=True,
        help='Number of output classes',
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        required=True,
        help='Path to CSV with test labels',
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        required=True,
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
        default='multiclass',
        choices=['binary', 'multiclass'],
        help='Task type: binary or multiclass (default: multiclass)',
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
