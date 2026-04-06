#!/usr/bin/env python3
"""
MIL Inference Script - Predict labels for pre-encoded slides.

Pure prediction without ground truth labels. For evaluation with metrics,
use eval_mil.py instead.

Usage:
    # With config file - single slide
    python infer_mil.py \
        --config configs/panda_config.json \
        --checkpoint experiments/panda_cv/best_model.pth \
        --features /path/to/slide.h5

    # With config file - all slides in directory
    python infer_mil.py \
        --config configs/panda_config.json \
        --checkpoint experiments/panda_cv/best_model.pth \
        --output predictions.csv

    # Without config - single slide
    python infer_mil.py \
        --checkpoint experiments/panda_cv/best_model.pth \
        --model-name abmil.base.uni_v2.none \
        --num-classes 6 \
        --features /path/to/slide.h5 \
        --label-names "0,1,2,3,4,5"

    # Without config - multiple slides from directory
    python infer_mil.py \
        --checkpoint experiments/panda_cv/best_model.pth \
        --model-name abmil.base.uni_v2.none \
        --num-classes 6 \
        --features-dir /path/to/features/ \
        --slide-ids slide1,slide2,slide3 \
        --output predictions.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from data_loading.feature_loader import (
    CLAMFeatureLoader,
    get_slide_ids,
    load_features,
    prepare_for_mil,
)
from training.config import ExperimentConfig
from src.builder import create_model


def load_model(
    checkpoint_path: str,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_name: Model architecture string (e.g., abmil.base.uni_v2.none)
        num_classes: Number of output classes
        device: Device to load model onto

    Returns:
        Model in eval mode
    """
    model = create_model(model_name, num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def predict_single(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Run inference on a single slide.

    Args:
        model: Trained MIL model
        features: [M, D] tensor of patch features
        device: Device to run inference on

    Returns:
        Tuple of (predicted_class_index, confidence)
    """
    # Prepare features: [M, D] -> [1, M, D]
    features = prepare_for_mil(features).to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(features)

    # Get probabilities and prediction
    probs = F.softmax(logits, dim=1)
    confidence, pred_idx = probs.max(dim=1)

    return pred_idx.item(), confidence.item()


def predict_batch(
    model: torch.nn.Module,
    feature_loader: CLAMFeatureLoader,
    device: torch.device,
) -> List[Tuple[str, int, float]]:
    """
    Run inference on multiple slides.

    Args:
        model: Trained MIL model
        feature_loader: Iterator yielding (slide_id, features) tuples
        device: Device to run inference on

    Returns:
        List of (slide_id, predicted_class_index, confidence) tuples
    """
    results = []

    for slide_id, features in feature_loader:
        pred_idx, confidence = predict_single(model, features, device)
        results.append((slide_id, pred_idx, confidence))

    return results


def write_csv(
    results: List[Tuple[str, int, float]],
    label_names: List[str],
    output_path: Optional[str] = None,
):
    """
    Write predictions to CSV.

    Args:
        results: List of (slide_id, pred_idx, confidence) tuples
        label_names: List of class names for human-readable labels
        output_path: Path to output CSV (None for stdout)
    """
    output = open(output_path, 'w', newline='') if output_path else sys.stdout

    try:
        writer = csv.writer(output)
        writer.writerow(['slide_id', 'predicted_label', 'confidence'])

        for slide_id, pred_idx, confidence in results:
            label = label_names[pred_idx]
            writer.writerow([slide_id, label, f'{confidence:.4f}'])
    finally:
        if output_path:
            output.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MIL inference on pre-encoded slide features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file (makes model-name, num-classes, features-dir optional)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file (provides model-name, num-classes, features-dir)',
    )

    # Required arguments (or from config)
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth)',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model architecture (e.g., abmil.base.uni_v2.none)',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Number of output classes',
    )

    # Input options
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Path to single .h5 feature file',
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default=None,
        help='Directory containing .h5 feature files',
    )

    # Optional arguments
    parser.add_argument(
        '--slide-ids',
        type=str,
        default=None,
        help='Comma-separated list of slide IDs (default: all in directory)',
    )
    parser.add_argument(
        '--label-names',
        type=str,
        default=None,
        help='Comma-separated class labels (default: 0,1,2,...)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: stdout)',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config if provided
    config = None
    if args.config:
        try:
            config = ExperimentConfig.load(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {args.config}", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            sys.exit(1)

    # Resolve parameters: CLI args override config
    model_name = args.model_name or (config.model_name if config else None)
    num_classes = args.num_classes or (config.num_classes if config else None)
    features_dir = args.features_dir or (config.data.features_dir if config else None)

    # Validate required parameters
    missing = []
    if not model_name:
        missing.append('--model-name')
    if not num_classes:
        missing.append('--num-classes')

    # Need either --features or --features-dir
    if not args.features and not features_dir:
        missing.append('--features or --features-dir')

    if missing:
        print(f"Error: Missing required arguments: {', '.join(missing)}", file=sys.stderr)
        print("Provide these via --config or as command line arguments.", file=sys.stderr)
        sys.exit(1)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse label names
    if args.label_names:
        label_names = [name.strip() for name in args.label_names.split(',')]
        if len(label_names) != num_classes:
            print(
                f'Error: --label-names has {len(label_names)} labels but '
                f'--num-classes is {num_classes}',
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        label_names = [str(i) for i in range(num_classes)]

    # Load model
    print(f'Loading model: {model_name}', file=sys.stderr)
    print(f'Checkpoint: {args.checkpoint}', file=sys.stderr)
    print(f'Device: {device}', file=sys.stderr)

    model = load_model(args.checkpoint, model_name, num_classes, device)

    # Run inference
    if args.features:
        # Single slide (explicit --features takes precedence)
        features_path = Path(args.features)
        slide_id = features_path.stem

        print(f'Processing: {slide_id}', file=sys.stderr)
        features = load_features(features_path)
        pred_idx, confidence = predict_single(model, features, device)
        results = [(slide_id, pred_idx, confidence)]

    else:
        # Multiple slides from directory
        features_dir_path = Path(features_dir)

        if args.slide_ids:
            slide_ids = [s.strip() for s in args.slide_ids.split(',')]
        else:
            slide_ids = get_slide_ids(features_dir_path)

        print(f'Processing {len(slide_ids)} slides from: {features_dir_path}', file=sys.stderr)

        loader = CLAMFeatureLoader(features_dir_path, slide_ids=slide_ids)
        results = predict_batch(model, loader, device)

    # Output results
    write_csv(results, label_names, args.output)

    if args.output:
        print(f'Predictions saved to: {args.output}', file=sys.stderr)
    else:
        print(f'Processed {len(results)} slides', file=sys.stderr)


if __name__ == '__main__':
    main()
