#!/usr/bin/env python3
"""
Feature Extraction Entry Point

Extract patch features from whole-slide images using Trident encoders.
Outputs H5 files compatible with IbinSina's MIL training pipeline.

Pipeline: WSI → Segmentation → Patching → Feature Extraction → H5

Usage:
    # With config file
    python extract_features.py --config configs/extraction_panda_uni_v2.json

    # With CLI arguments
    python extract_features.py \
        --wsi-dir /path/to/slides \
        --output-dir /path/to/output \
        --encoder uni_v2 \
        --magnification 20 \
        --patch-size 256

    # Resume interrupted extraction
    python extract_features.py --config config.json --resume

Example Config:
    {
        "input": {
            "wsi_dir": "/data/slides",
            "wsi_extensions": [".svs", ".ndpi"]
        },
        "encoder": {
            "name": "uni_v2",
            "precision": "fp16",
            "batch_size": 512
        },
        "patching": {
            "magnification": 20,
            "patch_size": 256
        },
        "output_dir": "/data/features"
    }
"""

import argparse
import json
import sys

from feature_extraction.foundational_models import (
    ExtractionConfig,
    InputConfig,
    SegmentationConfig,
    PatchingConfig,
    EncoderConfig,
    ProcessingConfig,
    TridentExtractor,
)
from downstream.classification.multiple_instance_learning.training.config import TrackingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract patch features from WSIs using Trident",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python extract_features.py --config configs/extraction_panda_uni_v2.json

  # Using CLI arguments
  python extract_features.py \\
      --wsi-dir /data/panda/wsis \\
      --output-dir /data/panda/features \\
      --encoder uni_v2 \\
      --magnification 20

  # Resume with skip_errors enabled
  python extract_features.py --config config.json --resume --skip-errors
        """,
    )

    # Config file (takes precedence)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to extraction config JSON file",
    )

    # Input settings
    parser.add_argument(
        "--wsi-dir",
        type=str,
        default=None,
        help="Directory containing WSI files",
    )
    parser.add_argument(
        "--wsi-extensions",
        type=str,
        nargs="+",
        default=[".svs", ".ndpi", ".tiff"],
        help="WSI file extensions to include (default: .svs .ndpi .tiff)",
    )
    parser.add_argument(
        "--slide-list",
        type=str,
        default=None,
        help="Optional CSV/TXT file with list of slides to process",
    )
    parser.add_argument(
        "--search-nested",
        action="store_true",
        help="Recursively search subdirectories for WSIs",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for features",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this extraction run",
    )

    # Encoder settings
    parser.add_argument(
        "--encoder",
        type=str,
        default="uni_v2",
        help="Encoder model name (default: uni_v2)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Floating-point precision (default: fp16)",
    )
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=512,
        help="Batch size for feature extraction (default: 512)",
    )

    # Patching settings
    parser.add_argument(
        "--magnification",
        type=int,
        default=20,
        help="Target magnification for patches (default: 20)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size in pixels (default: 256)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between patches in pixels (default: 0)",
    )
    parser.add_argument(
        "--min-tissue",
        type=float,
        default=0.0,
        help="Minimum tissue proportion per patch (default: 0.0)",
    )

    # Segmentation settings
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="grandqc",
        choices=["grandqc", "hest", "otsu"],
        help="Tissue segmentation model (default: grandqc)",
    )
    parser.add_argument(
        "--segmentation-mag",
        type=int,
        default=10,
        help="Magnification for segmentation (default: 10)",
    )
    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=16,
        help="Batch size for segmentation (default: 16)",
    )

    # Processing settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Computation device (default: cuda:0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue if individual slides fail",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip slides that already have features extracted",
    )

    # Tracking settings
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable experiment tracking (MLflow)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="feature-extraction",
        help="Experiment name for tracking (default: feature-extraction)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config from file or build from args
    if args.config:
        try:
            config = ExtractionConfig.load(args.config)
            print(f"Loaded config from: {args.config}\n")

            # Apply CLI overrides
            if args.resume:
                config.processing.resume = True
            if args.skip_errors:
                config.processing.skip_errors = True
            if args.device:
                config.processing.device = args.device

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {args.config}")
            print(f"  {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid config value: {e}")
            sys.exit(1)
    else:
        # Build config from CLI arguments
        if not args.wsi_dir or not args.output_dir:
            print("Error: --wsi-dir and --output-dir are required when not using --config")
            print("\nExample usage:")
            print("  python extract_features.py --wsi-dir /data/slides --output-dir /data/features")
            print("  python extract_features.py --config configs/extraction_config.json")
            sys.exit(1)

        try:
            config = ExtractionConfig(
                input=InputConfig(
                    wsi_dir=args.wsi_dir,
                    wsi_extensions=args.wsi_extensions,
                    slide_list=args.slide_list,
                    search_nested=args.search_nested,
                ),
                segmentation=SegmentationConfig(
                    model=args.segmentation_model,
                    magnification=args.segmentation_mag,
                    batch_size=args.segmentation_batch_size,
                ),
                patching=PatchingConfig(
                    magnification=args.magnification,
                    patch_size=args.patch_size,
                    overlap=args.overlap,
                    min_tissue_proportion=args.min_tissue,
                ),
                encoder=EncoderConfig(
                    name=args.encoder,
                    precision=args.precision,
                    batch_size=args.encoder_batch_size,
                ),
                processing=ProcessingConfig(
                    device=args.device,
                    num_workers=args.num_workers,
                    skip_errors=args.skip_errors,
                    resume=args.resume,
                ),
                output_dir=args.output_dir,
                run_name=args.run_name,
                tracking=TrackingConfig(
                    enabled=args.track,
                    experiment_name=args.experiment_name,
                    backend="mlflow",
                ) if args.track else None,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Run extraction
    extractor = TridentExtractor(config)
    results = extractor.run()

    # Print final location
    if results["features_dir"]:
        print(f"\nFeatures ready for MIL training at: {results['features_dir']}")
    else:
        print("\nNo features extracted. Check warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
