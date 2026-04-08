#!/usr/bin/env python3
"""
TridentExtractor: Wrapper for Trident-based WSI feature extraction.

Provides a clean interface for running the full extraction pipeline:
WSI → Segmentation → Patching → Feature Extraction → H5
"""

import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import ExtractionConfig
from downstream.classification.multiple_instance_learning.training.tracking import (
    ExperimentTracker,
    create_tracker,
    get_git_info,
)
from downstream.classification.multiple_instance_learning.training.encoder_mapping import (
    get_encoder_dim,
)


class TridentExtractor:
    """Feature extraction pipeline using Trident.

    This class orchestrates the full feature extraction workflow:
    1. Initialize Trident Processor with WSI source
    2. Run tissue segmentation
    3. Extract patch coordinates
    4. Compute patch features
    5. Save metadata and log to tracker

    Example:
        config = ExtractionConfig.load("configs/extraction.json")
        extractor = TridentExtractor(config)
        results = extractor.run()
        print(f"Features saved to: {results['features_dir']}")
    """

    def __init__(
        self,
        config: ExtractionConfig,
        tracker: Optional[ExperimentTracker] = None,
    ):
        """Initialize the extractor.

        Args:
            config: Extraction configuration
            tracker: Optional experiment tracker (created from config if not provided)
        """
        self.config = config
        self.tracker = tracker
        self._own_tracker = False

        # Will be initialized in run()
        self._processor = None
        self._encoder = None
        self._segmentation_model = None

    def run(self) -> Dict[str, Any]:
        """Run the full extraction pipeline.

        Returns:
            Dictionary with:
                - features_dir: Path to extracted features
                - stats: Dictionary with extraction statistics
                - job_dir: Path to job directory with all outputs
        """
        from trident import Processor
        from trident.patch_encoder_models import encoder_factory
        from trident.segmentation_models import segmentation_model_factory

        config = self.config
        job_dir = config.get_job_dir()

        print("=" * 80)
        print("TRIDENT FEATURE EXTRACTION")
        print(f"WSI Source: {config.input.wsi_dir}")
        print(f"Output: {job_dir}")
        print(f"Encoder: {config.encoder.name}")
        print("=" * 80 + "\n")

        # Initialize tracker if not provided
        if self.tracker is None and config.tracking and config.tracking.enabled:
            # Create a minimal config object for create_tracker
            class _ConfigWrapper:
                def __init__(self, tracking):
                    self.tracking = tracking

            self.tracker = create_tracker(_ConfigWrapper(config.tracking))
            self._own_tracker = True

        # Start tracking run
        tracker_context = (
            self.tracker.start_run(
                run_name=config.run_name,
                tags={
                    "encoder": config.encoder.name,
                    "magnification": str(config.patching.magnification),
                    "patch_size": str(config.patching.patch_size),
                    "task": "feature-extraction",
                },
            )
            if self.tracker and self._own_tracker
            else nullcontext()
        )

        with tracker_context:
            # Log parameters
            if self.tracker and self._own_tracker:
                self.tracker.log_params(get_git_info())
                self.tracker.log_params(config.to_mlflow_params())

            # Initialize Processor
            print("Initializing Trident Processor...")
            self._processor = Processor(
                job_dir=str(job_dir),
                wsi_source=config.input.wsi_dir,
                wsi_ext=config.input.wsi_extensions,
                custom_list_of_wsis=config.input.slide_list,
                skip_errors=config.processing.skip_errors,
                search_nested=config.input.search_nested,
                max_workers=config.processing.num_workers,
            )

            num_slides = len(self._processor.wsis)
            print(f"Found {num_slides} slides to process\n")

            if num_slides == 0:
                print("WARNING: No slides found. Check wsi_dir and wsi_extensions.")
                return {
                    "features_dir": None,
                    "stats": {"slides_found": 0},
                    "job_dir": str(job_dir),
                }

            # Step 1: Segmentation
            print("=" * 70)
            print("STEP 1: TISSUE SEGMENTATION")
            print("=" * 70 + "\n")

            self._segmentation_model = segmentation_model_factory(
                model_name=config.segmentation.model
            )

            segmentation_dir = self._processor.run_segmentation_job(
                segmentation_model=self._segmentation_model,
                seg_mag=config.segmentation.magnification,
                batch_size=config.segmentation.batch_size,
                device=config.processing.device,
            )
            print(f"Segmentation saved to: {segmentation_dir}\n")

            # Step 2: Patching
            print("=" * 70)
            print("STEP 2: PATCH EXTRACTION")
            print("=" * 70 + "\n")

            coords_dir = self._processor.run_patching_job(
                target_magnification=config.patching.magnification,
                patch_size=config.patching.patch_size,
                overlap=config.patching.overlap,
                min_tissue_proportion=config.patching.min_tissue_proportion,
                visualize=True,
            )
            print(f"Patch coordinates saved to: {coords_dir}\n")

            # Step 3: Feature Extraction
            print("=" * 70)
            print("STEP 3: FEATURE EXTRACTION")
            print("=" * 70 + "\n")

            print(f"Loading encoder: {config.encoder.name}...")
            self._encoder = encoder_factory(config.encoder.name)
            self._encoder = self._encoder.to(config.processing.device)
            self._encoder.eval()

            # Set precision
            if config.encoder.precision == "fp16":
                self._encoder = self._encoder.half()
            elif config.encoder.precision == "bf16":
                self._encoder = self._encoder.to(torch.bfloat16)

            features_dir = self._processor.run_patch_feature_extraction_job(
                coords_dir=coords_dir,
                patch_encoder=self._encoder,
                device=config.processing.device,
                batch_limit=config.encoder.batch_size,
                saveas="h5",
            )
            print(f"Features saved to: {features_dir}\n")

            # Collect statistics
            stats = self._collect_stats(features_dir, coords_dir)

            # Save metadata
            self._save_metadata(job_dir, stats)

            # Log metrics to tracker
            if self.tracker:
                self.tracker.log_metrics({
                    "slides_processed": stats["slides_processed"],
                    "slides_skipped": stats["slides_skipped"],
                    "total_patches": stats["total_patches"],
                    "avg_patches_per_slide": stats["avg_patches_per_slide"],
                })

                # Log config as artifact
                config_path = job_dir / "extraction_config.json"
                if config_path.exists():
                    self.tracker.log_artifact(config_path)

            # Release resources
            self._processor.release()
            self._processor = None
            self._encoder = None
            self._segmentation_model = None

            # Print summary
            self._print_summary(stats, features_dir)

            return {
                "features_dir": str(features_dir),
                "stats": stats,
                "job_dir": str(job_dir),
            }

    def _collect_stats(self, features_dir: str, coords_dir: str) -> Dict[str, Any]:
        """Collect extraction statistics.

        Args:
            features_dir: Path to extracted features
            coords_dir: Path to patch coordinates

        Returns:
            Dictionary with extraction statistics
        """
        features_path = Path(features_dir)
        coords_path = Path(coords_dir)

        # Count processed slides
        h5_files = list(features_path.glob("*.h5"))
        slides_processed = len(h5_files)

        # Count total patches
        total_patches = 0
        patch_counts = []

        try:
            import h5py

            for h5_file in h5_files:
                with h5py.File(h5_file, "r") as f:
                    if "features" in f:
                        num_patches = f["features"].shape[0]
                        total_patches += num_patches
                        patch_counts.append(num_patches)
        except ImportError:
            print("Warning: h5py not available, skipping patch count statistics")
        except Exception as e:
            print(f"Warning: Could not read H5 files for statistics: {e}")

        # Calculate averages
        avg_patches = total_patches / slides_processed if slides_processed > 0 else 0

        # Count expected vs processed
        coord_files = list(coords_path.glob("*.h5"))
        slides_skipped = len(coord_files) - slides_processed

        return {
            "slides_processed": slides_processed,
            "slides_skipped": max(0, slides_skipped),
            "total_patches": total_patches,
            "avg_patches_per_slide": round(avg_patches, 1),
            "min_patches": min(patch_counts) if patch_counts else 0,
            "max_patches": max(patch_counts) if patch_counts else 0,
            "encoder_name": self.config.encoder.name,
            "encoder_dim": get_encoder_dim(self.config.encoder.name),
            "magnification": self.config.patching.magnification,
            "patch_size": self.config.patching.patch_size,
        }

    def _save_metadata(self, job_dir: Path, stats: Dict[str, Any]):
        """Save extraction metadata to JSON files.

        Args:
            job_dir: Job directory path
            stats: Extraction statistics
        """
        # Save extraction config
        config_path = job_dir / "extraction_config.json"
        self.config.save(str(config_path))
        print(f"Config saved to: {config_path}")

        # Save metadata with stats
        metadata = {
            "extraction_timestamp": datetime.now().isoformat(),
            "encoder": {
                "name": self.config.encoder.name,
                "dim": get_encoder_dim(self.config.encoder.name),
                "precision": self.config.encoder.precision,
            },
            "patching": {
                "magnification": self.config.patching.magnification,
                "patch_size": self.config.patching.patch_size,
                "overlap": self.config.patching.overlap,
            },
            "segmentation": {
                "model": self.config.segmentation.model,
                "magnification": self.config.segmentation.magnification,
            },
            "statistics": stats,
        }

        metadata_path = job_dir / "extraction_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    def _print_summary(self, stats: Dict[str, Any], features_dir: str):
        """Print extraction summary.

        Args:
            stats: Extraction statistics
            features_dir: Path to extracted features
        """
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\nSlides processed:     {stats['slides_processed']}")
        print(f"Slides skipped:       {stats['slides_skipped']}")
        print(f"Total patches:        {stats['total_patches']:,}")
        print(f"Avg patches/slide:    {stats['avg_patches_per_slide']:.1f}")
        print(f"Encoder:              {stats['encoder_name']} (dim={stats['encoder_dim']})")
        print(f"Patch size:           {stats['patch_size']}px @ {stats['magnification']}x")
        print(f"\nFeatures saved to: {features_dir}")
