#!/usr/bin/env python3
"""
Configuration dataclasses for Trident feature extraction.

Mirrors the training config pattern with dataclasses that can be loaded from JSON.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import json

from downstream.classification.multiple_instance_learning.training.config import TrackingConfig


@dataclass
class InputConfig:
    """Configuration for input WSI files.

    Attributes:
        wsi_dir: Directory containing WSI files
        wsi_extensions: List of accepted file extensions (e.g., [".svs", ".ndpi"])
        slide_list: Optional path to CSV/TXT file listing slides to process
        search_nested: If True, recursively search subdirectories for WSIs
    """

    wsi_dir: str
    wsi_extensions: List[str] = field(default_factory=lambda: [".svs", ".ndpi", ".tiff"])
    slide_list: Optional[str] = None
    search_nested: bool = False

    def __post_init__(self):
        # Validate wsi_dir exists
        if not Path(self.wsi_dir).exists():
            raise FileNotFoundError(f"WSI directory not found: {self.wsi_dir}")

        # Validate slide_list if provided
        if self.slide_list and not Path(self.slide_list).exists():
            raise FileNotFoundError(f"Slide list file not found: {self.slide_list}")

        # Ensure extensions start with dot
        self.wsi_extensions = [
            ext if ext.startswith(".") else f".{ext}" for ext in self.wsi_extensions
        ]


@dataclass
class SegmentationConfig:
    """Configuration for tissue segmentation.

    Attributes:
        model: Segmentation model name (grandqc, hest, otsu)
        magnification: Magnification level for segmentation (typically 10x)
        batch_size: Batch size for segmentation model inference
    """

    model: str = "grandqc"
    magnification: int = 10
    batch_size: int = 16

    def __post_init__(self):
        valid_models = {"grandqc", "hest", "otsu"}
        if self.model.lower() not in valid_models:
            raise ValueError(
                f"Invalid segmentation model: '{self.model}'. "
                f"Valid options: {sorted(valid_models)}"
            )


@dataclass
class PatchingConfig:
    """Configuration for patch extraction.

    Attributes:
        magnification: Target magnification for patches (e.g., 20x)
        patch_size: Patch size in pixels at target magnification
        overlap: Overlap between adjacent patches in pixels
        min_tissue_proportion: Minimum tissue proportion to keep a patch (0.0-1.0)
    """

    magnification: int = 20
    patch_size: int = 256
    overlap: int = 0
    min_tissue_proportion: float = 0.0

    def __post_init__(self):
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")

        if not 0.0 <= self.min_tissue_proportion <= 1.0:
            raise ValueError(
                f"min_tissue_proportion must be in [0, 1], got {self.min_tissue_proportion}"
            )


@dataclass
class EncoderConfig:
    """Configuration for patch encoder.

    Attributes:
        name: Encoder model name (e.g., "uni_v2", "conch_v15", "gigapath")
        precision: Floating-point precision (fp32, fp16, bf16)
        batch_size: Batch size for feature extraction
    """

    name: str = "uni_v2"
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    batch_size: int = 512

    def __post_init__(self):
        # Import here to avoid circular imports
        from downstream.classification.multiple_instance_learning.training.encoder_mapping import get_encoder_dim

        # Validate encoder name
        dim = get_encoder_dim(self.name)
        if dim is None:
            from downstream.classification.multiple_instance_learning.training.encoder_mapping import list_available_encoders

            available = list_available_encoders()
            raise ValueError(
                f"Unknown encoder: '{self.name}'. "
                f"Available encoders: {available[:10]}... "
                f"(see encoder_mapping module for full list)"
            )


@dataclass
class ProcessingConfig:
    """Configuration for processing behavior.

    Attributes:
        device: Computation device (e.g., "cuda:0", "cpu")
        num_workers: Number of data loading workers
        skip_errors: If True, continue processing if a slide fails
        resume: If True, skip slides that already have features extracted
    """

    device: str = "cuda:0"
    num_workers: int = 4
    skip_errors: bool = True
    resume: bool = True


@dataclass
class ExtractionConfig:
    """Top-level configuration for feature extraction.

    This config mirrors the ExperimentConfig pattern from training,
    with nested configs for each pipeline stage.
    """

    input: InputConfig
    segmentation: SegmentationConfig
    patching: PatchingConfig
    encoder: EncoderConfig
    processing: ProcessingConfig
    output_dir: str
    run_name: Optional[str] = None
    tracking: Optional[TrackingConfig] = None

    def __post_init__(self):
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Generate run name if not provided
        if self.run_name is None:
            self.run_name = f"{self.encoder.name}_{self.patching.magnification}x"

        # Default tracking config
        if self.tracking is None:
            self.tracking = TrackingConfig(
                enabled=False,
                experiment_name="feature-extraction",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return asdict(self)

    def to_mlflow_params(self) -> Dict[str, Any]:
        """Flatten config to MLflow-compatible params dict."""
        return {
            # Encoder params
            "encoder.name": self.encoder.name,
            "encoder.precision": self.encoder.precision,
            "encoder.batch_size": self.encoder.batch_size,
            # Patching params
            "patching.magnification": self.patching.magnification,
            "patching.patch_size": self.patching.patch_size,
            "patching.overlap": self.patching.overlap,
            "patching.min_tissue_proportion": self.patching.min_tissue_proportion,
            # Segmentation params
            "segmentation.model": self.segmentation.model,
            "segmentation.magnification": self.segmentation.magnification,
            # Processing params
            "processing.device": self.processing.device,
            "processing.num_workers": self.processing.num_workers,
            "processing.skip_errors": self.processing.skip_errors,
            "processing.resume": self.processing.resume,
            # Input params
            "input.wsi_dir": self.input.wsi_dir,
            "input.search_nested": self.input.search_nested,
        }

    def save(self, path: str):
        """Save config to JSON file."""
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExtractionConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Load nested configs
        input_config = InputConfig(**data["input"])
        segmentation_config = SegmentationConfig(**data.get("segmentation", {}))
        patching_config = PatchingConfig(**data.get("patching", {}))
        encoder_config = EncoderConfig(**data.get("encoder", {}))
        processing_config = ProcessingConfig(**data.get("processing", {}))

        # Load tracking config if present
        tracking_config = None
        if "tracking" in data:
            tracking_config = TrackingConfig(**data["tracking"])

        return cls(
            input=input_config,
            segmentation=segmentation_config,
            patching=patching_config,
            encoder=encoder_config,
            processing=processing_config,
            output_dir=data["output_dir"],
            run_name=data.get("run_name"),
            tracking=tracking_config,
        )

    def get_job_dir(self) -> Path:
        """Get the job directory path based on patching config.

        Returns the path following Trident's naming convention:
        {output_dir}/{magnification}x_{patch_size}px_{overlap}px_overlap/
        """
        subdir = (
            f"{self.patching.magnification}x_"
            f"{self.patching.patch_size}px_"
            f"{self.patching.overlap}px_overlap"
        )
        return Path(self.output_dir) / subdir

    def get_features_dir(self) -> Path:
        """Get the features output directory.

        Returns the path where H5 feature files will be saved:
        {job_dir}/features_{encoder_name}/
        """
        return self.get_job_dir() / f"features_{self.encoder.name}"
