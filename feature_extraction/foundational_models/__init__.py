"""Feature extraction module for Trident-based WSI processing.

This module provides config-driven feature extraction from whole-slide images (WSIs)
using Trident encoders, outputting H5 files compatible with IbinSina's MIL training pipeline.

Pipeline: WSI → Segmentation → Patching → Feature Extraction → H5

Example usage:
    from extraction import ExtractionConfig, TridentExtractor

    config = ExtractionConfig.load("configs/extraction_config.json")
    extractor = TridentExtractor(config)
    results = extractor.run()
"""

from .config import (
    ExtractionConfig,
    InputConfig,
    SegmentationConfig,
    PatchingConfig,
    EncoderConfig,
    ProcessingConfig,
)
from .extractor import TridentExtractor

__all__ = [
    "ExtractionConfig",
    "InputConfig",
    "SegmentationConfig",
    "PatchingConfig",
    "EncoderConfig",
    "ProcessingConfig",
    "TridentExtractor",
]
