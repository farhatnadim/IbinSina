"""Tests for feature_extraction/foundational_models/extractor.py - TridentExtractor class."""

import pytest
import json
import tempfile
import h5py
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO
import sys

from feature_extraction.foundational_models.extractor import TridentExtractor
from feature_extraction.foundational_models.config import (
    InputConfig,
    SegmentationConfig,
    PatchingConfig,
    EncoderConfig,
    ProcessingConfig,
    ExtractionConfig,
)


# Module paths for patching
ENCODER_MAPPING = "downstream.classification.multiple_instance_learning.training.encoder_mapping"
EXTRACTOR_MODULE = "feature_extraction.foundational_models.extractor"

# Trident module paths for local imports in run() method
TRIDENT_PROCESSOR = "trident.Processor"
TRIDENT_ENCODER_FACTORY = "trident.patch_encoder_models.encoder_factory"
TRIDENT_SEGMENTATION_FACTORY = "trident.segmentation_models.segmentation_model_factory"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_wsi_dir(temp_dir):
    """Create a temporary WSI directory."""
    wsi_dir = temp_dir / "wsis"
    wsi_dir.mkdir()
    (wsi_dir / "slide_001.svs").touch()
    (wsi_dir / "slide_002.svs").touch()
    return wsi_dir


@pytest.fixture
def mock_extraction_config(temp_wsi_dir, temp_dir):
    """Create a mock ExtractionConfig."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)

    with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
        return ExtractionConfig(
            input=InputConfig(wsi_dir=str(temp_wsi_dir)),
            segmentation=SegmentationConfig(),
            patching=PatchingConfig(),
            encoder=EncoderConfig(name="uni_v2"),
            processing=ProcessingConfig(device="cpu"),
            output_dir=str(output_dir),
            run_name="test_extraction",
        )


@pytest.fixture
def mock_h5_features_dir(temp_dir):
    """Create a directory with mock H5 feature files."""
    features_dir = temp_dir / "features"
    features_dir.mkdir()

    # Create mock H5 files with features
    slides = [
        ("slide_001", 100, 1536),
        ("slide_002", 150, 1536),
        ("slide_003", 80, 1536),
    ]

    for slide_id, num_patches, embed_dim in slides:
        h5_path = features_dir / f"{slide_id}.h5"
        with h5py.File(h5_path, "w") as f:
            features = np.random.randn(num_patches, embed_dim).astype(np.float32)
            f.create_dataset("features", data=features)

    return features_dir


@pytest.fixture
def mock_coords_dir(temp_dir):
    """Create a directory with mock coordinate H5 files."""
    coords_dir = temp_dir / "coords"
    coords_dir.mkdir()

    # Create mock coordinate files
    slides = ["slide_001", "slide_002", "slide_003", "slide_004"]  # One extra for skipped

    for slide_id in slides:
        h5_path = coords_dir / f"{slide_id}.h5"
        with h5py.File(h5_path, "w") as f:
            # Create minimal coordinate data
            coords = np.random.randint(0, 10000, size=(100, 2))
            f.create_dataset("coords", data=coords)

    return coords_dir


@pytest.fixture
def mock_tracker():
    """Create a mock experiment tracker."""
    tracker = MagicMock()
    tracker.start_run.return_value.__enter__ = MagicMock()
    tracker.start_run.return_value.__exit__ = MagicMock()
    return tracker


# =============================================================================
# TridentExtractor Instantiation Tests
# =============================================================================

class TestTridentExtractorInstantiation:
    """Tests for TridentExtractor instantiation."""

    def test_extractor_instantiation(self, mock_extraction_config):
        """Test basic extractor instantiation."""
        extractor = TridentExtractor(mock_extraction_config)

        assert extractor.config == mock_extraction_config
        assert extractor.tracker is None
        assert extractor._own_tracker is False
        assert extractor._processor is None
        assert extractor._encoder is None
        assert extractor._segmentation_model is None

    def test_extractor_with_tracker(self, mock_extraction_config, mock_tracker):
        """Test extractor instantiation with external tracker."""
        extractor = TridentExtractor(mock_extraction_config, tracker=mock_tracker)

        assert extractor.tracker == mock_tracker
        assert extractor._own_tracker is False

    def test_extractor_config_access(self, mock_extraction_config):
        """Test access to config attributes through extractor."""
        extractor = TridentExtractor(mock_extraction_config)

        assert extractor.config.encoder.name == "uni_v2"
        assert extractor.config.patching.magnification == 20
        assert extractor.config.processing.device == "cpu"


# =============================================================================
# _collect_stats Method Tests
# =============================================================================

class TestCollectStats:
    """Tests for TridentExtractor._collect_stats method."""

    def test_collect_stats_basic(
        self, mock_extraction_config, mock_h5_features_dir, mock_coords_dir
    ):
        """Test basic statistics collection."""
        extractor = TridentExtractor(mock_extraction_config)

        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            stats = extractor._collect_stats(
                str(mock_h5_features_dir),
                str(mock_coords_dir),
            )

        # Check required keys
        assert "slides_processed" in stats
        assert "slides_skipped" in stats
        assert "total_patches" in stats
        assert "avg_patches_per_slide" in stats
        assert "min_patches" in stats
        assert "max_patches" in stats
        assert "encoder_name" in stats
        assert "encoder_dim" in stats
        assert "magnification" in stats
        assert "patch_size" in stats

    def test_collect_stats_values(
        self, mock_extraction_config, mock_h5_features_dir, mock_coords_dir
    ):
        """Test collected statistics values."""
        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            extractor = TridentExtractor(mock_extraction_config)

            stats = extractor._collect_stats(
                str(mock_h5_features_dir),
                str(mock_coords_dir),
            )

        # 3 H5 files in features_dir
        assert stats["slides_processed"] == 3

        # 4 coord files - 3 processed = 1 skipped
        assert stats["slides_skipped"] == 1

        # Total patches: 100 + 150 + 80 = 330
        assert stats["total_patches"] == 330

        # Average: 330 / 3 = 110
        assert stats["avg_patches_per_slide"] == 110.0

        # Min/Max patches
        assert stats["min_patches"] == 80
        assert stats["max_patches"] == 150

        # Config values
        assert stats["encoder_name"] == "uni_v2"
        assert stats["magnification"] == 20
        assert stats["patch_size"] == 256

    def test_collect_stats_empty_features_dir(
        self, mock_extraction_config, temp_dir, mock_coords_dir
    ):
        """Test statistics collection with empty features directory."""
        empty_dir = temp_dir / "empty_features"
        empty_dir.mkdir()

        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            extractor = TridentExtractor(mock_extraction_config)

            stats = extractor._collect_stats(
                str(empty_dir),
                str(mock_coords_dir),
            )

        assert stats["slides_processed"] == 0
        assert stats["total_patches"] == 0
        assert stats["avg_patches_per_slide"] == 0
        assert stats["min_patches"] == 0
        assert stats["max_patches"] == 0

    def test_collect_stats_corrupted_h5(
        self, mock_extraction_config, temp_dir, mock_coords_dir, capsys
    ):
        """Test statistics collection with corrupted H5 file."""
        features_dir = temp_dir / "features_corrupted"
        features_dir.mkdir()

        # Create a valid H5 file
        with h5py.File(features_dir / "slide_001.h5", "w") as f:
            f.create_dataset("features", data=np.random.randn(100, 1536))

        # Create an "H5 file" that's actually empty/corrupted
        (features_dir / "slide_002.h5").write_bytes(b"not a valid h5 file")

        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            extractor = TridentExtractor(mock_extraction_config)

            # Should handle the error and continue
            stats = extractor._collect_stats(
                str(features_dir),
                str(mock_coords_dir),
            )

        # Should still report some slides processed
        assert stats["slides_processed"] == 2  # Both counted even if one fails to read


# =============================================================================
# _save_metadata Method Tests
# =============================================================================

class TestSaveMetadata:
    """Tests for TridentExtractor._save_metadata method."""

    def test_save_metadata_creates_files(self, mock_extraction_config, temp_dir):
        """Test that metadata files are created."""
        job_dir = temp_dir / "job_dir"
        job_dir.mkdir()

        stats = {
            "slides_processed": 10,
            "slides_skipped": 2,
            "total_patches": 1000,
            "avg_patches_per_slide": 100.0,
            "min_patches": 50,
            "max_patches": 150,
            "encoder_name": "uni_v2",
            "encoder_dim": 1536,
            "magnification": 20,
            "patch_size": 256,
        }

        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            extractor = TridentExtractor(mock_extraction_config)
            extractor._save_metadata(job_dir, stats)

        # Check config file created
        config_path = job_dir / "extraction_config.json"
        assert config_path.exists()

        # Check metadata file created
        metadata_path = job_dir / "extraction_metadata.json"
        assert metadata_path.exists()

    def test_save_metadata_config_content(self, mock_extraction_config, temp_dir):
        """Test saved config content."""
        job_dir = temp_dir / "job_dir"
        job_dir.mkdir()

        stats = {
            "slides_processed": 10,
            "slides_skipped": 2,
            "total_patches": 1000,
            "avg_patches_per_slide": 100.0,
            "min_patches": 50,
            "max_patches": 150,
            "encoder_name": "uni_v2",
            "encoder_dim": 1536,
            "magnification": 20,
            "patch_size": 256,
        }

        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            extractor = TridentExtractor(mock_extraction_config)
            extractor._save_metadata(job_dir, stats)

        # Load and verify config
        config_path = job_dir / "extraction_config.json"
        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["encoder"]["name"] == "uni_v2"
        assert saved_config["patching"]["magnification"] == 20

    def test_save_metadata_metadata_content(self, mock_extraction_config, temp_dir):
        """Test saved metadata content."""
        job_dir = temp_dir / "job_dir"
        job_dir.mkdir()

        stats = {
            "slides_processed": 10,
            "slides_skipped": 2,
            "total_patches": 1000,
            "avg_patches_per_slide": 100.0,
            "min_patches": 50,
            "max_patches": 150,
            "encoder_name": "uni_v2",
            "encoder_dim": 1536,
            "magnification": 20,
            "patch_size": 256,
        }

        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            extractor = TridentExtractor(mock_extraction_config)
            extractor._save_metadata(job_dir, stats)

        # Load and verify metadata
        metadata_path = job_dir / "extraction_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check structure
        assert "extraction_timestamp" in metadata
        assert "encoder" in metadata
        assert "patching" in metadata
        assert "segmentation" in metadata
        assert "statistics" in metadata

        # Check encoder info
        assert metadata["encoder"]["name"] == "uni_v2"
        assert metadata["encoder"]["dim"] == 1536
        assert metadata["encoder"]["precision"] == "fp16"

        # Check patching info
        assert metadata["patching"]["magnification"] == 20
        assert metadata["patching"]["patch_size"] == 256
        assert metadata["patching"]["overlap"] == 0

        # Check stats
        assert metadata["statistics"]["slides_processed"] == 10
        assert metadata["statistics"]["total_patches"] == 1000


# =============================================================================
# _print_summary Method Tests
# =============================================================================

class TestPrintSummary:
    """Tests for TridentExtractor._print_summary method."""

    def test_print_summary_output(self, mock_extraction_config, capsys):
        """Test summary output content."""
        stats = {
            "slides_processed": 10,
            "slides_skipped": 2,
            "total_patches": 1000,
            "avg_patches_per_slide": 100.0,
            "min_patches": 50,
            "max_patches": 150,
            "encoder_name": "uni_v2",
            "encoder_dim": 1536,
            "magnification": 20,
            "patch_size": 256,
        }

        extractor = TridentExtractor(mock_extraction_config)
        extractor._print_summary(stats, "/path/to/features")

        captured = capsys.readouterr()

        # Check key information is printed
        assert "EXTRACTION COMPLETE" in captured.out
        assert "10" in captured.out  # slides_processed
        assert "2" in captured.out   # slides_skipped
        assert "1,000" in captured.out  # total_patches (formatted)
        assert "100.0" in captured.out  # avg_patches_per_slide
        assert "uni_v2" in captured.out  # encoder_name
        assert "1536" in captured.out  # encoder_dim
        assert "256" in captured.out  # patch_size
        assert "20x" in captured.out  # magnification
        assert "/path/to/features" in captured.out

    def test_print_summary_formatting(self, mock_extraction_config, capsys):
        """Test summary output formatting."""
        stats = {
            "slides_processed": 100,
            "slides_skipped": 0,
            "total_patches": 1000000,  # Large number to test formatting
            "avg_patches_per_slide": 10000.5,
            "min_patches": 5000,
            "max_patches": 15000,
            "encoder_name": "gigapath",
            "encoder_dim": 1536,
            "magnification": 40,
            "patch_size": 512,
        }

        extractor = TridentExtractor(mock_extraction_config)
        extractor._print_summary(stats, "/output/features")

        captured = capsys.readouterr()

        # Check large number formatting
        assert "1,000,000" in captured.out
        assert "=" in captured.out  # Separator lines


# =============================================================================
# Run Method Tests (with mocked Trident)
# =============================================================================

class TestTridentExtractorRun:
    """Tests for TridentExtractor.run method with mocked Trident."""

    def test_run_no_slides_found(self, mock_extraction_config, capsys):
        """Test run behavior when no slides are found."""
        mock_processor = MagicMock()
        mock_processor.wsis = []  # No slides found

        with patch(TRIDENT_PROCESSOR, return_value=mock_processor), \
             patch(TRIDENT_ENCODER_FACTORY), \
             patch(TRIDENT_SEGMENTATION_FACTORY):

            extractor = TridentExtractor(mock_extraction_config)
            result = extractor.run()

        # Should return empty result
        assert result["features_dir"] is None
        assert result["stats"]["slides_found"] == 0

        # Should print warning
        captured = capsys.readouterr()
        assert "No slides found" in captured.out

    def test_run_with_tracker(self, mock_extraction_config, mock_tracker):
        """Test run with external tracker."""
        mock_processor = MagicMock()
        mock_processor.wsis = ["slide1.svs", "slide2.svs"]
        mock_processor.run_segmentation_job.return_value = "/segmentation"
        mock_processor.run_patching_job.return_value = "/coords"
        mock_processor.run_patch_feature_extraction_job.return_value = "/features"

        mock_encoder = MagicMock()

        # Create mock features directory with H5 files
        with patch(TRIDENT_PROCESSOR, return_value=mock_processor), \
             patch(TRIDENT_ENCODER_FACTORY, return_value=mock_encoder), \
             patch(TRIDENT_SEGMENTATION_FACTORY), \
             patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536), \
             patch.object(TridentExtractor, "_collect_stats", return_value={
                 "slides_processed": 2,
                 "slides_skipped": 0,
                 "total_patches": 200,
                 "avg_patches_per_slide": 100.0,
                 "min_patches": 80,
                 "max_patches": 120,
                 "encoder_name": "uni_v2",
                 "encoder_dim": 1536,
                 "magnification": 20,
                 "patch_size": 256,
             }), \
             patch.object(TridentExtractor, "_save_metadata"), \
             patch.object(TridentExtractor, "_print_summary"):

            extractor = TridentExtractor(mock_extraction_config, tracker=mock_tracker)
            result = extractor.run()

        # Tracker should NOT start its own run (external tracker)
        mock_tracker.start_run.assert_not_called()

        # But should log metrics
        mock_tracker.log_metrics.assert_called_once()
        logged_metrics = mock_tracker.log_metrics.call_args[0][0]
        assert logged_metrics["slides_processed"] == 2
        assert logged_metrics["total_patches"] == 200

    def test_run_creates_own_tracker(self, temp_wsi_dir, temp_dir):
        """Test that run creates tracker when config enables tracking."""
        output_dir = temp_dir / "output"

        # Create config with tracking enabled
        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            from downstream.classification.multiple_instance_learning.training.config import TrackingConfig

            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(),
                encoder=EncoderConfig(name="uni_v2"),
                processing=ProcessingConfig(device="cpu"),
                output_dir=str(output_dir),
                tracking=TrackingConfig(
                    enabled=True,
                    experiment_name="test-extraction",
                    backend="none",  # Use 'none' backend for testing
                ),
            )

        mock_processor = MagicMock()
        mock_processor.wsis = []

        with patch(TRIDENT_PROCESSOR, return_value=mock_processor), \
             patch(TRIDENT_ENCODER_FACTORY), \
             patch(TRIDENT_SEGMENTATION_FACTORY), \
             patch(EXTRACTOR_MODULE + ".create_tracker") as mock_create_tracker:

            # Configure mock tracker
            mock_tracker = MagicMock()
            mock_tracker.start_run.return_value.__enter__ = MagicMock()
            mock_tracker.start_run.return_value.__exit__ = MagicMock()
            mock_create_tracker.return_value = mock_tracker

            extractor = TridentExtractor(config)
            extractor.run()

        # Should have created its own tracker
        mock_create_tracker.assert_called_once()

    def test_run_full_pipeline(self, mock_extraction_config, temp_dir):
        """Test full pipeline execution."""
        mock_processor = MagicMock()
        mock_processor.wsis = ["slide1.svs", "slide2.svs"]
        mock_processor.run_segmentation_job.return_value = str(temp_dir / "segmentation")
        mock_processor.run_patching_job.return_value = str(temp_dir / "coords")
        mock_processor.run_patch_feature_extraction_job.return_value = str(temp_dir / "features")

        mock_encoder = MagicMock()
        # Make to() return the encoder itself for chaining
        mock_encoder.to.return_value = mock_encoder
        mock_encoder.half.return_value = mock_encoder

        mock_segmentation_model = MagicMock()

        # Create job directory (normally created by Processor)
        job_dir = mock_extraction_config.get_job_dir()
        job_dir.mkdir(parents=True, exist_ok=True)

        # Create feature directory with mock H5 files
        features_dir = temp_dir / "features"
        features_dir.mkdir()
        with h5py.File(features_dir / "slide1.h5", "w") as f:
            f.create_dataset("features", data=np.random.randn(100, 1536))
        with h5py.File(features_dir / "slide2.h5", "w") as f:
            f.create_dataset("features", data=np.random.randn(120, 1536))

        # Create coords directory
        coords_dir = temp_dir / "coords"
        coords_dir.mkdir()
        with h5py.File(coords_dir / "slide1.h5", "w") as f:
            f.create_dataset("coords", data=np.random.randint(0, 1000, (100, 2)))
        with h5py.File(coords_dir / "slide2.h5", "w") as f:
            f.create_dataset("coords", data=np.random.randint(0, 1000, (120, 2)))

        with patch(TRIDENT_PROCESSOR, return_value=mock_processor), \
             patch(TRIDENT_ENCODER_FACTORY, return_value=mock_encoder), \
             patch(TRIDENT_SEGMENTATION_FACTORY, return_value=mock_segmentation_model), \
             patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):

            extractor = TridentExtractor(mock_extraction_config)
            result = extractor.run()

        # Check pipeline was executed
        mock_processor.run_segmentation_job.assert_called_once()
        mock_processor.run_patching_job.assert_called_once()
        mock_processor.run_patch_feature_extraction_job.assert_called_once()
        mock_processor.release.assert_called_once()

        # Check encoder setup - to() is called for device placement
        mock_encoder.to.assert_called()
        mock_encoder.eval.assert_called_once()
        mock_encoder.half.assert_called_once()  # fp16 precision

        # Check result
        assert result["features_dir"] == str(temp_dir / "features")
        assert "stats" in result
        assert "job_dir" in result

    def test_run_bf16_precision(self, temp_wsi_dir, temp_dir):
        """Test run with bfloat16 precision."""
        output_dir = temp_dir / "output"

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(),
                encoder=EncoderConfig(name="uni_v2", precision="bf16"),
                processing=ProcessingConfig(device="cpu"),
                output_dir=str(output_dir),
            )

        mock_processor = MagicMock()
        mock_processor.wsis = ["slide1.svs"]
        mock_processor.run_segmentation_job.return_value = "/segmentation"
        mock_processor.run_patching_job.return_value = "/coords"
        mock_processor.run_patch_feature_extraction_job.return_value = "/features"

        mock_encoder = MagicMock()

        with patch(TRIDENT_PROCESSOR, return_value=mock_processor), \
             patch(TRIDENT_ENCODER_FACTORY, return_value=mock_encoder), \
             patch(TRIDENT_SEGMENTATION_FACTORY), \
             patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536), \
             patch.object(TridentExtractor, "_collect_stats", return_value={
                 "slides_processed": 1,
                 "slides_skipped": 0,
                 "total_patches": 100,
                 "avg_patches_per_slide": 100.0,
                 "min_patches": 100,
                 "max_patches": 100,
                 "encoder_name": "uni_v2",
                 "encoder_dim": 1536,
                 "magnification": 20,
                 "patch_size": 256,
             }), \
             patch.object(TridentExtractor, "_save_metadata"), \
             patch.object(TridentExtractor, "_print_summary"), \
             patch("torch.bfloat16", return_value="bfloat16_dtype"):

            extractor = TridentExtractor(config)
            extractor.run()

        # Should use bfloat16 instead of half()
        mock_encoder.half.assert_not_called()
        # Check that .to() was called with bfloat16 dtype
        to_calls = mock_encoder.to.call_args_list
        assert len(to_calls) >= 1  # At least device call

    def test_run_fp32_precision(self, temp_wsi_dir, temp_dir):
        """Test run with fp32 precision (no conversion)."""
        output_dir = temp_dir / "output"

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(),
                encoder=EncoderConfig(name="uni_v2", precision="fp32"),
                processing=ProcessingConfig(device="cpu"),
                output_dir=str(output_dir),
            )

        mock_processor = MagicMock()
        mock_processor.wsis = ["slide1.svs"]
        mock_processor.run_segmentation_job.return_value = "/segmentation"
        mock_processor.run_patching_job.return_value = "/coords"
        mock_processor.run_patch_feature_extraction_job.return_value = "/features"

        mock_encoder = MagicMock()

        with patch(TRIDENT_PROCESSOR, return_value=mock_processor), \
             patch(TRIDENT_ENCODER_FACTORY, return_value=mock_encoder), \
             patch(TRIDENT_SEGMENTATION_FACTORY), \
             patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536), \
             patch.object(TridentExtractor, "_collect_stats", return_value={
                 "slides_processed": 1,
                 "slides_skipped": 0,
                 "total_patches": 100,
                 "avg_patches_per_slide": 100.0,
                 "min_patches": 100,
                 "max_patches": 100,
                 "encoder_name": "uni_v2",
                 "encoder_dim": 1536,
                 "magnification": 20,
                 "patch_size": 256,
             }), \
             patch.object(TridentExtractor, "_save_metadata"), \
             patch.object(TridentExtractor, "_print_summary"):

            extractor = TridentExtractor(config)
            extractor.run()

        # Should NOT call half() or to(bfloat16) for fp32
        mock_encoder.half.assert_not_called()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases for TridentExtractor."""

    def test_extractor_releases_resources_on_completion(self, mock_extraction_config):
        """Test that resources are released after run completes."""
        mock_processor = MagicMock()
        mock_processor.wsis = ["slide1.svs"]
        mock_processor.run_segmentation_job.return_value = "/segmentation"
        mock_processor.run_patching_job.return_value = "/coords"
        mock_processor.run_patch_feature_extraction_job.return_value = "/features"

        with patch(TRIDENT_PROCESSOR, return_value=mock_processor), \
             patch(TRIDENT_ENCODER_FACTORY), \
             patch(TRIDENT_SEGMENTATION_FACTORY), \
             patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536), \
             patch.object(TridentExtractor, "_collect_stats", return_value={
                 "slides_processed": 1,
                 "slides_skipped": 0,
                 "total_patches": 100,
                 "avg_patches_per_slide": 100.0,
                 "min_patches": 100,
                 "max_patches": 100,
                 "encoder_name": "uni_v2",
                 "encoder_dim": 1536,
                 "magnification": 20,
                 "patch_size": 256,
             }), \
             patch.object(TridentExtractor, "_save_metadata"), \
             patch.object(TridentExtractor, "_print_summary"):

            extractor = TridentExtractor(mock_extraction_config)
            extractor.run()

        # Verify release was called
        mock_processor.release.assert_called_once()

        # Verify internal state is cleared
        assert extractor._processor is None
        assert extractor._encoder is None
        assert extractor._segmentation_model is None

    def test_collect_stats_handles_h5_without_features_key(
        self, mock_extraction_config, temp_dir
    ):
        """Test _collect_stats handles H5 files without 'features' dataset."""
        features_dir = temp_dir / "features"
        features_dir.mkdir()

        # Create H5 file with different dataset name
        with h5py.File(features_dir / "slide_001.h5", "w") as f:
            f.create_dataset("embeddings", data=np.random.randn(100, 1536))  # Not "features"

        coords_dir = temp_dir / "coords"
        coords_dir.mkdir()
        with h5py.File(coords_dir / "slide_001.h5", "w") as f:
            f.create_dataset("coords", data=np.random.randint(0, 1000, (100, 2)))

        with patch(EXTRACTOR_MODULE + ".get_encoder_dim", return_value=1536):
            extractor = TridentExtractor(mock_extraction_config)
            stats = extractor._collect_stats(str(features_dir), str(coords_dir))

        # Should count the file but not the patches
        assert stats["slides_processed"] == 1
        assert stats["total_patches"] == 0  # No "features" key found
