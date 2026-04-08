"""Tests for feature_extraction/foundational_models/config.py - Configuration dataclasses."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from feature_extraction.foundational_models.config import (
    InputConfig,
    SegmentationConfig,
    PatchingConfig,
    EncoderConfig,
    ProcessingConfig,
    ExtractionConfig,
)
from downstream.classification.multiple_instance_learning.training.config import TrackingConfig


# Module path for patching encoder_mapping functions (used in local imports)
ENCODER_MAPPING = "downstream.classification.multiple_instance_learning.training.encoder_mapping"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_wsi_dir(temp_dir):
    """Create a temporary WSI directory."""
    wsi_dir = temp_dir / "wsis"
    wsi_dir.mkdir()
    # Create a dummy WSI file
    (wsi_dir / "slide_001.svs").touch()
    return wsi_dir


@pytest.fixture
def temp_slide_list(temp_dir):
    """Create a temporary slide list file."""
    slide_list = temp_dir / "slides.txt"
    slide_list.write_text("slide_001\nslide_002\nslide_003\n")
    return slide_list


@pytest.fixture
def sample_input_config(temp_wsi_dir):
    """Create a sample InputConfig."""
    return InputConfig(wsi_dir=str(temp_wsi_dir))


@pytest.fixture
def sample_extraction_config(temp_wsi_dir, temp_dir):
    """Create a sample ExtractionConfig with mocked encoder validation."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()

    with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
        return ExtractionConfig(
            input=InputConfig(wsi_dir=str(temp_wsi_dir)),
            segmentation=SegmentationConfig(),
            patching=PatchingConfig(),
            encoder=EncoderConfig(name="uni_v2"),
            processing=ProcessingConfig(),
            output_dir=str(output_dir),
        )


# =============================================================================
# InputConfig Tests
# =============================================================================

class TestInputConfig:
    """Tests for InputConfig dataclass."""

    def test_input_config_defaults(self, temp_wsi_dir):
        """Test default values."""
        config = InputConfig(wsi_dir=str(temp_wsi_dir))

        assert config.wsi_dir == str(temp_wsi_dir)
        assert config.wsi_extensions == [".svs", ".ndpi", ".tiff"]
        assert config.slide_list is None
        assert config.search_nested is False

    def test_input_config_custom_extensions(self, temp_wsi_dir):
        """Test custom WSI extensions."""
        config = InputConfig(
            wsi_dir=str(temp_wsi_dir),
            wsi_extensions=[".svs", ".mrxs", "tif"],  # Note: "tif" without dot
        )

        # Should normalize extensions to start with dot
        assert ".svs" in config.wsi_extensions
        assert ".mrxs" in config.wsi_extensions
        assert ".tif" in config.wsi_extensions

    def test_input_config_missing_wsi_dir(self):
        """Test error when WSI directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="WSI directory not found"):
            InputConfig(wsi_dir="/nonexistent/path/wsis")

    def test_input_config_missing_slide_list(self, temp_wsi_dir):
        """Test error when slide list file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Slide list file not found"):
            InputConfig(
                wsi_dir=str(temp_wsi_dir),
                slide_list="/nonexistent/path/slides.txt",
            )

    def test_input_config_valid_slide_list(self, temp_wsi_dir, temp_slide_list):
        """Test valid slide list file."""
        config = InputConfig(
            wsi_dir=str(temp_wsi_dir),
            slide_list=str(temp_slide_list),
        )

        assert config.slide_list == str(temp_slide_list)

    def test_input_config_search_nested(self, temp_wsi_dir):
        """Test nested search setting."""
        config = InputConfig(
            wsi_dir=str(temp_wsi_dir),
            search_nested=True,
        )

        assert config.search_nested is True


# =============================================================================
# SegmentationConfig Tests
# =============================================================================

class TestSegmentationConfig:
    """Tests for SegmentationConfig dataclass."""

    def test_segmentation_config_defaults(self):
        """Test default values."""
        config = SegmentationConfig()

        assert config.model == "grandqc"
        assert config.magnification == 10
        assert config.batch_size == 16

    def test_segmentation_config_valid_models(self):
        """Test all valid segmentation models."""
        valid_models = ["grandqc", "hest", "otsu"]

        for model in valid_models:
            config = SegmentationConfig(model=model)
            assert config.model == model

    def test_segmentation_config_case_insensitive(self):
        """Test that model validation is case-insensitive."""
        # Should accept uppercase/mixed case
        config = SegmentationConfig(model="GRANDQC")
        assert config.model == "GRANDQC"  # Original case preserved

        config = SegmentationConfig(model="HeSt")
        assert config.model == "HeSt"

    def test_segmentation_config_invalid_model(self):
        """Test error for invalid segmentation model."""
        with pytest.raises(ValueError, match="Invalid segmentation model"):
            SegmentationConfig(model="invalid_model")

    def test_segmentation_config_custom_values(self):
        """Test custom configuration values."""
        config = SegmentationConfig(
            model="hest",
            magnification=5,
            batch_size=32,
        )

        assert config.model == "hest"
        assert config.magnification == 5
        assert config.batch_size == 32


# =============================================================================
# PatchingConfig Tests
# =============================================================================

class TestPatchingConfig:
    """Tests for PatchingConfig dataclass."""

    def test_patching_config_defaults(self):
        """Test default values."""
        config = PatchingConfig()

        assert config.magnification == 20
        assert config.patch_size == 256
        assert config.overlap == 0
        assert config.min_tissue_proportion == 0.0

    def test_patching_config_custom_values(self):
        """Test custom configuration values."""
        config = PatchingConfig(
            magnification=40,
            patch_size=512,
            overlap=64,
            min_tissue_proportion=0.5,
        )

        assert config.magnification == 40
        assert config.patch_size == 512
        assert config.overlap == 64
        assert config.min_tissue_proportion == 0.5

    def test_patching_config_invalid_patch_size(self):
        """Test error for invalid patch size."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            PatchingConfig(patch_size=0)

        with pytest.raises(ValueError, match="patch_size must be positive"):
            PatchingConfig(patch_size=-256)

    def test_patching_config_invalid_overlap(self):
        """Test error for negative overlap."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            PatchingConfig(overlap=-10)

    def test_patching_config_invalid_tissue_proportion(self):
        """Test error for invalid tissue proportion."""
        with pytest.raises(ValueError, match="min_tissue_proportion must be in"):
            PatchingConfig(min_tissue_proportion=-0.1)

        with pytest.raises(ValueError, match="min_tissue_proportion must be in"):
            PatchingConfig(min_tissue_proportion=1.5)

    def test_patching_config_boundary_tissue_proportion(self):
        """Test boundary values for tissue proportion."""
        # Should accept 0.0 and 1.0
        config = PatchingConfig(min_tissue_proportion=0.0)
        assert config.min_tissue_proportion == 0.0

        config = PatchingConfig(min_tissue_proportion=1.0)
        assert config.min_tissue_proportion == 1.0


# =============================================================================
# EncoderConfig Tests
# =============================================================================

class TestEncoderConfig:
    """Tests for EncoderConfig dataclass."""

    def test_encoder_config_defaults(self):
        """Test default values with mocked encoder validation."""
        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = EncoderConfig()

            assert config.name == "uni_v2"
            assert config.precision == "fp16"
            assert config.batch_size == 512

    def test_encoder_config_valid_encoders(self):
        """Test valid encoder configurations."""
        valid_encoders = [
            ("uni_v2", 1536),
            ("conch_v15", 768),
            ("gigapath", 1536),
            ("phikon", 768),
        ]

        for encoder_name, dim in valid_encoders:
            with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=dim):
                config = EncoderConfig(name=encoder_name)
                assert config.name == encoder_name

    def test_encoder_config_invalid_encoder(self):
        """Test error for invalid encoder name."""
        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=None), \
             patch(ENCODER_MAPPING + ".list_available_encoders", return_value=["uni_v2", "conch_v15", "gigapath"]):
            with pytest.raises(ValueError, match="Unknown encoder"):
                EncoderConfig(name="invalid_encoder")

    def test_encoder_config_precision_options(self):
        """Test precision options."""
        precisions = ["fp32", "fp16", "bf16"]

        for precision in precisions:
            with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
                config = EncoderConfig(precision=precision)
                assert config.precision == precision

    def test_encoder_config_custom_batch_size(self):
        """Test custom batch size."""
        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = EncoderConfig(batch_size=1024)
            assert config.batch_size == 1024


# =============================================================================
# ProcessingConfig Tests
# =============================================================================

class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""

    def test_processing_config_defaults(self):
        """Test default values."""
        config = ProcessingConfig()

        assert config.device == "cuda:0"
        assert config.num_workers == 4
        assert config.skip_errors is True
        assert config.resume is True

    def test_processing_config_custom_values(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            device="cuda:1",
            num_workers=8,
            skip_errors=False,
            resume=False,
        )

        assert config.device == "cuda:1"
        assert config.num_workers == 8
        assert config.skip_errors is False
        assert config.resume is False

    def test_processing_config_cpu_device(self):
        """Test CPU device configuration."""
        config = ProcessingConfig(device="cpu")
        assert config.device == "cpu"


# =============================================================================
# ExtractionConfig Tests
# =============================================================================

class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass."""

    def test_extraction_config_creates_output_dir(self, temp_wsi_dir, temp_dir):
        """Test that output directory is created."""
        output_dir = temp_dir / "new_output"
        assert not output_dir.exists()

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(),
                encoder=EncoderConfig(),
                processing=ProcessingConfig(),
                output_dir=str(output_dir),
            )

        assert output_dir.exists()

    def test_extraction_config_auto_run_name(self, temp_wsi_dir, temp_dir):
        """Test automatic run name generation."""
        output_dir = temp_dir / "output"

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(magnification=20),
                encoder=EncoderConfig(name="uni_v2"),
                processing=ProcessingConfig(),
                output_dir=str(output_dir),
            )

        assert config.run_name == "uni_v2_20x"

    def test_extraction_config_custom_run_name(self, temp_wsi_dir, temp_dir):
        """Test custom run name."""
        output_dir = temp_dir / "output"

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(),
                encoder=EncoderConfig(),
                processing=ProcessingConfig(),
                output_dir=str(output_dir),
                run_name="custom_run",
            )

        assert config.run_name == "custom_run"

    def test_extraction_config_default_tracking(self, sample_extraction_config):
        """Test default tracking configuration."""
        assert sample_extraction_config.tracking is not None
        assert sample_extraction_config.tracking.enabled is False
        assert sample_extraction_config.tracking.experiment_name == "feature-extraction"

    def test_extraction_config_custom_tracking(self, temp_wsi_dir, temp_dir):
        """Test custom tracking configuration."""
        output_dir = temp_dir / "output"

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(),
                encoder=EncoderConfig(),
                processing=ProcessingConfig(),
                output_dir=str(output_dir),
                tracking=TrackingConfig(
                    enabled=True,
                    experiment_name="custom-extraction",
                    backend="mlflow",
                ),
            )

        assert config.tracking.enabled is True
        assert config.tracking.experiment_name == "custom-extraction"

    def test_extraction_config_to_dict(self, sample_extraction_config):
        """Test conversion to dictionary."""
        config_dict = sample_extraction_config.to_dict()

        assert isinstance(config_dict, dict)
        assert "input" in config_dict
        assert "segmentation" in config_dict
        assert "patching" in config_dict
        assert "encoder" in config_dict
        assert "processing" in config_dict
        assert "output_dir" in config_dict

    def test_extraction_config_to_mlflow_params(self, sample_extraction_config):
        """Test conversion to MLflow params."""
        params = sample_extraction_config.to_mlflow_params()

        # Check encoder params
        assert params["encoder.name"] == "uni_v2"
        assert params["encoder.precision"] == "fp16"
        assert params["encoder.batch_size"] == 512

        # Check patching params
        assert params["patching.magnification"] == 20
        assert params["patching.patch_size"] == 256
        assert params["patching.overlap"] == 0

        # Check segmentation params
        assert params["segmentation.model"] == "grandqc"
        assert params["segmentation.magnification"] == 10

        # Check processing params
        assert params["processing.device"] == "cuda:0"
        assert params["processing.num_workers"] == 4

    def test_extraction_config_save_load_roundtrip(
        self, sample_extraction_config, temp_dir
    ):
        """Test JSON round-trip serialization."""
        config_path = temp_dir / "extraction_config.json"

        # Save config
        sample_extraction_config.save(str(config_path))
        assert config_path.exists()

        # Verify JSON content
        with open(config_path) as f:
            data = json.load(f)

        assert data["encoder"]["name"] == "uni_v2"
        assert data["patching"]["magnification"] == 20

        # Load config back (need to mock encoder validation)
        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            loaded_config = ExtractionConfig.load(str(config_path))

        # Verify loaded config matches original
        assert loaded_config.encoder.name == sample_extraction_config.encoder.name
        assert loaded_config.patching.magnification == sample_extraction_config.patching.magnification
        assert loaded_config.segmentation.model == sample_extraction_config.segmentation.model

    def test_extraction_config_load_with_tracking(self, temp_wsi_dir, temp_dir):
        """Test loading config with tracking settings."""
        output_dir = temp_dir / "output"
        config_path = temp_dir / "config.json"

        # Create config JSON with tracking
        config_data = {
            "input": {
                "wsi_dir": str(temp_wsi_dir),
            },
            "segmentation": {},
            "patching": {},
            "encoder": {"name": "uni_v2"},
            "processing": {},
            "output_dir": str(output_dir),
            "tracking": {
                "enabled": True,
                "experiment_name": "test-extraction",
                "backend": "wandb",
            },
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load config
        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            loaded_config = ExtractionConfig.load(str(config_path))

        assert loaded_config.tracking.enabled is True
        assert loaded_config.tracking.experiment_name == "test-extraction"
        assert loaded_config.tracking.backend == "wandb"

    def test_extraction_config_get_job_dir(self, sample_extraction_config):
        """Test get_job_dir path generation."""
        job_dir = sample_extraction_config.get_job_dir()

        # Should follow Trident naming convention
        expected_subdir = "20x_256px_0px_overlap"
        assert expected_subdir in str(job_dir)
        assert isinstance(job_dir, Path)

    def test_extraction_config_get_job_dir_custom_params(self, temp_wsi_dir, temp_dir):
        """Test get_job_dir with custom patching parameters."""
        output_dir = temp_dir / "output"

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(
                    magnification=40,
                    patch_size=512,
                    overlap=128,
                ),
                encoder=EncoderConfig(),
                processing=ProcessingConfig(),
                output_dir=str(output_dir),
            )

        job_dir = config.get_job_dir()
        expected_subdir = "40x_512px_128px_overlap"
        assert expected_subdir in str(job_dir)

    def test_extraction_config_get_features_dir(self, sample_extraction_config):
        """Test get_features_dir path generation."""
        features_dir = sample_extraction_config.get_features_dir()

        # Should include encoder name
        assert "features_uni_v2" in str(features_dir)
        assert isinstance(features_dir, Path)

        # Should be a subdirectory of job_dir
        job_dir = sample_extraction_config.get_job_dir()
        assert str(features_dir).startswith(str(job_dir))

    def test_extraction_config_get_features_dir_different_encoder(
        self, temp_wsi_dir, temp_dir
    ):
        """Test get_features_dir with different encoder."""
        output_dir = temp_dir / "output"

        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=768):
            config = ExtractionConfig(
                input=InputConfig(wsi_dir=str(temp_wsi_dir)),
                segmentation=SegmentationConfig(),
                patching=PatchingConfig(),
                encoder=EncoderConfig(name="conch_v15"),
                processing=ProcessingConfig(),
                output_dir=str(output_dir),
            )

        features_dir = config.get_features_dir()
        assert "features_conch_v15" in str(features_dir)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestConfigEdgeCases:
    """Edge cases and integration tests for configs."""

    def test_input_config_extensions_without_dot(self, temp_wsi_dir):
        """Test that extensions without dots are normalized."""
        config = InputConfig(
            wsi_dir=str(temp_wsi_dir),
            wsi_extensions=["svs", "ndpi", ".tiff"],
        )

        assert all(ext.startswith(".") for ext in config.wsi_extensions)
        assert ".svs" in config.wsi_extensions
        assert ".ndpi" in config.wsi_extensions
        assert ".tiff" in config.wsi_extensions

    def test_config_serialization_preserves_types(self, sample_extraction_config, temp_dir):
        """Test that serialization preserves correct types."""
        config_path = temp_dir / "config.json"
        sample_extraction_config.save(str(config_path))

        with open(config_path) as f:
            data = json.load(f)

        # Check numeric types
        assert isinstance(data["patching"]["magnification"], int)
        assert isinstance(data["patching"]["min_tissue_proportion"], float)
        assert isinstance(data["processing"]["skip_errors"], bool)

    def test_config_load_minimal(self, temp_wsi_dir, temp_dir):
        """Test loading config with only required fields."""
        output_dir = temp_dir / "output"
        config_path = temp_dir / "minimal_config.json"

        # Minimal config with only required fields
        config_data = {
            "input": {
                "wsi_dir": str(temp_wsi_dir),
            },
            "output_dir": str(output_dir),
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load config - should use defaults for missing fields
        with patch(ENCODER_MAPPING + ".get_encoder_dim", return_value=1536):
            loaded_config = ExtractionConfig.load(str(config_path))

        # Check defaults are applied
        assert loaded_config.segmentation.model == "grandqc"
        assert loaded_config.patching.magnification == 20
        assert loaded_config.encoder.name == "uni_v2"
        assert loaded_config.processing.device == "cuda:0"
