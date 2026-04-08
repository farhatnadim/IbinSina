#!/usr/bin/env python3
"""
Encoder dimension mapping and utilities.

Consolidates encoder information from MIL-Lab and Trident packages,
providing dimension lookups, model name parsing, and validation.

This module supports two workflows:
1. Pre-extracted H5 features: Encoder metadata is logged for tracking
2. On-the-fly extraction: Encoder is loaded via Trident for inference
"""

import re
from typing import Optional, List, Tuple

# =============================================================================
# Encoder Dimension Mapping
# =============================================================================

# Maps encoder names to their output embedding dimensions
# Sources: MIL-Lab (src/constants.py), Trident (patch_encoder_models)
ENCODER_DIM_MAPPING = {
    # -----------------------------------------------------------------
    # MIL-Lab encoders (from src/constants.py ENCODER_DIM_MAPPING)
    # -----------------------------------------------------------------
    # UNI family
    'uni': 1024,
    'uni_v1': 1024,
    'uni_v2': 1536,
    # CONCH family
    'conch_v1': 512,
    'conch_v15': 768,
    # Virchow family
    'virchow': 2560,
    'virchow2': 2560,
    # Other foundation models
    'gigapath': 1536,
    'ctranspath': 768,
    'phikon': 768,
    'phikon_v2': 1024,
    'hoptimus0': 1536,
    'hoptimus1': 1536,
    'resnet50': 1024,
    'musk': 1024,
    # -----------------------------------------------------------------
    # Additional Trident encoders
    # -----------------------------------------------------------------
    'h0_mini': 1536,
    'openmidnight': 1536,
    'gpfm': 1024,
    'hibou_l': 1024,
    'hibou_b': 768,
    'midnight12k': 1536,
    'genbio_pathfm': 1024,
    # Kaiko family
    'kaiko_vitb8': 768,
    'kaiko_vitb16': 768,
    'kaiko_vits8': 384,
    'kaiko_vits16': 384,
    'kaiko_vitl14': 1024,
    # Lunit
    'lunit_vits8': 384,
    # -----------------------------------------------------------------
    # Aliases (common alternative names)
    # -----------------------------------------------------------------
    'uni1': 1024,
    'uni2': 1536,
    'conch': 512,        # Default to v1
    'conch_v1.5': 768,   # Alternative for v15
    'hoptimus': 1536,    # Default to h0
    'h0': 1536,          # Alias for hoptimus0
    'h1': 1536,          # Alias for hoptimus1
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_encoder_dim(name: str) -> Optional[int]:
    """
    Get the embedding dimension for an encoder.

    Args:
        name: Encoder name (case-insensitive, underscores/hyphens normalized)

    Returns:
        Embedding dimension, or None if encoder not found

    Examples:
        >>> get_encoder_dim('uni_v2')
        1536
        >>> get_encoder_dim('UNI-V2')
        1536
        >>> get_encoder_dim('unknown_encoder')
        None
    """
    # Normalize name: lowercase, replace hyphens with underscores
    normalized = name.lower().replace('-', '_')
    return ENCODER_DIM_MAPPING.get(normalized)


def parse_encoder_from_model_name(model_name: str) -> Optional[str]:
    """
    Extract encoder name from MIL-Lab model naming convention.

    Model name format: <model>.<config>.<encoder>.<pretrain>
    Examples:
        - abmil.base.uni_v2.none -> uni_v2
        - transmil.base.conch_v15.pc108-24k -> conch_v15
        - clam.sb.gigapath.none -> gigapath

    Args:
        model_name: Full model name string

    Returns:
        Encoder name if parseable, None otherwise
    """
    parts = model_name.split('.')
    if len(parts) >= 3:
        return parts[2]
    return None


def validate_encoder_consistency(
    model_name: str,
    config_encoder: Optional[str],
    actual_dim: Optional[int],
) -> Tuple[bool, str]:
    """
    Validate that encoder specifications are consistent.

    Checks for mismatches between:
    - Encoder name in model_name (e.g., abmil.base.uni_v2.none)
    - Encoder name in config (if specified)
    - Actual feature dimension (if provided)

    Args:
        model_name: Full MIL model name
        config_encoder: Encoder name from EncoderConfig (optional)
        actual_dim: Actual embedding dimension from loaded features (optional)

    Returns:
        Tuple of (is_valid, message)

    Examples:
        >>> validate_encoder_consistency('abmil.base.uni_v2.none', 'uni_v2', 1536)
        (True, 'Encoder configuration is consistent')

        >>> validate_encoder_consistency('abmil.base.uni_v2.none', 'conch_v15', None)
        (False, "Encoder mismatch: model_name specifies 'uni_v2' but config specifies 'conch_v15'")
    """
    # Parse encoder from model name
    model_encoder = parse_encoder_from_model_name(model_name)

    # Check model_name vs config_encoder
    if config_encoder and model_encoder:
        # Normalize for comparison
        norm_config = config_encoder.lower().replace('-', '_')
        norm_model = model_encoder.lower().replace('-', '_')
        if norm_config != norm_model:
            return (
                False,
                f"Encoder mismatch: model_name specifies '{model_encoder}' "
                f"but config specifies '{config_encoder}'"
            )

    # Get expected dimension
    encoder_name = config_encoder or model_encoder
    if encoder_name:
        expected_dim = get_encoder_dim(encoder_name)

        # Check actual_dim vs expected_dim
        if actual_dim is not None and expected_dim is not None:
            if actual_dim != expected_dim:
                return (
                    False,
                    f"Dimension mismatch for encoder '{encoder_name}': "
                    f"expected {expected_dim}, got {actual_dim}"
                )

    return (True, 'Encoder configuration is consistent')


def list_available_encoders() -> List[str]:
    """
    List all available encoder names.

    Returns:
        Sorted list of encoder names (excluding aliases)
    """
    # Filter out obvious aliases (those with digits at end like uni1, uni2)
    primary_encoders = []
    for name in ENCODER_DIM_MAPPING.keys():
        # Skip aliases that are just shortened versions
        if name in ('uni1', 'uni2', 'conch', 'hoptimus', 'h0', 'h1', 'conch_v1.5'):
            continue
        primary_encoders.append(name)
    return sorted(primary_encoders)


def get_encoder_info(name: str) -> Optional[dict]:
    """
    Get detailed information about an encoder.

    Args:
        name: Encoder name

    Returns:
        Dictionary with encoder info, or None if not found
    """
    dim = get_encoder_dim(name)
    if dim is None:
        return None

    # Normalize name
    normalized = name.lower().replace('-', '_')

    # Determine family
    if 'uni' in normalized:
        family = 'UNI'
    elif 'conch' in normalized:
        family = 'CONCH'
    elif 'virchow' in normalized:
        family = 'Virchow'
    elif 'hoptimus' in normalized or normalized in ('h0', 'h1', 'h0_mini'):
        family = 'Hoptimus'
    elif 'kaiko' in normalized:
        family = 'Kaiko'
    elif 'phikon' in normalized:
        family = 'Phikon'
    elif 'hibou' in normalized:
        family = 'Hibou'
    else:
        family = 'Other'

    return {
        'name': normalized,
        'dim': dim,
        'family': family,
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Encoder Dimension Mapping")
    print("=" * 60)

    print("\nAvailable encoders:")
    for encoder in list_available_encoders():
        dim = get_encoder_dim(encoder)
        print(f"  {encoder}: {dim}")

    print("\nParsing examples:")
    test_models = [
        'abmil.base.uni_v2.none',
        'transmil.base.conch_v15.pc108-24k',
        'clam.sb.gigapath.none',
        'dftd.base.phikon.none',
    ]
    for model in test_models:
        encoder = parse_encoder_from_model_name(model)
        dim = get_encoder_dim(encoder) if encoder else None
        print(f"  {model} -> {encoder} (dim={dim})")

    print("\nValidation examples:")
    # Valid case
    valid, msg = validate_encoder_consistency('abmil.base.uni_v2.none', 'uni_v2', 1536)
    print(f"  Valid case: {valid}, {msg}")

    # Mismatch case
    valid, msg = validate_encoder_consistency('abmil.base.uni_v2.none', 'conch_v15', None)
    print(f"  Mismatch case: {valid}, {msg}")

    # Dimension mismatch
    valid, msg = validate_encoder_consistency('abmil.base.uni_v2.none', None, 768)
    print(f"  Dim mismatch: {valid}, {msg}")
