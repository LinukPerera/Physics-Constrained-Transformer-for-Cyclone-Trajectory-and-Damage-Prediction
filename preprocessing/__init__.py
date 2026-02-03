"""
Preprocessing Module for Cyclone Prediction System.

This module provides data preprocessing operations that preserve
physical meaning and dimensional interpretability.
"""

from preprocessing.alignment import (
    TemporalAligner,
    SpatialRegridder,
    StormFollowingTransform,
)

from preprocessing.normalization import (
    PhysicsAwareNormalizer,
    ReversibleTransform,
    NormalizationParams,
)

from preprocessing.feature_engineering import (
    DerivedFeatureCalculator,
    FeatureCategory,
    compute_vorticity,
    compute_divergence,
)

__all__ = [
    "TemporalAligner",
    "SpatialRegridder",
    "StormFollowingTransform",
    "PhysicsAwareNormalizer",
    "ReversibleTransform",
    "NormalizationParams",
    "DerivedFeatureCalculator",
    "FeatureCategory",
    "compute_vorticity",
    "compute_divergence",
]
