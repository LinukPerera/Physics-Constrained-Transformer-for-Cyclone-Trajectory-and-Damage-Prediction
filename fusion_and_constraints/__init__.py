"""
Fusion and Constraints Module for Cyclone Prediction.

This module combines multiple model branches and enforces physical constraints.
"""

from fusion_and_constraints.fusion_layer import (
    MultiBranchFusion,
    FusionConfig,
)

from fusion_and_constraints.constraint_enforcement import (
    HardConstraintEnforcer,
    SoftConstraintEnforcer,
    ConstraintViolationLogger,
)

__all__ = [
    "MultiBranchFusion",
    "FusionConfig",
    "HardConstraintEnforcer",
    "SoftConstraintEnforcer",
    "ConstraintViolationLogger",
]
