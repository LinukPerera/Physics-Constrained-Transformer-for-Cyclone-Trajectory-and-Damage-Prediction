"""
Common utilities and infrastructure for the Physics-Constrained Cyclone Prediction System.

This package provides foundational components used across all modules:
- Physical constants with uncertainty bounds
- Unit registry and dimensional analysis
- Type definitions with physical units
- Logging and audit trail infrastructure
"""

from common.constants import PhysicalConstants
from common.units import UnitRegistry, validate_units
from common.types import (
    GeoCoordinate,
    AtmosphericState,
    CycloneState,
    TrajectoryPoint,
)
from common.logging_config import get_logger, AuditLogger

__all__ = [
    "PhysicalConstants",
    "UnitRegistry",
    "validate_units",
    "GeoCoordinate",
    "AtmosphericState",
    "CycloneState",
    "TrajectoryPoint",
    "get_logger",
    "AuditLogger",
]
