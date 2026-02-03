"""
Physics Extraction Module for Cyclone Prediction.

This module converts learned representations and raw observations into
physically interpretable quantities. NO ML predictions are made here -
only physics-based transformations.
"""

from physics_extraction.atmospheric_dynamics import (
    compute_potential_intensity,
    compute_thermal_wind,
    compute_pressure_gradient_force,
    compute_coriolis_acceleration,
)

from physics_extraction.force_terms import (
    compute_beta_drift,
    compute_environmental_steering,
    compute_land_friction,
    CycloneForceCalculator,
)

__all__ = [
    "compute_potential_intensity",
    "compute_thermal_wind",
    "compute_pressure_gradient_force",
    "compute_coriolis_acceleration",
    "compute_beta_drift",
    "compute_environmental_steering",
    "compute_land_friction",
    "CycloneForceCalculator",
]
