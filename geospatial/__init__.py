"""
Geospatial Module for Physics-Constrained Cyclone Prediction System.

This is a CRITICAL MODULE. All Earth-surface calculations system-wide
MUST originate from this module. No downstream module may implement
geometry calculations independently.

This module provides:
- WGS84 ellipsoidal coordinate models
- Geodesic distance calculations (globally valid)
- Map projections with distortion tracking
"""

from geospatial.coordinate_models import (
    WGS84Ellipsoid,
    geodetic_to_ecef,
    ecef_to_geodetic,
    geodetic_to_enu,
    radius_of_curvature_meridian,
    radius_of_curvature_prime_vertical,
)

from geospatial.distance_calculations import (
    geodesic_inverse,
    geodesic_direct,
    compute_azimuth,
    geodesic_distance,
    geodesic_distance_batch,
)

from geospatial.projections import (
    ProjectionAdapter,
    TransverseMercator,
    LambertConformalConic,
    compute_tissot_indicatrix,
    select_optimal_projection,
)

__all__ = [
    # Coordinate models
    "WGS84Ellipsoid",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "geodetic_to_enu",
    "radius_of_curvature_meridian",
    "radius_of_curvature_prime_vertical",
    # Distance calculations
    "geodesic_inverse",
    "geodesic_direct",
    "compute_azimuth",
    "geodesic_distance",
    "geodesic_distance_batch",
    # Projections
    "ProjectionAdapter",
    "TransverseMercator",
    "LambertConformalConic",
    "compute_tissot_indicatrix",
    "select_optimal_projection",
]
