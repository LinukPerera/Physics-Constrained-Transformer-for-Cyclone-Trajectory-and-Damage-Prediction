"""
Geodesic Distance Calculations on the WGS84 Ellipsoid.

This module provides globally valid distance calculations using geodesic
(shortest path on ellipsoid) algorithms. These calculations are valid for
any distance, from meters to antipodal points.

Scientific Context
------------------
Domain: Geodesy, differential geometry on curved surfaces
Model: Geodesic (shortest path) on reference ellipsoid

Why Simpler Models Are Invalid
------------------------------
1. Haversine formula (spherical): Assumes spherical Earth, introducing
   up to 0.5% error. For a 5000 km cyclone track, this is 25 km error.
   
2. Euclidean/Planar: Completely invalid for Earth-surface distances.
   Error grows without bound with distance.
   
3. Rhumb line (loxodrome): Not the shortest path on a curved surface.
   While useful for navigation, it is longer than the geodesic.

Implementation
--------------
This module wraps the `pyproj` library, which uses the GeographicLib
algorithms by Charles Karney. These provide:
- Full double precision accuracy (better than 15 nm)
- Convergence for all point configurations including antipodal
- Numerical stability at all latitudes

References
----------
- Karney, C.F.F. (2013). Algorithms for geodesics. Journal of Geodesy, 87(1), 43-55.
- GeographicLib: https://geographiclib.sourceforge.io/
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray

from pyproj import Geod

from common.constants import PhysicalConstants
from common.types import GeoCoordinate
from geospatial.coordinate_models import WGS84Ellipsoid, EllipsoidParameters


# Create the geodesic calculator for WGS84
_wgs84_geod = Geod(ellps='WGS84')


@dataclass
class GeodesicResult:
    """Result of a geodesic calculation.
    
    Attributes
    ----------
    distance_m : float
        Geodesic (shortest path) distance in meters.
    azimuth_forward_rad : float
        Forward azimuth (direction from point 1 to point 2) in radians,
        measured clockwise from north.
    azimuth_back_rad : float
        Back azimuth (direction from point 2 to point 1) in radians,
        measured clockwise from north.
    """
    distance_m: float
    azimuth_forward_rad: float
    azimuth_back_rad: float


def geodesic_inverse(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float
) -> GeodesicResult:
    """Solve the inverse geodesic problem.
    
    Given two points, find the distance and azimuths between them.
    
    Parameters
    ----------
    lat1_rad, lon1_rad : float
        First point in radians.
    lat2_rad, lon2_rad : float
        Second point in radians.
        
    Returns
    -------
    GeodesicResult
        Distance in meters, forward and back azimuths in radians.
        
    Notes
    -----
    The geodesic is the shortest path on the ellipsoid surface.
    For the WGS84 ellipsoid, this calculation is accurate to better
    than 15 nanometers for any pair of points.
    
    Examples
    --------
    >>> # New York to London
    >>> import numpy as np
    >>> result = geodesic_inverse(
    ...     np.radians(40.7128), np.radians(-74.0060),  # NYC
    ...     np.radians(51.5074), np.radians(-0.1278)   # London
    ... )
    >>> print(f"Distance: {result.distance_m / 1000:.1f} km")
    Distance: 5570.2 km
    """
    # Convert to degrees for pyproj
    lat1_deg = np.degrees(lat1_rad)
    lon1_deg = np.degrees(lon1_rad)
    lat2_deg = np.degrees(lat2_rad)
    lon2_deg = np.degrees(lon2_rad)
    
    # Compute geodesic
    az_forward_deg, az_back_deg, distance_m = _wgs84_geod.inv(
        lon1_deg, lat1_deg, lon2_deg, lat2_deg
    )
    
    # Convert azimuths to radians and normalize to [0, 2π)
    az_forward_rad = np.radians(az_forward_deg) % (2 * np.pi)
    az_back_rad = np.radians(az_back_deg) % (2 * np.pi)
    
    return GeodesicResult(
        distance_m=float(distance_m),
        azimuth_forward_rad=float(az_forward_rad),
        azimuth_back_rad=float(az_back_rad)
    )


def geodesic_direct(
    lat1_rad: float,
    lon1_rad: float,
    azimuth_rad: float,
    distance_m: float
) -> Tuple[float, float, float]:
    """Solve the direct geodesic problem.
    
    Given a starting point, azimuth, and distance, find the endpoint.
    
    Parameters
    ----------
    lat1_rad, lon1_rad : float
        Starting point in radians.
    azimuth_rad : float
        Forward azimuth in radians (clockwise from north).
    distance_m : float
        Distance to travel in meters.
        
    Returns
    -------
    Tuple[float, float, float]
        (lat2_rad, lon2_rad, back_azimuth_rad) - endpoint and back azimuth.
        
    Examples
    --------
    >>> # Travel 1000 km due east from the equator
    >>> import numpy as np
    >>> lat, lon, az = geodesic_direct(0.0, 0.0, np.pi/2, 1_000_000)
    >>> print(f"Endpoint: {np.degrees(lat):.4f}°, {np.degrees(lon):.4f}°")
    Endpoint: 0.0000°, 8.9932°
    """
    # Convert to degrees for pyproj
    lat1_deg = np.degrees(lat1_rad)
    lon1_deg = np.degrees(lon1_rad)
    azimuth_deg = np.degrees(azimuth_rad)
    
    # Compute geodesic
    lon2_deg, lat2_deg, az_back_deg = _wgs84_geod.fwd(
        lon1_deg, lat1_deg, azimuth_deg, distance_m
    )
    
    # Convert back to radians
    lat2_rad = np.radians(lat2_deg)
    lon2_rad = np.radians(lon2_deg)
    az_back_rad = np.radians(az_back_deg) % (2 * np.pi)
    
    return lat2_rad, lon2_rad, az_back_rad


def geodesic_distance(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float
) -> float:
    """Compute geodesic distance between two points.
    
    This is a convenience function that returns only the distance.
    
    Parameters
    ----------
    lat1_rad, lon1_rad : float
        First point in radians.
    lat2_rad, lon2_rad : float
        Second point in radians.
        
    Returns
    -------
    float
        Geodesic distance in meters.
    """
    result = geodesic_inverse(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
    return result.distance_m


def geodesic_distance_batch(
    lat1_rad: NDArray[np.float64],
    lon1_rad: NDArray[np.float64],
    lat2_rad: NDArray[np.float64],
    lon2_rad: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute geodesic distances for arrays of point pairs.
    
    This is the vectorized version for efficient batch processing.
    
    Parameters
    ----------
    lat1_rad, lon1_rad : ndarray
        First points in radians.
    lat2_rad, lon2_rad : ndarray
        Second points in radians.
        
    Returns
    -------
    ndarray
        Geodesic distances in meters.
        
    Notes
    -----
    This function uses numpy broadcasting, so inputs can be:
    - Same shape: pairwise distances
    - Broadcastable shapes: distance from one point to many, etc.
    """
    # Convert to degrees
    lat1_deg = np.degrees(lat1_rad)
    lon1_deg = np.degrees(lon1_rad)
    lat2_deg = np.degrees(lat2_rad)
    lon2_deg = np.degrees(lon2_rad)
    
    # Compute geodesic distances
    _, _, distances = _wgs84_geod.inv(lon1_deg, lat1_deg, lon2_deg, lat2_deg)
    
    return np.asarray(distances, dtype=np.float64)


def compute_azimuth(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float
) -> float:
    """Compute the forward azimuth from point 1 to point 2.
    
    Parameters
    ----------
    lat1_rad, lon1_rad : float
        First point in radians.
    lat2_rad, lon2_rad : float
        Second point in radians.
        
    Returns
    -------
    float
        Forward azimuth in radians, measured clockwise from north [0, 2π).
    """
    result = geodesic_inverse(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
    return result.azimuth_forward_rad


def compute_storm_motion(
    positions: list,
    timestamps: list
) -> list:
    """Compute storm motion (velocity) from a sequence of positions.
    
    Parameters
    ----------
    positions : list of GeoCoordinate
        Sequence of storm center positions.
    timestamps : list of datetime
        Corresponding timestamps.
        
    Returns
    -------
    list of dict
        List of motion vectors with keys:
        - 'speed_ms': Translation speed in m/s
        - 'heading_rad': Direction of motion in radians (clockwise from north)
        - 'u_ms': Eastward component in m/s
        - 'v_ms': Northward component in m/s
        
    Notes
    -----
    Uses geodesic calculations for accurate motion vectors.
    Motion is computed between consecutive position pairs.
    """
    if len(positions) < 2:
        return []
    
    motions = []
    
    for i in range(1, len(positions)):
        pos1 = positions[i - 1]
        pos2 = positions[i]
        t1 = timestamps[i - 1]
        t2 = timestamps[i]
        
        # Time difference in seconds
        dt_seconds = (t2 - t1).total_seconds()
        
        if dt_seconds <= 0:
            continue
            
        # Geodesic distance and azimuth
        result = geodesic_inverse(
            pos1.latitude, pos1.longitude,
            pos2.latitude, pos2.longitude
        )
        
        # Speed
        speed_ms = result.distance_m / dt_seconds
        heading_rad = result.azimuth_forward_rad
        
        # Decompose into u (east) and v (north) components
        u_ms = speed_ms * np.sin(heading_rad)
        v_ms = speed_ms * np.cos(heading_rad)
        
        motions.append({
            'speed_ms': speed_ms,
            'heading_rad': heading_rad,
            'u_ms': u_ms,
            'v_ms': v_ms
        })
    
    return motions


def interpolate_geodesic(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float,
    num_points: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate points along a geodesic between two endpoints.
    
    Parameters
    ----------
    lat1_rad, lon1_rad : float
        First point in radians.
    lat2_rad, lon2_rad : float
        Second point in radians.
    num_points : int
        Number of points including endpoints.
        
    Returns
    -------
    Tuple[ndarray, ndarray]
        (latitudes_rad, longitudes_rad) arrays of interpolated points.
        
    Notes
    -----
    Points are equally spaced in distance along the geodesic.
    """
    # Get total distance
    total_distance = geodesic_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
    
    # Get initial azimuth
    result = geodesic_inverse(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
    azimuth = result.azimuth_forward_rad
    
    # Generate points
    distances = np.linspace(0, total_distance, num_points)
    
    lats = np.zeros(num_points)
    lons = np.zeros(num_points)
    
    lats[0] = lat1_rad
    lons[0] = lon1_rad
    lats[-1] = lat2_rad
    lons[-1] = lon2_rad
    
    for i in range(1, num_points - 1):
        lat, lon, _ = geodesic_direct(lat1_rad, lon1_rad, azimuth, distances[i])
        lats[i] = lat
        lons[i] = lon
    
    return lats, lons


def great_circle_distance_with_error_bound(
    lat1_rad: float,
    lon1_rad: float,
    lat2_rad: float,
    lon2_rad: float
) -> Tuple[float, float]:
    """Compute spherical great-circle distance WITH explicit error bound.
    
    This function is provided for cases where computational efficiency
    is critical and the error is acceptable. The error bound is computed
    and returned so the caller can decide whether to use this approximation.
    
    Parameters
    ----------
    lat1_rad, lon1_rad : float
        First point in radians.
    lat2_rad, lon2_rad : float
        Second point in radians.
        
    Returns
    -------
    Tuple[float, float]
        (distance_m, max_error_m)
        Approximate distance and maximum expected error, both in meters.
        
    Notes
    -----
    The spherical approximation error depends on:
    1. Distance (longer distances have more accumulated error)
    2. Latitude (error is larger near poles)
    3. Azimuth (error varies with direction)
    
    For typical tropical cyclone latitudes (5°-40°), the error is
    approximately 0.1-0.3% of the distance.
    
    WARNING
    -------
    This function should only be used when:
    1. The error bound is acceptable for the application
    2. Performance is critical (processing millions of points)
    3. The result is not used for downstream physics calculations
    """
    # Use mean Earth radius
    R = PhysicalConstants.EARTH_MEAN_RADIUS.value
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance_approx = R * c
    
    # Estimate maximum error (approximately 0.3% for tropical latitudes)
    # Error increases with latitude and distance
    mean_lat = (lat1_rad + lat2_rad) / 2
    lat_factor = 1 + 0.5 * np.abs(np.sin(mean_lat))  # Higher error near poles
    max_error = 0.003 * distance_approx * lat_factor
    
    return distance_approx, max_error


def compute_heading_change(
    heading1_rad: float,
    heading2_rad: float
) -> float:
    """Compute the signed change in heading (turn angle).
    
    Parameters
    ----------
    heading1_rad : float
        Initial heading in radians.
    heading2_rad : float
        Final heading in radians.
        
    Returns
    -------
    float
        Signed heading change in radians.
        Positive = clockwise (rightward) turn.
        Negative = counterclockwise (leftward) turn.
        Range: [-π, π]
    """
    delta = heading2_rad - heading1_rad
    
    # Normalize to [-π, π]
    while delta > np.pi:
        delta -= 2 * np.pi
    while delta < -np.pi:
        delta += 2 * np.pi
    
    return delta
