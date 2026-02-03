"""
Coordinate Models for Ellipsoidal Earth Geometry.

This module implements coordinate transformations assuming a non-planar Earth
using the WGS84 reference ellipsoid. All calculations are valid for global
distances and intercontinental trajectories.

Scientific Context
------------------
Domain: Geodesy, Earth geometry
Model: WGS84 reference ellipsoid (not spherical approximation)

Why Simpler Models Are Invalid
------------------------------
1. Spherical Earth assumption: Introduces up to 0.3% error in distances,
   which equals ~21 km error over a 7000 km trans-Pacific cyclone track.
   
2. Planar/Cartesian approximation: Error grows quadratically with distance.
   Invalid beyond ~100 km, completely fails for trans-oceanic tracks.

3. Equal-area approximation: Cannot preserve both distance and direction,
   critical for trajectory forecasting.

References
----------
- NIMA TR8350.2: WGS84 parameters
- Torge, W. (2001). Geodesy (3rd ed.). de Gruyter.
- Hofmann-Wellenhof, B. et al. (2008). GNSS: GPS, GLONASS, Galileo.
"""

from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray

from common.constants import PhysicalConstants
from common.types import GeoCoordinate


@dataclass(frozen=True)
class EllipsoidParameters:
    """Parameters defining a reference ellipsoid.
    
    Attributes
    ----------
    a : float
        Semi-major axis (equatorial radius) in meters.
    f : float
        Flattening: f = (a - b) / a
    name : str
        Identifier for the ellipsoid.
        
    Derived Parameters
    ------------------
    b : float
        Semi-minor axis (polar radius) in meters.
    e2 : float
        First eccentricity squared: e² = (a² - b²) / a²
    ep2 : float
        Second eccentricity squared: e'² = (a² - b²) / b²
    """
    a: float
    f: float
    name: str
    
    @property
    def b(self) -> float:
        """Semi-minor axis in meters."""
        return self.a * (1 - self.f)
    
    @property
    def e2(self) -> float:
        """First eccentricity squared."""
        return self.f * (2 - self.f)
    
    @property
    def ep2(self) -> float:
        """Second eccentricity squared."""
        return self.e2 / (1 - self.e2)


# WGS84 ellipsoid - the standard reference for this system
WGS84Ellipsoid = EllipsoidParameters(
    a=PhysicalConstants.EARTH_SEMI_MAJOR_AXIS.value,
    f=PhysicalConstants.EARTH_FLATTENING.value,
    name="WGS84"
)


def radius_of_curvature_meridian(
    latitude_rad: float,
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid
) -> float:
    """Compute the radius of curvature in the meridian plane.
    
    This is the radius of curvature for north-south motion along
    a meridian (line of constant longitude).
    
    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude in radians.
    ellipsoid : EllipsoidParameters
        Reference ellipsoid (default: WGS84).
        
    Returns
    -------
    float
        Radius of curvature M in meters.
        
    Notes
    -----
    M = a(1 - e²) / (1 - e² sin²φ)^(3/2)
    
    At the equator (φ=0): M ≈ 6,335,439 m
    At the poles (φ=±90°): M ≈ 6,399,594 m
    """
    sin_lat = np.sin(latitude_rad)
    denominator = (1 - ellipsoid.e2 * sin_lat**2) ** 1.5
    return ellipsoid.a * (1 - ellipsoid.e2) / denominator


def radius_of_curvature_prime_vertical(
    latitude_rad: float,
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid
) -> float:
    """Compute the radius of curvature in the prime vertical.
    
    This is the radius of curvature for east-west motion along
    a parallel (line of constant latitude).
    
    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude in radians.
    ellipsoid : EllipsoidParameters
        Reference ellipsoid (default: WGS84).
        
    Returns
    -------
    float
        Radius of curvature N in meters.
        
    Notes
    -----
    N = a / (1 - e² sin²φ)^(1/2)
    
    At the equator (φ=0): N = a ≈ 6,378,137 m
    At the poles (φ=±90°): N ≈ 6,399,594 m
    """
    sin_lat = np.sin(latitude_rad)
    denominator = np.sqrt(1 - ellipsoid.e2 * sin_lat**2)
    return ellipsoid.a / denominator


def geodetic_to_ecef(
    latitude_rad: float,
    longitude_rad: float,
    altitude_m: float = 0.0,
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid
) -> Tuple[float, float, float]:
    """Convert geodetic coordinates to Earth-Centered Earth-Fixed (ECEF).
    
    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude in radians.
    longitude_rad : float
        Geodetic longitude in radians.
    altitude_m : float
        Height above ellipsoid in meters.
    ellipsoid : EllipsoidParameters
        Reference ellipsoid (default: WGS84).
        
    Returns
    -------
    Tuple[float, float, float]
        (X, Y, Z) coordinates in meters in ECEF frame.
        
    Notes
    -----
    The ECEF frame has:
    - Origin at Earth's center of mass
    - X-axis through the prime meridian (0° longitude) at equator
    - Y-axis through 90°E at equator
    - Z-axis through the North Pole
    
    References
    ----------
    Hofmann-Wellenhof, B. et al. (2008). GNSS. Section 5.6.
    """
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_lon = np.sin(longitude_rad)
    cos_lon = np.cos(longitude_rad)
    
    N = radius_of_curvature_prime_vertical(latitude_rad, ellipsoid)
    
    X = (N + altitude_m) * cos_lat * cos_lon
    Y = (N + altitude_m) * cos_lat * sin_lon
    Z = (N * (1 - ellipsoid.e2) + altitude_m) * sin_lat
    
    return X, Y, Z


def ecef_to_geodetic(
    X: float,
    Y: float,
    Z: float,
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid,
    max_iterations: int = 10,
    tolerance: float = 1e-12
) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to geodetic (latitude, longitude, altitude).
    
    Uses Bowring's iterative method for numerical stability.
    
    Parameters
    ----------
    X, Y, Z : float
        ECEF coordinates in meters.
    ellipsoid : EllipsoidParameters
        Reference ellipsoid (default: WGS84).
    max_iterations : int
        Maximum iterations for convergence.
    tolerance : float
        Convergence tolerance in radians.
        
    Returns
    -------
    Tuple[float, float, float]
        (latitude_rad, longitude_rad, altitude_m)
        
    Notes
    -----
    Bowring's method typically converges in 2-3 iterations for
    points on or near Earth's surface.
    
    References
    ----------
    Bowring, B.R. (1976). Transformation from spatial to geographical
    coordinates. Survey Review, 23(181), 323-327.
    """
    # Longitude is straightforward
    longitude_rad = np.arctan2(Y, X)
    
    # Distance from Z-axis
    p = np.sqrt(X**2 + Y**2)
    
    # Handle polar singularity
    if p < 1e-10:
        # At the pole
        latitude_rad = np.sign(Z) * np.pi / 2
        altitude_m = np.abs(Z) - ellipsoid.b
        return latitude_rad, longitude_rad, altitude_m
    
    # Initial approximation using spherical Earth
    latitude_rad = np.arctan2(Z, p * (1 - ellipsoid.e2))
    
    # Iterate to refine latitude
    for _ in range(max_iterations):
        sin_lat = np.sin(latitude_rad)
        N = radius_of_curvature_prime_vertical(latitude_rad, ellipsoid)
        
        latitude_new = np.arctan2(
            Z + ellipsoid.e2 * N * sin_lat,
            p
        )
        
        if np.abs(latitude_new - latitude_rad) < tolerance:
            latitude_rad = latitude_new
            break
            
        latitude_rad = latitude_new
    
    # Compute altitude
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    N = radius_of_curvature_prime_vertical(latitude_rad, ellipsoid)
    
    if np.abs(cos_lat) > 1e-10:
        altitude_m = p / cos_lat - N
    else:
        altitude_m = np.abs(Z) / np.abs(sin_lat) - N * (1 - ellipsoid.e2)
    
    return latitude_rad, longitude_rad, altitude_m


def geodetic_to_enu(
    target_lat_rad: float,
    target_lon_rad: float,
    target_alt_m: float,
    origin_lat_rad: float,
    origin_lon_rad: float,
    origin_alt_m: float = 0.0,
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid
) -> Tuple[float, float, float]:
    """Convert geodetic coordinates to local East-North-Up (ENU) frame.
    
    The ENU frame is a local tangent plane coordinate system centered
    at the origin point. This is useful for local-area calculations
    while maintaining awareness of the underlying curved geometry.
    
    Parameters
    ----------
    target_lat_rad, target_lon_rad, target_alt_m : float
        Target point in geodetic coordinates.
    origin_lat_rad, origin_lon_rad, origin_alt_m : float
        Origin point of the ENU frame.
    ellipsoid : EllipsoidParameters
        Reference ellipsoid (default: WGS84).
        
    Returns
    -------
    Tuple[float, float, float]
        (East, North, Up) coordinates in meters relative to origin.
        
    Notes
    -----
    - East: positive towards increasing longitude
    - North: positive towards increasing latitude
    - Up: positive away from Earth's center
    
    WARNING: The ENU frame is only valid for local calculations.
    For distances beyond ~100 km, use geodesic calculations directly.
    """
    # Convert both points to ECEF
    X_target, Y_target, Z_target = geodetic_to_ecef(
        target_lat_rad, target_lon_rad, target_alt_m, ellipsoid
    )
    X_origin, Y_origin, Z_origin = geodetic_to_ecef(
        origin_lat_rad, origin_lon_rad, origin_alt_m, ellipsoid
    )
    
    # Vector from origin to target in ECEF
    dX = X_target - X_origin
    dY = Y_target - Y_origin
    dZ = Z_target - Z_origin
    
    # Rotation matrix from ECEF to ENU
    sin_lat = np.sin(origin_lat_rad)
    cos_lat = np.cos(origin_lat_rad)
    sin_lon = np.sin(origin_lon_rad)
    cos_lon = np.cos(origin_lon_rad)
    
    # Apply rotation
    East = -sin_lon * dX + cos_lon * dY
    North = -sin_lat * cos_lon * dX - sin_lat * sin_lon * dY + cos_lat * dZ
    Up = cos_lat * cos_lon * dX + cos_lat * sin_lon * dY + sin_lat * dZ
    
    return East, North, Up


def enu_to_geodetic(
    east_m: float,
    north_m: float,
    up_m: float,
    origin_lat_rad: float,
    origin_lon_rad: float,
    origin_alt_m: float = 0.0,
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid
) -> Tuple[float, float, float]:
    """Convert local ENU coordinates back to geodetic.
    
    Parameters
    ----------
    east_m, north_m, up_m : float
        ENU coordinates in meters.
    origin_lat_rad, origin_lon_rad, origin_alt_m : float
        Origin point of the ENU frame.
    ellipsoid : EllipsoidParameters
        Reference ellipsoid (default: WGS84).
        
    Returns
    -------
    Tuple[float, float, float]
        (latitude_rad, longitude_rad, altitude_m)
    """
    # Get origin in ECEF
    X_origin, Y_origin, Z_origin = geodetic_to_ecef(
        origin_lat_rad, origin_lon_rad, origin_alt_m, ellipsoid
    )
    
    # Rotation matrix from ENU to ECEF (transpose of ECEF to ENU)
    sin_lat = np.sin(origin_lat_rad)
    cos_lat = np.cos(origin_lat_rad)
    sin_lon = np.sin(origin_lon_rad)
    cos_lon = np.cos(origin_lon_rad)
    
    # Apply inverse rotation
    dX = -sin_lon * east_m - sin_lat * cos_lon * north_m + cos_lat * cos_lon * up_m
    dY = cos_lon * east_m - sin_lat * sin_lon * north_m + cos_lat * sin_lon * up_m
    dZ = cos_lat * north_m + sin_lat * up_m
    
    # Convert back to geodetic
    X_target = X_origin + dX
    Y_target = Y_origin + dY
    Z_target = Z_origin + dZ
    
    return ecef_to_geodetic(X_target, Y_target, Z_target, ellipsoid)


def compute_local_radius(
    latitude_rad: float,
    azimuth_rad: float,
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid
) -> float:
    """Compute the radius of curvature in an arbitrary direction.
    
    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude in radians.
    azimuth_rad : float
        Azimuth direction in radians (clockwise from north).
    ellipsoid : EllipsoidParameters
        Reference ellipsoid (default: WGS84).
        
    Returns
    -------
    float
        Radius of curvature in the azimuth direction, in meters.
        
    Notes
    -----
    Uses Euler's formula: 1/R = cos²α/M + sin²α/N
    where α is azimuth, M is meridian radius, N is prime vertical radius.
    """
    M = radius_of_curvature_meridian(latitude_rad, ellipsoid)
    N = radius_of_curvature_prime_vertical(latitude_rad, ellipsoid)
    
    cos_az = np.cos(azimuth_rad)
    sin_az = np.sin(azimuth_rad)
    
    # Euler's formula
    inv_R = (cos_az**2 / M) + (sin_az**2 / N)
    
    return 1.0 / inv_R


# Vectorized versions for batch processing
def geodetic_to_ecef_batch(
    latitudes_rad: NDArray[np.float64],
    longitudes_rad: NDArray[np.float64],
    altitudes_m: NDArray[np.float64],
    ellipsoid: EllipsoidParameters = WGS84Ellipsoid
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized geodetic to ECEF conversion.
    
    Parameters
    ----------
    latitudes_rad : ndarray
        Array of latitudes in radians.
    longitudes_rad : ndarray
        Array of longitudes in radians.
    altitudes_m : ndarray
        Array of altitudes in meters.
    ellipsoid : EllipsoidParameters
        Reference ellipsoid.
        
    Returns
    -------
    Tuple[ndarray, ndarray, ndarray]
        (X, Y, Z) arrays in meters.
    """
    sin_lat = np.sin(latitudes_rad)
    cos_lat = np.cos(latitudes_rad)
    sin_lon = np.sin(longitudes_rad)
    cos_lon = np.cos(longitudes_rad)
    
    N = ellipsoid.a / np.sqrt(1 - ellipsoid.e2 * sin_lat**2)
    
    X = (N + altitudes_m) * cos_lat * cos_lon
    Y = (N + altitudes_m) * cos_lat * sin_lon
    Z = (N * (1 - ellipsoid.e2) + altitudes_m) * sin_lat
    
    return X, Y, Z
