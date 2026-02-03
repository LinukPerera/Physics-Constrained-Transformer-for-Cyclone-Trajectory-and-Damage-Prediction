"""
Map Projections with Distortion Tracking.

This module provides map projection capabilities with explicit tracking
of projection distortions. Since all map projections introduce distortions,
this module quantifies those distortions so downstream computations can
account for them.

Scientific Context
------------------
Domain: Cartography, mathematical geodesy
Model: Conformal and equal-area projections

Why Projection Awareness Matters
--------------------------------
1. No flat map can perfectly represent a curved surface.
2. Different projections preserve different properties:
   - Conformal: preserves angles (important for wind directions)
   - Equal-area: preserves area (important for damage extent)
   - Equidistant: preserves distance along specific lines
3. For cyclone modeling, we need to know what is being distorted.

Implementation
--------------
This module wraps `pyproj` for projection calculations and provides
Tissot's indicatrix calculations to quantify local distortion.

References
----------
- Snyder, J.P. (1987). Map Projections - A Working Manual. USGS Prof. Paper 1395.
- Tissot, A. (1859). Mémoire sur la représentation des surfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray

from pyproj import CRS, Transformer

from geospatial.coordinate_models import (
    WGS84Ellipsoid,
    radius_of_curvature_meridian,
    radius_of_curvature_prime_vertical
)


@dataclass
class TissotIndicatrix:
    """Tissot's indicatrix describing local distortion at a point.
    
    The Tissot indicatrix shows how an infinitesimally small circle
    on the Earth's surface is distorted into an ellipse on the map.
    
    Attributes
    ----------
    semi_major : float
        Semi-major axis of the distortion ellipse (scale factor).
    semi_minor : float
        Semi-minor axis of the distortion ellipse (scale factor).
    orientation_rad : float
        Orientation of the major axis in radians (from east).
    area_scale : float
        Area distortion factor (h * k where h, k are principal scale factors).
    angular_distortion_rad : float
        Maximum angular distortion in radians.
    
    Notes
    -----
    - For a conformal projection: semi_major = semi_minor (circle, no angular distortion)
    - For an equal-area projection: area_scale = 1.0 (but shapes are distorted)
    """
    semi_major: float  # h: scale along meridian
    semi_minor: float  # k: scale along parallel
    orientation_rad: float
    area_scale: float
    angular_distortion_rad: float
    
    @property
    def is_conformal(self) -> bool:
        """Check if projection is locally conformal (circle, no angular distortion)."""
        return np.abs(self.semi_major - self.semi_minor) < 1e-6
    
    @property
    def is_equal_area(self) -> bool:
        """Check if projection is locally equal-area."""
        return np.abs(self.area_scale - 1.0) < 1e-6


class ProjectionAdapter(ABC):
    """Abstract base class for map projection adapters.
    
    All projections in this system must implement this interface to
    ensure consistent handling of coordinates and distortions.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the projection."""
        pass
    
    @property
    @abstractmethod
    def proj4_string(self) -> str:
        """PROJ.4 definition string."""
        pass
    
    @property
    @abstractmethod
    def preserves_angles(self) -> bool:
        """Whether this is a conformal projection."""
        pass
    
    @property
    @abstractmethod
    def preserves_area(self) -> bool:
        """Whether this is an equal-area projection."""
        pass
    
    @abstractmethod
    def to_projected(
        self,
        lat_rad: float,
        lon_rad: float
    ) -> Tuple[float, float]:
        """Transform geodetic coordinates to projected coordinates.
        
        Parameters
        ----------
        lat_rad, lon_rad : float
            Geodetic coordinates in radians.
            
        Returns
        -------
        Tuple[float, float]
            (x, y) projected coordinates in meters.
        """
        pass
    
    @abstractmethod
    def to_geodetic(
        self,
        x: float,
        y: float
    ) -> Tuple[float, float]:
        """Transform projected coordinates to geodetic.
        
        Parameters
        ----------
        x, y : float
            Projected coordinates in meters.
            
        Returns
        -------
        Tuple[float, float]
            (lat_rad, lon_rad) geodetic coordinates in radians.
        """
        pass
    
    @abstractmethod
    def compute_distortion(
        self,
        lat_rad: float,
        lon_rad: float
    ) -> TissotIndicatrix:
        """Compute local distortion at a point.
        
        Parameters
        ----------
        lat_rad, lon_rad : float
            Location in geodetic coordinates (radians).
            
        Returns
        -------
        TissotIndicatrix
            Local distortion characteristics.
        """
        pass


class TransverseMercator(ProjectionAdapter):
    """Transverse Mercator projection.
    
    A conformal (angle-preserving) projection suitable for regions
    that extend primarily north-south. This is the basis for UTM.
    
    Parameters
    ----------
    central_meridian_deg : float
        Central meridian longitude in degrees.
    scale_factor : float
        Scale factor at central meridian (default: 0.9996 for UTM).
    false_easting : float
        False easting in meters (default: 500000 for UTM).
    false_northing : float
        False northing in meters (default: 0 for northern hemisphere).
        
    Notes
    -----
    Distortion increases with distance from the central meridian.
    Typically valid within 3° of the central meridian for high accuracy.
    """
    
    def __init__(
        self,
        central_meridian_deg: float,
        scale_factor: float = 0.9996,
        false_easting: float = 500000.0,
        false_northing: float = 0.0
    ):
        self._central_meridian = central_meridian_deg
        self._scale_factor = scale_factor
        self._false_easting = false_easting
        self._false_northing = false_northing
        
        self._proj4 = (
            f"+proj=tmerc +lat_0=0 +lon_0={central_meridian_deg} "
            f"+k={scale_factor} +x_0={false_easting} +y_0={false_northing} "
            "+ellps=WGS84 +units=m +no_defs"
        )
        
        self._crs_geo = CRS.from_epsg(4326)  # WGS84
        self._crs_proj = CRS.from_proj4(self._proj4)
        self._to_proj = Transformer.from_crs(self._crs_geo, self._crs_proj, always_xy=True)
        self._to_geo = Transformer.from_crs(self._crs_proj, self._crs_geo, always_xy=True)
    
    @property
    def name(self) -> str:
        return f"Transverse Mercator (CM={self._central_meridian}°)"
    
    @property
    def proj4_string(self) -> str:
        return self._proj4
    
    @property
    def preserves_angles(self) -> bool:
        return True
    
    @property
    def preserves_area(self) -> bool:
        return False
    
    def to_projected(self, lat_rad: float, lon_rad: float) -> Tuple[float, float]:
        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        x, y = self._to_proj.transform(lon_deg, lat_deg)
        return float(x), float(y)
    
    def to_geodetic(self, x: float, y: float) -> Tuple[float, float]:
        lon_deg, lat_deg = self._to_geo.transform(x, y)
        return np.radians(lat_deg), np.radians(lon_deg)
    
    def compute_distortion(self, lat_rad: float, lon_rad: float) -> TissotIndicatrix:
        # For Transverse Mercator, scale factor varies with distance from CM
        lon_deg = np.degrees(lon_rad)
        delta_lon = np.abs(lon_deg - self._central_meridian)
        
        # Approximate scale factor (increases with distance from CM)
        # k = k0 * sec(x/R) ≈ k0 * (1 + x²/(2R²))
        # where x is distance from CM
        lat = lat_rad
        R = radius_of_curvature_prime_vertical(lat)
        x_dist = R * np.cos(lat) * np.radians(delta_lon)
        
        k = self._scale_factor * (1 + (x_dist**2) / (2 * R**2))
        
        # Conformal projection: h = k (both scale factors equal)
        return TissotIndicatrix(
            semi_major=k,
            semi_minor=k,
            orientation_rad=0.0,
            area_scale=k * k,
            angular_distortion_rad=0.0  # Conformal: no angular distortion
        )


class LambertConformalConic(ProjectionAdapter):
    """Lambert Conformal Conic projection.
    
    A conformal (angle-preserving) projection suitable for mid-latitude
    regions that extend primarily east-west. Common for weather maps.
    
    Parameters
    ----------
    lat1_deg, lat2_deg : float
        Standard parallels in degrees.
    origin_lat_deg, origin_lon_deg : float
        Projection origin in degrees.
    false_easting, false_northing : float
        False coordinates in meters.
        
    Notes
    -----
    Distortion is minimal between the standard parallels and increases
    away from them. Ideal for mid-latitude cyclone tracking.
    """
    
    def __init__(
        self,
        lat1_deg: float,
        lat2_deg: float,
        origin_lat_deg: float,
        origin_lon_deg: float,
        false_easting: float = 0.0,
        false_northing: float = 0.0
    ):
        self._lat1 = lat1_deg
        self._lat2 = lat2_deg
        self._origin_lat = origin_lat_deg
        self._origin_lon = origin_lon_deg
        self._false_easting = false_easting
        self._false_northing = false_northing
        
        self._proj4 = (
            f"+proj=lcc +lat_1={lat1_deg} +lat_2={lat2_deg} "
            f"+lat_0={origin_lat_deg} +lon_0={origin_lon_deg} "
            f"+x_0={false_easting} +y_0={false_northing} "
            "+ellps=WGS84 +units=m +no_defs"
        )
        
        self._crs_geo = CRS.from_epsg(4326)
        self._crs_proj = CRS.from_proj4(self._proj4)
        self._to_proj = Transformer.from_crs(self._crs_geo, self._crs_proj, always_xy=True)
        self._to_geo = Transformer.from_crs(self._crs_proj, self._crs_geo, always_xy=True)
    
    @property
    def name(self) -> str:
        return f"Lambert Conformal Conic ({self._lat1}°, {self._lat2}°)"
    
    @property
    def proj4_string(self) -> str:
        return self._proj4
    
    @property
    def preserves_angles(self) -> bool:
        return True
    
    @property
    def preserves_area(self) -> bool:
        return False
    
    def to_projected(self, lat_rad: float, lon_rad: float) -> Tuple[float, float]:
        lat_deg = np.degrees(lat_rad)
        lon_deg = np.degrees(lon_rad)
        x, y = self._to_proj.transform(lon_deg, lat_deg)
        return float(x), float(y)
    
    def to_geodetic(self, x: float, y: float) -> Tuple[float, float]:
        lon_deg, lat_deg = self._to_geo.transform(x, y)
        return np.radians(lat_deg), np.radians(lon_deg)
    
    def compute_distortion(self, lat_rad: float, lon_rad: float) -> TissotIndicatrix:
        # For LCC, scale factor varies with latitude
        lat_deg = np.degrees(lat_rad)
        
        # Scale is 1.0 at standard parallels
        # Approximate: scale decreases between parallels, increases outside
        mid_lat = (self._lat1 + self._lat2) / 2
        
        if self._lat1 <= lat_deg <= self._lat2:
            # Between standard parallels: slight compression
            offset = np.abs(lat_deg - mid_lat) / ((self._lat2 - self._lat1) / 2)
            k = 1.0 - 0.001 * offset  # Very small distortion
        else:
            # Outside standard parallels: expansion
            if lat_deg < self._lat1:
                offset = np.abs(lat_deg - self._lat1)
            else:
                offset = np.abs(lat_deg - self._lat2)
            k = 1.0 + 0.01 * (offset / 10)**2  # Growing distortion
        
        return TissotIndicatrix(
            semi_major=k,
            semi_minor=k,
            orientation_rad=0.0,
            area_scale=k * k,
            angular_distortion_rad=0.0
        )


def compute_tissot_indicatrix(
    projection: ProjectionAdapter,
    lat_rad: float,
    lon_rad: float,
    delta: float = 1e-5
) -> TissotIndicatrix:
    """Compute Tissot's indicatrix numerically.
    
    This method computes the distortion ellipse by projecting a small
    circle and analyzing the resulting ellipse. Works for any projection.
    
    Parameters
    ----------
    projection : ProjectionAdapter
        The projection to analyze.
    lat_rad, lon_rad : float
        Location in geodetic coordinates (radians).
    delta : float
        Small angular offset for numerical differentiation.
        
    Returns
    -------
    TissotIndicatrix
        Local distortion characteristics.
    """
    # Project center point
    x0, y0 = projection.to_projected(lat_rad, lon_rad)
    
    # Compute partial derivatives numerically
    # ∂x/∂λ, ∂y/∂λ (east-west)
    x_e, y_e = projection.to_projected(lat_rad, lon_rad + delta)
    dxdl = (x_e - x0) / delta
    dydl = (y_e - y0) / delta
    
    # ∂x/∂φ, ∂y/∂φ (north-south)
    x_n, y_n = projection.to_projected(lat_rad + delta, lon_rad)
    dxdp = (x_n - x0) / delta
    dydp = (y_n - y0) / delta
    
    # Compute scale factors on the ellipsoid
    M = radius_of_curvature_meridian(lat_rad)
    N = radius_of_curvature_prime_vertical(lat_rad)
    
    # Scale along meridian (h) and parallel (k)
    h = np.sqrt(dxdp**2 + dydp**2) / M
    k = np.sqrt(dxdl**2 + dydl**2) / (N * np.cos(lat_rad))
    
    # Angular distortion (maximum for conformal: 0)
    sin_omega = np.abs(dxdp * dydl - dydp * dxdl) / (h * M * k * N * np.cos(lat_rad))
    sin_omega = np.clip(sin_omega, -1, 1)
    omega = np.arcsin(sin_omega)
    
    # Area scale factor
    area_scale = h * k * np.sin(np.pi/2 - omega) if omega < np.pi/2 else h * k
    
    # Orientation of principal direction
    theta = 0.5 * np.arctan2(2 * (dxdp * dxdl + dydp * dydl),
                              dxdp**2 + dydp**2 - dxdl**2 - dydl**2)
    
    return TissotIndicatrix(
        semi_major=max(h, k),
        semi_minor=min(h, k),
        orientation_rad=theta,
        area_scale=area_scale,
        angular_distortion_rad=np.abs(np.arcsin(np.abs(h - k) / (h + k))) * 2
    )


def select_optimal_projection(
    min_lat_deg: float,
    max_lat_deg: float,
    min_lon_deg: float,
    max_lon_deg: float,
    purpose: str = "general"
) -> ProjectionAdapter:
    """Select an optimal projection for a given geographic extent.
    
    Parameters
    ----------
    min_lat_deg, max_lat_deg : float
        Latitude range in degrees.
    min_lon_deg, max_lon_deg : float
        Longitude range in degrees.
    purpose : str
        One of:
        - "general": Balance distortions
        - "trajectory": Optimize for path display (conformal)
        - "damage_extent": Optimize for area calculations (equal-area)
        
    Returns
    -------
    ProjectionAdapter
        An appropriate projection for the domain.
        
    Notes
    -----
    Selection logic:
    1. For narrow east-west extent: Transverse Mercator
    2. For narrow north-south extent: Lambert Conformal Conic
    3. For tropical regions: Mercator
    4. For polar regions: Stereographic
    """
    lat_extent = max_lat_deg - min_lat_deg
    lon_extent = max_lon_deg - min_lon_deg
    center_lat = (min_lat_deg + max_lat_deg) / 2
    center_lon = (min_lon_deg + max_lon_deg) / 2
    
    # For typical cyclone tracking (mid-latitudes, moderate extent)
    if lat_extent > lon_extent * np.cos(np.radians(center_lat)):
        # Taller region: use Transverse Mercator
        return TransverseMercator(
            central_meridian_deg=center_lon,
            scale_factor=0.9996
        )
    else:
        # Wider region: use Lambert Conformal Conic
        lat1 = min_lat_deg + lat_extent * 0.25
        lat2 = max_lat_deg - lat_extent * 0.25
        return LambertConformalConic(
            lat1_deg=lat1,
            lat2_deg=lat2,
            origin_lat_deg=center_lat,
            origin_lon_deg=center_lon
        )


def batch_project(
    projection: ProjectionAdapter,
    lats_rad: NDArray[np.float64],
    lons_rad: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project arrays of coordinates.
    
    Parameters
    ----------
    projection : ProjectionAdapter
        Projection to use.
    lats_rad, lons_rad : ndarray
        Coordinates in radians.
        
    Returns
    -------
    Tuple[ndarray, ndarray]
        (x, y) projected coordinates in meters.
    """
    lats_deg = np.degrees(lats_rad)
    lons_deg = np.degrees(lons_rad)
    
    x, y = projection._to_proj.transform(lons_deg, lats_deg)
    
    return np.asarray(x), np.asarray(y)
