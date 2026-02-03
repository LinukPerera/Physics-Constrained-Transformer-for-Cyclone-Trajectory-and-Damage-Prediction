"""
Wind Field and Damage Models for Tropical Cyclones.

This module provides parametric wind field models and damage estimation
based on wind exposure.

Wind Field Models
-----------------
- Holland (1980): Parametric model based on pressure profile
- Willoughby (2006): Improved profile with dual exponential

CRITICAL: Wind fields require geospatial module for distance calculations.

References
----------
- Holland, G.J. (1980). An analytic model of the wind and pressure profiles.
- Willoughby, H.E. et al. (2006). Parametric representation of the primary
  hurricane vortex. Part I: Observations and evaluation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from common.logging_config import get_logger
from common.constants import PhysicalConstants
from geospatial.distance_calculations import geodesic_distance_batch

logger = get_logger(__name__)


@dataclass
class WindFieldResult:
    """Result from wind field calculation.
    
    Attributes
    ----------
    wind_speed : ndarray
        Wind speed at each point in m/s.
    wind_direction : ndarray
        Wind direction in radians (from which wind blows).
    u_wind : ndarray
        Eastward wind component in m/s.
    v_wind : ndarray
        Northward wind component in m/s.
    """
    wind_speed: NDArray[np.float64]
    wind_direction: NDArray[np.float64]
    u_wind: NDArray[np.float64]
    v_wind: NDArray[np.float64]


class HollandWindModel:
    """Holland (1980) parametric wind profile model.
    
    This model computes wind speed as a function of radius based on
    the cyclone's pressure profile.
    
    Model Equation
    --------------
    V(r) = sqrt((B/ρ) × (Rm/r)^B × (Pc - P∞) × exp(-(Rm/r)^B) + (rf/2)²) - rf/2
    
    where:
    - B: Holland shape parameter (1.0-2.5)
    - r: Radius from storm center
    - Rm: Radius of maximum winds
    - Pc: Central pressure
    - P∞: Ambient pressure
    - ρ: Air density
    - f: Coriolis parameter
    
    Assumptions
    -----------
    - Axisymmetric wind field
    - Gradient wind balance
    - Sea-level (surface friction not modeled)
    """
    
    def __init__(
        self,
        central_pressure_Pa: float,
        ambient_pressure_Pa: float = 101300,
        radius_max_wind_m: float = 50000,
        latitude_deg: float = 20.0,
        holland_B: Optional[float] = None
    ):
        """Initialize Holland wind model.
        
        Parameters
        ----------
        central_pressure_Pa : float
            Central pressure in Pascals.
        ambient_pressure_Pa : float
            Ambient (environmental) pressure in Pascals.
        radius_max_wind_m : float
            Radius of maximum winds in meters.
        latitude_deg : float
            Storm center latitude for Coriolis calculation.
        holland_B : float, optional
            Holland shape parameter. If None, estimated from pressure.
        """
        self.Pc = central_pressure_Pa
        self.P_inf = ambient_pressure_Pa
        self.Rm = radius_max_wind_m
        self.latitude = np.radians(latitude_deg)
        
        # Air density at surface (approximate)
        self.rho = 1.15  # kg/m³
        
        # Coriolis parameter
        self.f = PhysicalConstants.coriolis_parameter(self.latitude)
        
        # Pressure deficit
        self.delta_P = self.P_inf - self.Pc
        
        # Holland B parameter
        if holland_B is not None:
            self.B = holland_B
        else:
            # Estimate from pressure deficit (Vickery & Wadhera 2008)
            self.B = self._estimate_holland_B()
        
        self._logger = get_logger("HollandWindModel")
    
    def _estimate_holland_B(self) -> float:
        """Estimate Holland B from storm properties.
        
        Uses empirical relationships from literature.
        """
        # Vickery and Wadhera (2008) relationship
        # B ≈ 1.0 + 0.5 × (ΔP / 50000) for weak to moderate storms
        B = 1.0 + 0.5 * (self.delta_P / 50000)
        return np.clip(B, 1.0, 2.5)
    
    def wind_at_radius(self, radius_m: float) -> float:
        """Calculate wind speed at given radius.
        
        Parameters
        ----------
        radius_m : float
            Radius from storm center in meters.
            
        Returns
        -------
        float
            Wind speed at that radius in m/s.
        """
        if radius_m <= 0:
            return 0.0
        
        r = radius_m
        Rm = self.Rm
        
        # Holland formula
        B = self.B
        f = self.f
        rho = self.rho
        
        term1 = (B / rho) * ((Rm / r) ** B) * self.delta_P * np.exp(-(Rm / r) ** B)
        term2 = (r * f / 2) ** 2
        
        V = np.sqrt(term1 + term2) - r * np.abs(f) / 2
        
        return max(0, V)
    
    def wind_field(
        self,
        center_lat_deg: float,
        center_lon_deg: float,
        lats_deg: NDArray[np.float64],
        lons_deg: NDArray[np.float64],
        include_translation: bool = False,
        translation_speed_ms: float = 0.0,
        translation_direction_rad: float = 0.0
    ) -> WindFieldResult:
        """Compute wind field on a lat/lon grid.
        
        Parameters
        ----------
        center_lat_deg, center_lon_deg : float
            Storm center in degrees.
        lats_deg, lons_deg : ndarray
            1D arrays of latitude and longitude grid points.
        include_translation : bool
            Whether to add storm translation velocity.
        translation_speed_ms : float
            Storm forward speed in m/s.
        translation_direction_rad : float
            Storm motion direction in radians (toward which moving).
            
        Returns
        -------
        WindFieldResult
            Wind field on the grid.
        """
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons_deg, lats_deg)
        
        # Calculate distances using geodesic
        center_lat_rad = np.radians(center_lat_deg)
        center_lon_rad = np.radians(center_lon_deg)
        
        distances = geodesic_distance_batch(
            np.radians(lat_grid),
            np.radians(lon_grid),
            np.full_like(lat_grid, center_lat_rad),
            np.full_like(lon_grid, center_lon_rad)
        )
        
        # Calculate azimuth from storm center to each point
        delta_lon = np.radians(lon_grid) - center_lon_rad
        azimuth = np.arctan2(
            np.sin(delta_lon),
            np.tan(np.radians(lat_grid)) * np.cos(center_lat_rad) -
            np.sin(center_lat_rad) * np.cos(delta_lon)
        )
        
        # Calculate wind speed at each radius
        wind_speed = np.vectorize(self.wind_at_radius)(distances)
        
        # Wind direction: perpendicular to radius (cyclonic)
        # Northern hemisphere: counterclockwise
        # Southern hemisphere: clockwise
        hemisphere = np.sign(center_lat_deg)
        inflow_angle = np.radians(20)  # Typical boundary layer inflow
        
        wind_direction = azimuth + hemisphere * (np.pi / 2) + inflow_angle
        
        # Convert to u, v components
        u_wind = -wind_speed * np.sin(wind_direction)
        v_wind = -wind_speed * np.cos(wind_direction)
        
        # Add translation velocity if requested
        if include_translation:
            u_translation = translation_speed_ms * np.sin(translation_direction_rad)
            v_translation = translation_speed_ms * np.cos(translation_direction_rad)
            u_wind = u_wind + u_translation
            v_wind = v_wind + v_translation
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)
            wind_direction = np.arctan2(-u_wind, -v_wind)
        
        return WindFieldResult(
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            u_wind=u_wind,
            v_wind=v_wind
        )


class WindDamageModel:
    """Model for estimating wind damage from wind speed.
    
    Uses damage functions that relate wind speed to damage ratio.
    
    References
    ----------
    - Vickery, P.J. et al. (2006). HAZUS-MH wind model.
    - Emanuel, K. (2011). Global warming effects on U.S. hurricane damage.
    """
    
    # Wind speed thresholds for damage categories (m/s)
    CATEGORIES = {
        'none': (0, 17),       # < 34 kt
        'minor': (17, 26),     # 34-50 kt
        'moderate': (26, 33),  # 50-64 kt
        'significant': (33, 43), # Cat 1-2
        'severe': (43, 50),    # Cat 2-3
        'extreme': (50, 58),   # Cat 3-4
        'catastrophic': (58, np.inf),  # Cat 4-5
    }
    
    def __init__(
        self,
        building_vulnerability: float = 0.5,
        infrastructure_vulnerability: float = 0.3
    ):
        """Initialize damage model.
        
        Parameters
        ----------
        building_vulnerability : float
            Relative vulnerability of buildings (0-1).
        infrastructure_vulnerability : float
            Relative vulnerability of infrastructure (0-1).
        """
        self.building_vuln = building_vulnerability
        self.infra_vuln = infrastructure_vulnerability
    
    def damage_ratio(
        self,
        wind_speed_ms: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute damage ratio from wind speed.
        
        Uses a cubic damage function following Emanuel (2011):
        D ∝ V³ for V > threshold
        
        Parameters
        ----------
        wind_speed_ms : ndarray
            Wind speeds in m/s.
            
        Returns
        -------
        ndarray
            Damage ratios (0 to 1).
        """
        # Threshold below which no significant damage
        V_threshold = 17.0  # ~34 kt
        
        # Normalize wind speed
        V_norm = np.maximum(wind_speed_ms - V_threshold, 0) / 50  # Scale
        
        # Cubic damage function
        damage = self.building_vuln * V_norm ** 3
        
        # Clip to [0, 1]
        return np.clip(damage, 0, 1)
    
    def categorize_damage(
        self,
        wind_speed_ms: NDArray[np.float64]
    ) -> NDArray[np.str_]:
        """Categorize wind speed into damage categories.
        
        Parameters
        ----------
        wind_speed_ms : ndarray
            Wind speeds in m/s.
            
        Returns
        -------
        ndarray
            Category labels.
        """
        categories = np.full_like(wind_speed_ms, 'none', dtype=object)
        
        for cat, (low, high) in self.CATEGORIES.items():
            mask = (wind_speed_ms >= low) & (wind_speed_ms < high)
            categories[mask] = cat
        
        return categories


def compute_wind_field(
    track_point,
    lats: NDArray[np.float64],
    lons: NDArray[np.float64]
) -> WindFieldResult:
    """Convenience function to compute wind field from a track point.
    
    Parameters
    ----------
    track_point : TrackPoint or similar
        Object with latitude, longitude, max_wind_ms, central_pressure_hPa.
    lats, lons : ndarray
        Grid coordinates.
        
    Returns
    -------
    WindFieldResult
        Computed wind field.
    """
    # Estimate RMW if not provided
    rmw_m = getattr(track_point, 'radius_max_wind_km', 50) * 1000
    
    model = HollandWindModel(
        central_pressure_Pa=track_point.central_pressure_hPa * 100,
        radius_max_wind_m=rmw_m,
        latitude_deg=track_point.latitude
    )
    
    return model.wind_field(
        center_lat_deg=track_point.latitude,
        center_lon_deg=track_point.longitude,
        lats_deg=lats,
        lons_deg=lons
    )
