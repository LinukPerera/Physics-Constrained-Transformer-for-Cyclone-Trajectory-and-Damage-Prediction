"""
Flood Risk Models for Tropical Cyclone Impacts.

This module provides models for:
1. Storm surge estimation
2. Rainfall-runoff modeling
3. Flood extent estimation

CRITICAL: These are simplified models. Operational flood forecasting
requires high-resolution bathymetry, topography, and hydrodynamic models.

References
----------
- Jelesnianski, C.P. (1972). SPLASH model for storm surge.
- SCS (1986). Urban Hydrology for Small Watersheds TR-55.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SurgeResult:
    """Result from storm surge calculation.
    
    Attributes
    ----------
    surge_height_m : ndarray
        Surge height above normal tide in meters.
    total_water_level_m : ndarray
        Total water level including surge and tide.
    inundation_extent_m : ndarray
        Inland penetration distance estimate.
    """
    surge_height_m: NDArray[np.float64]
    total_water_level_m: NDArray[np.float64]
    inundation_extent_m: NDArray[np.float64]


class SurgeModel:
    """Simplified parametric storm surge model.
    
    Based on empirical relationships between storm intensity, size,
    and forward speed. NOT a hydrodynamic model.
    
    Limitations
    -----------
    - Does not resolve complex coastal geometry
    - No wave setup included
    - Assumes open coast (not valid for bays/estuaries)
    
    For operational use, couple with ADCIRC or similar models.
    """
    
    def __init__(
        self,
        offshore_depth_m: float = 50.0,
        coastal_slope: float = 0.001
    ):
        """Initialize surge model.
        
        Parameters
        ----------
        offshore_depth_m : float
            Representative offshore water depth.
        coastal_slope : float
            Average coastal slope (rise/run).
        """
        self.depth = offshore_depth_m
        self.slope = coastal_slope
        self._logger = get_logger("SurgeModel")
    
    def estimate_surge(
        self,
        central_pressure_hPa: float,
        max_wind_ms: float,
        radius_max_wind_km: float,
        forward_speed_ms: float,
        approach_angle_deg: float = 90.0
    ) -> float:
        """Estimate peak storm surge height.
        
        Uses Irish et al. (2008) parameterization:
        η ∝ ΔP × Rm / (g × d)
        
        Parameters
        ----------
        central_pressure_hPa : float
            Central pressure in hPa.
        max_wind_ms : float
            Maximum sustained wind in m/s.
        radius_max_wind_km : float
            Radius of maximum winds in km.
        forward_speed_ms : float
            Storm forward speed in m/s.
        approach_angle_deg : float
            Angle of approach to coast (90 = perpendicular).
            
        Returns
        -------
        float
            Estimated peak surge height in meters.
        """
        # Pressure deficit contribution
        delta_P = (1013 - central_pressure_hPa) * 100  # Pa
        
        # Inverse barometer effect (~10 cm per 10 hPa)
        ib_surge = delta_P / (1025 * 9.81)  # ρg
        
        # Wind setup contribution
        # Simplified: proportional to V² × Rm / depth
        g = 9.81
        wind_factor = (max_wind_ms ** 2) * (radius_max_wind_km * 1000) / (g * self.depth)
        wind_surge = 0.001 * wind_factor  # Empirical coefficient
        
        # Forward speed effect (faster = higher surge)
        speed_factor = 1.0 + 0.02 * (forward_speed_ms - 5)
        
        # Approach angle effect (perpendicular = maximum)
        angle_factor = np.sin(np.radians(approach_angle_deg))
        
        # Total surge
        surge = (ib_surge + wind_surge) * speed_factor * angle_factor
        
        return surge
    
    def surge_profile(
        self,
        peak_surge_m: float,
        distance_from_coast_km: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute surge height profile inland from coast.
        
        Parameters
        ----------
        peak_surge_m : float
            Peak surge at coastline.
        distance_from_coast_km : ndarray
            Distance inland in km (negative = offshore).
            
        Returns
        -------
        ndarray
            Surge height at each distance.
        """
        # Offshore: surge decreases with depth
        offshore = distance_from_coast_km < 0
        
        # Onshore: surge decreases exponentially inland
        onshore = distance_from_coast_km >= 0
        
        surge = np.zeros_like(distance_from_coast_km)
        
        # Offshore decay
        surge[offshore] = peak_surge_m * np.exp(distance_from_coast_km[offshore] / 50)
        
        # Onshore: exponential decay plus slope effect
        decay_scale = 5.0  # km
        surge[onshore] = peak_surge_m * np.exp(-distance_from_coast_km[onshore] / decay_scale)
        
        return np.maximum(surge, 0)


class RainfallRunoffModel:
    """SCS Curve Number rainfall-runoff model.
    
    Estimates runoff from rainfall based on land cover and soil type.
    
    References
    ----------
    USDA-SCS (1986). Urban Hydrology for Small Watersheds. TR-55.
    """
    
    # Curve numbers by land cover and soil group (A, B, C, D)
    CURVE_NUMBERS = {
        'urban': {'A': 77, 'B': 85, 'C': 90, 'D': 92},
        'residential': {'A': 61, 'B': 75, 'C': 83, 'D': 87},
        'forest': {'A': 30, 'B': 55, 'C': 70, 'D': 77},
        'agriculture': {'A': 67, 'B': 78, 'C': 85, 'D': 89},
        'water': {'A': 100, 'B': 100, 'C': 100, 'D': 100},
    }
    
    def __init__(
        self,
        curve_number: float = 80,
        initial_abstraction_ratio: float = 0.2
    ):
        """Initialize runoff model.
        
        Parameters
        ----------
        curve_number : float
            SCS curve number (0-100).
        initial_abstraction_ratio : float
            Ratio of initial abstraction to potential retention.
        """
        self.CN = curve_number
        self.Ia_ratio = initial_abstraction_ratio
        
        # Potential maximum retention (S) in mm
        self.S = 25400 / self.CN - 254
        self.Ia = self.Ia_ratio * self.S
    
    def runoff(self, rainfall_mm: float) -> float:
        """Calculate runoff from rainfall.
        
        Parameters
        ----------
        rainfall_mm : float
            Total rainfall in mm.
            
        Returns
        -------
        float
            Runoff in mm.
        """
        P = rainfall_mm
        Ia = self.Ia
        S = self.S
        
        if P <= Ia:
            return 0.0
        
        Q = (P - Ia) ** 2 / (P - Ia + S)
        return Q
    
    def runoff_array(
        self,
        rainfall_mm: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate runoff for array of rainfall values."""
        return np.vectorize(self.runoff)(rainfall_mm)


class FloodExtentEstimator:
    """Estimate flood extent from water levels and terrain.
    
    Uses simple bathtub model for coastal flooding and
    topographic analysis for riverine flooding.
    """
    
    def __init__(
        self,
        dem: Optional[NDArray[np.float64]] = None,
        dem_resolution_m: float = 30.0
    ):
        """Initialize flood extent estimator.
        
        Parameters
        ----------
        dem : ndarray, optional
            Digital elevation model.
        dem_resolution_m : float
            DEM resolution in meters.
        """
        self.dem = dem
        self.resolution = dem_resolution_m
        self._logger = get_logger("FloodExtentEstimator")
    
    def bathtub_flood(
        self,
        water_level_m: float,
        connected_to_source: bool = True
    ) -> NDArray[np.bool_]:
        """Simple bathtub flood model.
        
        Parameters
        ----------
        water_level_m : float
            Water surface elevation.
        connected_to_source : bool
            Whether to require connectivity to water source.
            
        Returns
        -------
        ndarray
            Boolean mask of flooded areas.
        """
        if self.dem is None:
            raise ValueError("DEM required for flood extent estimation")
        
        flooded = self.dem < water_level_m
        
        if connected_to_source:
            # TODO: Implement connectivity analysis
            pass
        
        return flooded
    
    def estimate_flood_depth(
        self,
        water_level_m: float
    ) -> NDArray[np.float64]:
        """Estimate flood depth at each point.
        
        Parameters
        ----------
        water_level_m : float
            Water surface elevation.
            
        Returns
        -------
        ndarray
            Flood depth in meters (0 where not flooded).
        """
        if self.dem is None:
            raise ValueError("DEM required for flood depth estimation")
        
        depth = water_level_m - self.dem
        return np.maximum(depth, 0)
