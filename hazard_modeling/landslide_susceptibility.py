"""
Landslide Susceptibility Models for Cyclone Impact Assessment.

This module provides physics-based landslide susceptibility assessment
based on the infinite slope model and rainfall triggering.

References
----------
- Iverson, R.M. (2000). Landslide triggering by rain infiltration.
- Hammond, C. et al. (1992). Level I Stability Analysis (LISA).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SoilProperties:
    """Physical properties of soil for stability analysis.
    
    Attributes
    ----------
    cohesion_Pa : float
        Effective cohesion in Pascals.
    friction_angle_deg : float
        Internal friction angle in degrees.
    unit_weight_Nm3 : float
        Saturated unit weight in N/m³.
    unit_weight_dry_Nm3 : float
        Dry unit weight in N/m³.
    permeability_ms : float
        Saturated hydraulic conductivity in m/s.
    """
    cohesion_Pa: float = 5000.0  # 5 kPa typical for residual soils
    friction_angle_deg: float = 30.0
    unit_weight_Nm3: float = 19000.0  # ~19 kN/m³
    unit_weight_dry_Nm3: float = 16000.0
    permeability_ms: float = 1e-5


@dataclass
class StabilityResult:
    """Result from slope stability analysis.
    
    Attributes
    ----------
    factor_of_safety : float or ndarray
        Factor of safety (> 1 = stable).
    is_unstable : bool or ndarray
        Whether slope is predicted to fail.
    critical_depth_m : float or ndarray
        Depth of potential failure surface.
    rainfall_threshold_mm : float or ndarray
        Rainfall intensity that would trigger failure.
    """
    factor_of_safety: float
    is_unstable: bool
    critical_depth_m: float
    rainfall_threshold_mm: float


class InfiniteSlopeModel:
    """Infinite slope stability model.
    
    The infinite slope model is appropriate for shallow translational
    landslides on uniform slopes, which are common during heavy rainfall.
    
    Model Equation
    --------------
    FS = (c' + (γ - m×γw)×z×cos²β×tanφ') / (γ×z×sinβ×cosβ)
    
    where:
    - c': Effective cohesion
    - γ: Soil unit weight
    - γw: Water unit weight
    - z: Depth of failure surface
    - β: Slope angle
    - φ': Effective friction angle
    - m: Relative saturation depth (0-1)
    
    Assumptions
    -----------
    - Planar failure surface parallel to slope
    - Infinite slope length (end effects negligible)
    - Homogeneous soil properties
    - Hydrostatic pore pressure
    """
    
    WATER_UNIT_WEIGHT = 9810  # N/m³
    
    def __init__(self, soil: SoilProperties):
        """Initialize infinite slope model.
        
        Parameters
        ----------
        soil : SoilProperties
            Soil physical properties.
        """
        self.soil = soil
        self._logger = get_logger("InfiniteSlopeModel")
    
    def factor_of_safety(
        self,
        slope_angle_deg: float,
        depth_m: float,
        saturation_ratio: float = 0.0
    ) -> float:
        """Calculate factor of safety.
        
        Parameters
        ----------
        slope_angle_deg : float
            Slope angle in degrees.
        depth_m : float
            Depth to failure surface in meters.
        saturation_ratio : float
            Ratio of saturated depth to total depth (0-1).
            
        Returns
        -------
        float
            Factor of safety.
        """
        if depth_m <= 0 or slope_angle_deg <= 0:
            return float('inf')
        
        beta = np.radians(slope_angle_deg)
        phi = np.radians(self.soil.friction_angle_deg)
        
        c = self.soil.cohesion_Pa
        gamma = self.soil.unit_weight_Nm3
        gamma_w = self.WATER_UNIT_WEIGHT
        z = depth_m
        m = saturation_ratio
        
        # Driving force (gravity component parallel to slope)
        driving = gamma * z * np.sin(beta) * np.cos(beta)
        
        # Resisting force (cohesion + friction)
        effective_stress = (gamma - m * gamma_w) * z * np.cos(beta)**2
        resisting = c + effective_stress * np.tan(phi)
        
        if driving <= 0:
            return float('inf')
        
        return resisting / driving
    
    def critical_saturation(
        self,
        slope_angle_deg: float,
        depth_m: float
    ) -> float:
        """Find saturation ratio that gives FS = 1.
        
        Parameters
        ----------
        slope_angle_deg : float
            Slope angle.
        depth_m : float
            Depth to failure surface.
            
        Returns
        -------
        float
            Critical saturation ratio (0-1), or None if always stable.
        """
        # FS = 1 solution for m
        beta = np.radians(slope_angle_deg)
        phi = np.radians(self.soil.friction_angle_deg)
        
        c = self.soil.cohesion_Pa
        gamma = self.soil.unit_weight_Nm3
        gamma_w = self.WATER_UNIT_WEIGHT
        z = depth_m
        
        # From FS = 1: c + (γ - m×γw)×z×cos²β×tanφ = γ×z×sinβ×cosβ
        # Solve for m
        numerator = (c / (z * np.cos(beta)**2 * np.tan(phi)) + 
                    gamma - gamma * np.tan(beta) / np.tan(phi))
        denominator = gamma_w
        
        m_critical = numerator / denominator
        
        return np.clip(m_critical, 0, 1)
    
    def analyze_grid(
        self,
        slope_deg: NDArray[np.float64],
        depth_m: float = 2.0,
        saturation: float = 0.5
    ) -> NDArray[np.float64]:
        """Analyze stability for a grid of slopes.
        
        Parameters
        ----------
        slope_deg : ndarray
            Grid of slope angles in degrees.
        depth_m : float
            Assumed failure depth.
        saturation : float
            Assumed saturation ratio.
            
        Returns
        -------
        ndarray
            Factor of safety at each point.
        """
        return np.vectorize(
            lambda s: self.factor_of_safety(s, depth_m, saturation)
        )(slope_deg)


class LandslideSusceptibility:
    """Landslide susceptibility assessment combining multiple factors.
    
    Considers:
    - Slope angle
    - Rainfall intensity/duration
    - Soil type
    - Land cover
    """
    
    def __init__(
        self,
        slope_model: Optional[InfiniteSlopeModel] = None
    ):
        """Initialize susceptibility model.
        
        Parameters
        ----------
        slope_model : InfiniteSlopeModel, optional
            Slope stability model. Uses default soil if not provided.
        """
        self.slope_model = slope_model or InfiniteSlopeModel(SoilProperties())
        self._logger = get_logger("LandslideSusceptibility")
    
    def rainfall_trigger_threshold(
        self,
        slope_deg: float,
        soil_depth_m: float = 2.0
    ) -> float:
        """Estimate rainfall threshold for landslide triggering.
        
        Parameters
        ----------
        slope_deg : float
            Slope angle.
        soil_depth_m : float
            Soil depth.
            
        Returns
        -------
        float
            Approximate rainfall threshold in mm/day.
        """
        # Find critical saturation
        m_crit = self.slope_model.critical_saturation(slope_deg, soil_depth_m)
        
        if m_crit >= 1:
            # Always unstable even when dry
            return 0.0
        
        if m_crit <= 0:
            # Stable even when fully saturated
            return float('inf')
        
        # Estimate rainfall needed to achieve critical saturation
        # Simplified: assume infiltration rate and drainage
        k = self.slope_model.soil.permeability_ms
        porosity = 0.4  # Assumed
        
        # Water volume needed to saturate to m_crit
        water_needed_m = m_crit * soil_depth_m * porosity
        
        # Convert to mm/day (very rough estimate)
        # Assumes some drainage and evaporation
        rainfall_mm = water_needed_m * 1000 / 0.3  # 30% becomes saturation
        
        return rainfall_mm
    
    def susceptibility_map(
        self,
        slope_deg: NDArray[np.float64],
        rainfall_mm: float
    ) -> NDArray[np.str_]:
        """Create susceptibility classification map.
        
        Parameters
        ----------
        slope_deg : ndarray
            Grid of slope angles.
        rainfall_mm : float
            Anticipated rainfall in mm.
            
        Returns
        -------
        ndarray
            Classification: 'low', 'moderate', 'high', 'very_high'.
        """
        # Estimate saturation from rainfall (simplified)
        max_saturation = min(rainfall_mm / 200, 1.0)  # ~200mm to saturate
        
        # Calculate factor of safety
        fs = self.slope_model.analyze_grid(slope_deg, saturation=max_saturation)
        
        # Classify
        susceptibility = np.full_like(slope_deg, 'low', dtype=object)
        susceptibility[fs < 1.5] = 'moderate'
        susceptibility[fs < 1.2] = 'high'
        susceptibility[fs < 1.0] = 'very_high'
        
        return susceptibility
