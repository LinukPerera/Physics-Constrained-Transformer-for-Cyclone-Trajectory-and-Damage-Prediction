"""
Physical Constants for Cyclone Modeling.

This module provides physical constants with their uncertainty bounds and sources.
All constants are defined with SI units and traceable to authoritative sources.

References
----------
- WGS84 parameters: NIMA TR8350.2, Third Edition, 2000
- Atmospheric constants: AMS Glossary of Meteorology
- Thermodynamic constants: IAPWS-IF97
"""

from dataclasses import dataclass
from typing import Final
import numpy as np


@dataclass(frozen=True)
class Constant:
    """A physical constant with uncertainty and provenance.
    
    Attributes
    ----------
    value : float
        The nominal value of the constant.
    uncertainty : float
        The standard uncertainty (1-sigma) of the constant.
    unit : str
        The SI unit of the constant.
    source : str
        Reference for the constant value.
    description : str
        Human-readable description of the constant.
    """
    value: float
    uncertainty: float
    unit: str
    source: str
    description: str


class PhysicalConstants:
    """Registry of physical constants used throughout the system.
    
    All constants are class attributes with full metadata including
    uncertainty bounds and authoritative sources.
    
    Earth Geometry (WGS84)
    ----------------------
    These constants define the reference ellipsoid used for all
    geodetic calculations. The WGS84 ellipsoid is the standard for
    GPS and global applications.
    
    Atmospheric Physics
    -------------------
    Constants for atmospheric dynamics, thermodynamics, and
    cyclone-specific calculations.
    """
    
    # =========================================================================
    # WGS84 Ellipsoid Parameters
    # Reference: NIMA TR8350.2, Third Edition, 2000
    # =========================================================================
    
    EARTH_SEMI_MAJOR_AXIS: Final[Constant] = Constant(
        value=6_378_137.0,
        uncertainty=0.0,  # Defined exactly
        unit="m",
        source="WGS84, NIMA TR8350.2",
        description="Semi-major axis (equatorial radius) of WGS84 ellipsoid"
    )
    
    EARTH_SEMI_MINOR_AXIS: Final[Constant] = Constant(
        value=6_356_752.314245,
        uncertainty=0.0001,
        unit="m",
        source="WGS84, NIMA TR8350.2",
        description="Semi-minor axis (polar radius) of WGS84 ellipsoid"
    )
    
    EARTH_FLATTENING: Final[Constant] = Constant(
        value=1.0 / 298.257223563,
        uncertainty=0.0,  # Defined exactly
        unit="dimensionless",
        source="WGS84, NIMA TR8350.2",
        description="Flattening of WGS84 ellipsoid: f = (a - b) / a"
    )
    
    EARTH_ECCENTRICITY_SQUARED: Final[Constant] = Constant(
        value=0.00669437999014,
        uncertainty=1e-14,
        unit="dimensionless",
        source="WGS84, NIMA TR8350.2 (derived)",
        description="First eccentricity squared: e² = (a² - b²) / a²"
    )
    
    EARTH_SECOND_ECCENTRICITY_SQUARED: Final[Constant] = Constant(
        value=0.00673949674228,
        uncertainty=1e-14,
        unit="dimensionless",
        source="WGS84, NIMA TR8350.2 (derived)",
        description="Second eccentricity squared: e'² = (a² - b²) / b²"
    )
    
    EARTH_MEAN_RADIUS: Final[Constant] = Constant(
        value=6_371_008.8,
        uncertainty=0.1,
        unit="m",
        source="IUGG mean radius",
        description="Mean radius of Earth (for reference only, not for calculations)"
    )
    
    # =========================================================================
    # Earth Rotation and Gravity
    # =========================================================================
    
    EARTH_ANGULAR_VELOCITY: Final[Constant] = Constant(
        value=7.2921150e-5,
        uncertainty=1e-14,
        unit="rad/s",
        source="IERS Conventions (2010)",
        description="Earth's angular velocity of rotation"
    )
    
    STANDARD_GRAVITY: Final[Constant] = Constant(
        value=9.80665,
        uncertainty=0.0,  # Defined exactly
        unit="m/s²",
        source="ISO 80000-3:2006",
        description="Standard acceleration due to gravity"
    )
    
    # =========================================================================
    # Atmospheric Constants
    # =========================================================================
    
    DRY_AIR_GAS_CONSTANT: Final[Constant] = Constant(
        value=287.058,
        uncertainty=0.001,
        unit="J/(kg·K)",
        source="AMS Glossary",
        description="Specific gas constant for dry air"
    )
    
    WATER_VAPOR_GAS_CONSTANT: Final[Constant] = Constant(
        value=461.5,
        uncertainty=0.1,
        unit="J/(kg·K)",
        source="AMS Glossary",
        description="Specific gas constant for water vapor"
    )
    
    DRY_AIR_SPECIFIC_HEAT_CP: Final[Constant] = Constant(
        value=1004.0,
        uncertainty=1.0,
        unit="J/(kg·K)",
        source="AMS Glossary",
        description="Specific heat of dry air at constant pressure"
    )
    
    LATENT_HEAT_VAPORIZATION: Final[Constant] = Constant(
        value=2.501e6,
        uncertainty=1e3,
        unit="J/kg",
        source="AMS Glossary (at 0°C)",
        description="Latent heat of vaporization of water at 0°C"
    )
    
    STANDARD_SEA_LEVEL_PRESSURE: Final[Constant] = Constant(
        value=101325.0,
        uncertainty=0.0,  # Defined exactly
        unit="Pa",
        source="ISO 2533:1975",
        description="Standard atmospheric pressure at sea level"
    )
    
    STANDARD_SEA_LEVEL_TEMPERATURE: Final[Constant] = Constant(
        value=288.15,
        uncertainty=0.0,  # Defined exactly
        unit="K",
        source="ISO 2533:1975",
        description="Standard temperature at sea level (15°C)"
    )
    
    # =========================================================================
    # Coriolis and Beta Parameters
    # =========================================================================
    
    @staticmethod
    def coriolis_parameter(latitude_rad: float) -> float:
        """Compute Coriolis parameter f at a given latitude.
        
        Parameters
        ----------
        latitude_rad : float
            Latitude in radians.
            
        Returns
        -------
        float
            Coriolis parameter f in s⁻¹.
            
        Notes
        -----
        f = 2 * Ω * sin(φ), where Ω is Earth's angular velocity.
        """
        omega = PhysicalConstants.EARTH_ANGULAR_VELOCITY.value
        return 2.0 * omega * np.sin(latitude_rad)
    
    @staticmethod
    def beta_parameter(latitude_rad: float) -> float:
        """Compute beta parameter (df/dy) at a given latitude.
        
        Parameters
        ----------
        latitude_rad : float
            Latitude in radians.
            
        Returns
        -------
        float
            Beta parameter in (m·s)⁻¹.
            
        Notes
        -----
        β = (2 * Ω * cos(φ)) / a, where a is Earth's radius.
        This approximation uses the mean radius; for precise work,
        use the radius of curvature at the latitude.
        """
        omega = PhysicalConstants.EARTH_ANGULAR_VELOCITY.value
        a = PhysicalConstants.EARTH_SEMI_MAJOR_AXIS.value
        return (2.0 * omega * np.cos(latitude_rad)) / a
    
    # =========================================================================
    # Cyclone-Specific Constants
    # =========================================================================
    
    SAFFIR_SIMPSON_THRESHOLDS: Final[dict] = {
        "tropical_depression": (0, 17),      # m/s
        "tropical_storm": (17, 33),          # m/s
        "category_1": (33, 43),              # m/s
        "category_2": (43, 50),              # m/s
        "category_3": (50, 58),              # m/s
        "category_4": (58, 70),              # m/s
        "category_5": (70, float('inf')),    # m/s
    }
    
    KNOT_TO_MS: Final[Constant] = Constant(
        value=0.514444,
        uncertainty=0.0,  # Defined exactly
        unit="m/s per knot",
        source="IEEE/ASTM SI 10-2016",
        description="Conversion factor from knots to meters per second"
    )
    
    NAUTICAL_MILE_TO_M: Final[Constant] = Constant(
        value=1852.0,
        uncertainty=0.0,  # Defined exactly
        unit="m per nmi",
        source="IEEE/ASTM SI 10-2016",
        description="Conversion factor from nautical miles to meters"
    )
