"""
Type Definitions with Physical Units for Cyclone Prediction System.

This module defines dataclasses and type aliases that represent physical
quantities with attached units. These types enforce semantic correctness
and provide clear interfaces between modules.

Design Rationale
----------------
Using typed dataclasses instead of raw arrays/dicts provides:
1. Self-documenting code - field names describe the data
2. Compile-time checking with mypy
3. Runtime validation of required fields
4. Clear unit expectations in docstrings
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


@dataclass
class GeoCoordinate:
    """A geographic coordinate on Earth's surface.
    
    This is the fundamental spatial type for the system. All positions
    should be represented using this class to ensure consistent handling.
    
    Attributes
    ----------
    latitude : float
        Geodetic latitude in RADIANS (not degrees). Range: [-π/2, π/2].
    longitude : float
        Geodetic longitude in RADIANS (not degrees). Range: [-π, π].
    altitude : float, optional
        Height above WGS84 ellipsoid in METERS. Default is 0 (sea level).
        
    Notes
    -----
    - Latitude is positive north, negative south.
    - Longitude is positive east, negative west.
    - Using radians internally avoids constant conversion overhead.
    - For display, use the `to_degrees()` method.
    
    Examples
    --------
    >>> import numpy as np
    >>> coord = GeoCoordinate(latitude=np.radians(25.7617), longitude=np.radians(-80.1918))
    >>> lat_deg, lon_deg = coord.to_degrees()
    >>> print(f"Miami: {lat_deg:.4f}°N, {abs(lon_deg):.4f}°W")
    Miami: 25.7617°N, 80.1918°W
    """
    latitude: float  # radians
    longitude: float  # radians
    altitude: float = 0.0  # meters above ellipsoid
    
    def __post_init__(self):
        """Validate coordinate ranges."""
        if not -np.pi/2 <= self.latitude <= np.pi/2:
            raise ValueError(
                f"Latitude {self.latitude} rad out of range [-π/2, π/2]. "
                f"Did you pass degrees instead of radians?"
            )
        # Normalize longitude to [-π, π]
        self.longitude = np.arctan2(np.sin(self.longitude), np.cos(self.longitude))
    
    def to_degrees(self) -> Tuple[float, float]:
        """Convert to degrees for display.
        
        Returns
        -------
        Tuple[float, float]
            (latitude_degrees, longitude_degrees)
        """
        return np.degrees(self.latitude), np.degrees(self.longitude)
    
    @classmethod
    def from_degrees(cls, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> 'GeoCoordinate':
        """Create coordinate from degrees (convenience constructor).
        
        Parameters
        ----------
        lat_deg : float
            Latitude in degrees.
        lon_deg : float
            Longitude in degrees.
        alt_m : float, optional
            Altitude in meters.
            
        Returns
        -------
        GeoCoordinate
            Coordinate with internally stored radians.
        """
        return cls(
            latitude=np.radians(lat_deg),
            longitude=np.radians(lon_deg),
            altitude=alt_m
        )


@dataclass
class AtmosphericState:
    """Atmospheric state vector at a point in space and time.
    
    This represents the environmental conditions relevant to cyclone
    dynamics. All values should be in SI units.
    
    Attributes
    ----------
    position : GeoCoordinate
        The spatial location of this observation.
    timestamp : datetime
        The time of this observation (UTC).
    pressure : float
        Atmospheric pressure in PASCALS.
    temperature : float
        Air temperature in KELVIN.
    u_wind : float
        Eastward wind component in M/S.
    v_wind : float
        Northward wind component in M/S.
    specific_humidity : float, optional
        Specific humidity in KG/KG.
    geopotential_height : float, optional
        Geopotential height in METERS.
        
    Notes
    -----
    Wind components follow meteorological convention:
    - u > 0 means wind blowing towards the east
    - v > 0 means wind blowing towards the north
    """
    position: GeoCoordinate
    timestamp: datetime
    pressure: float  # Pa
    temperature: float  # K
    u_wind: float  # m/s
    v_wind: float  # m/s
    specific_humidity: Optional[float] = None  # kg/kg
    geopotential_height: Optional[float] = None  # m
    
    @property
    def wind_speed(self) -> float:
        """Compute total wind speed in m/s."""
        return np.sqrt(self.u_wind**2 + self.v_wind**2)
    
    @property
    def wind_direction(self) -> float:
        """Compute wind direction in radians (meteorological convention).
        
        Returns the direction FROM which the wind is blowing,
        measured clockwise from north.
        """
        # Meteorological direction (from which wind blows)
        return np.arctan2(-self.u_wind, -self.v_wind) % (2 * np.pi)


@dataclass
class CycloneState:
    """The state of a tropical cyclone at a given instant.
    
    This is the core representation of cyclone position and intensity
    used throughout the prediction system.
    
    Attributes
    ----------
    center : GeoCoordinate
        Position of the cyclone center (eye or circulation center).
    timestamp : datetime
        Time of this state observation (UTC).
    max_wind : float
        Maximum sustained wind speed in M/S (1-minute average).
    central_pressure : float
        Minimum central pressure in PASCALS.
    storm_id : str, optional
        Unique identifier (e.g., "AL092017" for Irma).
    basin : str, optional
        Ocean basin code (e.g., "AL" for Atlantic).
        
    Intensity Structure (optional)
    ------------------------------
    radius_max_wind : float, optional
        Radius of maximum winds in METERS.
    radius_34kt : Tuple[float, float, float, float], optional
        Radii of 34-knot winds in NE, SE, SW, NW quadrants (METERS).
    radius_50kt : Tuple[float, float, float, float], optional
        Radii of 50-knot winds in quadrants (METERS).
    radius_64kt : Tuple[float, float, float, float], optional
        Radii of 64-knot (hurricane force) winds in quadrants (METERS).
        
    Motion
    ------
    storm_motion_u : float, optional
        Eastward motion component in M/S.
    storm_motion_v : float, optional
        Northward motion component in M/S.
    """
    center: GeoCoordinate
    timestamp: datetime
    max_wind: float  # m/s
    central_pressure: float  # Pa
    storm_id: Optional[str] = None
    basin: Optional[str] = None
    
    # Intensity structure
    radius_max_wind: Optional[float] = None  # m
    radius_34kt: Optional[Tuple[float, float, float, float]] = None  # m
    radius_50kt: Optional[Tuple[float, float, float, float]] = None  # m
    radius_64kt: Optional[Tuple[float, float, float, float]] = None  # m
    
    # Motion
    storm_motion_u: Optional[float] = None  # m/s
    storm_motion_v: Optional[float] = None  # m/s
    
    @property
    def storm_speed(self) -> Optional[float]:
        """Compute storm translation speed in m/s."""
        if self.storm_motion_u is not None and self.storm_motion_v is not None:
            return np.sqrt(self.storm_motion_u**2 + self.storm_motion_v**2)
        return None
    
    @property
    def saffir_simpson_category(self) -> str:
        """Classify the storm using Saffir-Simpson scale."""
        wind_ms = self.max_wind
        if wind_ms < 17:
            return "tropical_depression"
        elif wind_ms < 33:
            return "tropical_storm"
        elif wind_ms < 43:
            return "category_1"
        elif wind_ms < 50:
            return "category_2"
        elif wind_ms < 58:
            return "category_3"
        elif wind_ms < 70:
            return "category_4"
        else:
            return "category_5"


@dataclass
class TrajectoryPoint:
    """A single point in a cyclone trajectory forecast.
    
    This extends CycloneState with forecast-specific information
    including uncertainty estimates.
    
    Attributes
    ----------
    state : CycloneState
        The predicted cyclone state.
    forecast_hour : int
        Hours from the initial time (e.g., 0, 6, 12, ..., 120).
    initial_time : datetime
        The analysis/initialization time of the forecast.
    
    Uncertainty Estimates
    ---------------------
    position_uncertainty_major : float, optional
        Semi-major axis of position uncertainty ellipse in METERS.
    position_uncertainty_minor : float, optional
        Semi-minor axis of position uncertainty ellipse in METERS.
    position_uncertainty_angle : float, optional
        Orientation of uncertainty ellipse in RADIANS from north.
    intensity_uncertainty : float, optional
        Uncertainty in max_wind in M/S.
    """
    state: CycloneState
    forecast_hour: int
    initial_time: datetime
    
    # Position uncertainty (error ellipse)
    position_uncertainty_major: Optional[float] = None  # m
    position_uncertainty_minor: Optional[float] = None  # m
    position_uncertainty_angle: Optional[float] = None  # radians
    
    # Intensity uncertainty
    intensity_uncertainty: Optional[float] = None  # m/s


@dataclass
class Trajectory:
    """A complete cyclone trajectory consisting of multiple points.
    
    Attributes
    ----------
    storm_id : str
        Unique identifier for the storm.
    basin : str
        Ocean basin code.
    initial_time : datetime
        Time of forecast initialization.
    points : List[TrajectoryPoint]
        Ordered list of forecast points (by forecast_hour).
    model_version : str
        Version identifier for the prediction model.
    metadata : dict
        Additional metadata (configuration hash, etc.).
    """
    storm_id: str
    basin: str
    initial_time: datetime
    points: List[TrajectoryPoint]
    model_version: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def max_forecast_hour(self) -> int:
        """Get the maximum forecast lead time."""
        return max(p.forecast_hour for p in self.points) if self.points else 0
    
    def get_point_at_hour(self, hour: int) -> Optional[TrajectoryPoint]:
        """Get the trajectory point at a specific forecast hour."""
        for point in self.points:
            if point.forecast_hour == hour:
                return point
        return None


# Type aliases for array types
CoordinateArray = NDArray[np.float64]  # Shape: (N, 2) for lat/lon or (N, 3) for lat/lon/alt
StateVector = NDArray[np.float64]  # Shape: (N, num_state_variables)
TimeSeriesArray = NDArray[np.float64]  # Shape: (T, ...) where T is time dimension
