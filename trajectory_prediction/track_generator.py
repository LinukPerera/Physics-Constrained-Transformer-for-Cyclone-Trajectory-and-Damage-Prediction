"""
Track Generator for Cyclone Trajectory Prediction.

This module generates forecast tracks from model outputs, including
uncertainty quantification and ensemble generation.

Output Format
-------------
Tracks are output as sequences of CycloneState objects with:
- Geographic position (lat, lon) via geodesic calculations
- Intensity (maximum sustained wind)
- Pressure (central pressure)
- Structure (radius of max wind, if available)

Uncertainty is represented via:
- Quantile predictions (10th, 50th, 90th percentiles)
- Ensemble of possible tracks
- Confidence ellipses for position
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

import torch

from common.logging_config import get_logger
from common.types import GeoCoordinate, CycloneState, Trajectory
from geospatial.distance_calculations import geodesic_direct

logger = get_logger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation.
    
    Attributes
    ----------
    forecast_hours : int
        Total forecast length in hours.
    time_step_hours : int
        Time step between forecast points.
    num_ensemble_members : int
        Number of ensemble members for uncertainty.
    position_uncertainty_km : float
        Base position uncertainty for ensemble generation.
    intensity_uncertainty_ms : float
        Base intensity uncertainty.
    """
    forecast_hours: int = 120  # 5 days
    time_step_hours: int = 6
    num_ensemble_members: int = 21
    position_uncertainty_km: float = 100.0
    intensity_uncertainty_ms: float = 5.0


@dataclass
class TrackPoint:
    """Single point on a forecast track.
    
    Attributes
    ----------
    valid_time : datetime
        Valid time of the forecast point.
    latitude : float
        Latitude in degrees.
    longitude : float
        Longitude in degrees.
    max_wind_ms : float
        Maximum sustained wind in m/s.
    central_pressure_hPa : float
        Central pressure in hPa.
    radius_max_wind_km : float, optional
        Radius of maximum winds in km.
    position_uncertainty_km : float, optional
        Position uncertainty (radius of confidence circle).
    intensity_uncertainty_ms : float, optional
        Intensity uncertainty.
    """
    valid_time: datetime
    latitude: float
    longitude: float
    max_wind_ms: float
    central_pressure_hPa: float
    radius_max_wind_km: Optional[float] = None
    position_uncertainty_km: Optional[float] = None
    intensity_uncertainty_ms: Optional[float] = None


@dataclass
class ForecastTrack:
    """Complete forecast track with metadata.
    
    Attributes
    ----------
    storm_id : str
        Storm identifier.
    init_time : datetime
        Forecast initialization time.
    points : List[TrackPoint]
        Sequence of forecast points.
    is_ensemble_member : bool
        Whether this is an ensemble member.
    ensemble_id : int, optional
        Ensemble member number.
    """
    storm_id: str
    init_time: datetime
    points: List[TrackPoint]
    is_ensemble_member: bool = False
    ensemble_id: Optional[int] = None
    
    @property
    def lead_times_hours(self) -> List[int]:
        """Lead times for each point."""
        return [
            int((p.valid_time - self.init_time).total_seconds() / 3600)
            for p in self.points
        ]
    
    def get_point_at_lead_time(self, hours: int) -> Optional[TrackPoint]:
        """Get track point at specific lead time."""
        for i, lt in enumerate(self.lead_times_hours):
            if lt == hours:
                return self.points[i]
        return None


class TrackGenerator:
    """Generator for forecast tracks from model outputs.
    
    This class converts raw model outputs (tensors) into structured
    forecast tracks with proper geographic coordinates.
    """
    
    def __init__(self, config: TrajectoryConfig):
        """Initialize track generator.
        
        Parameters
        ----------
        config : TrajectoryConfig
            Configuration for track generation.
        """
        self.config = config
        self._logger = get_logger("TrackGenerator")
    
    def generate_track(
        self,
        position_predictions: NDArray[np.float64],
        intensity_predictions: NDArray[np.float64],
        pressure_predictions: NDArray[np.float64],
        init_time: datetime,
        storm_id: str,
        position_uncertainty: Optional[NDArray[np.float64]] = None,
        intensity_uncertainty: Optional[NDArray[np.float64]] = None
    ) -> ForecastTrack:
        """Generate a forecast track from model predictions.
        
        Parameters
        ----------
        position_predictions : ndarray
            Position predictions (T, 2) in degrees [lat, lon].
        intensity_predictions : ndarray
            Intensity predictions (T,) in m/s.
        pressure_predictions : ndarray
            Pressure predictions (T,) in hPa.
        init_time : datetime
            Forecast initialization time.
        storm_id : str
            Storm identifier.
        position_uncertainty : ndarray, optional
            Position uncertainty (T,) in km.
        intensity_uncertainty : ndarray, optional
            Intensity uncertainty (T,) in m/s.
            
        Returns
        -------
        ForecastTrack
            Generated forecast track.
        """
        num_points = len(position_predictions)
        points = []
        
        for t in range(num_points):
            valid_time = init_time + timedelta(
                hours=self.config.time_step_hours * (t + 1)
            )
            
            point = TrackPoint(
                valid_time=valid_time,
                latitude=float(position_predictions[t, 0]),
                longitude=float(position_predictions[t, 1]),
                max_wind_ms=float(intensity_predictions[t]),
                central_pressure_hPa=float(pressure_predictions[t]),
                position_uncertainty_km=(
                    float(position_uncertainty[t]) 
                    if position_uncertainty is not None else None
                ),
                intensity_uncertainty_ms=(
                    float(intensity_uncertainty[t])
                    if intensity_uncertainty is not None else None
                ),
            )
            points.append(point)
        
        return ForecastTrack(
            storm_id=storm_id,
            init_time=init_time,
            points=points,
            is_ensemble_member=False
        )
    
    def generate_from_quantiles(
        self,
        position_quantiles: NDArray[np.float64],
        intensity_quantiles: NDArray[np.float64],
        pressure_quantiles: NDArray[np.float64],
        init_time: datetime,
        storm_id: str
    ) -> Tuple[ForecastTrack, ForecastTrack, ForecastTrack]:
        """Generate tracks from quantile predictions.
        
        Parameters
        ----------
        position_quantiles : ndarray
            Position quantiles (T, 2, 3) for 10th, 50th, 90th percentiles.
        intensity_quantiles : ndarray
            Intensity quantiles (T, 3).
        pressure_quantiles : ndarray
            Pressure quantiles (T, 3).
        init_time : datetime
            Initialization time.
        storm_id : str
            Storm identifier.
            
        Returns
        -------
        Tuple[ForecastTrack, ForecastTrack, ForecastTrack]
            (lower_bound, median, upper_bound) tracks.
        """
        # Median track (50th percentile)
        median_track = self.generate_track(
            position_quantiles[:, :, 1],
            intensity_quantiles[:, 1],
            pressure_quantiles[:, 1],
            init_time,
            storm_id,
            # Uncertainty from quantile spread
            position_uncertainty=np.sqrt(
                (position_quantiles[:, 0, 2] - position_quantiles[:, 0, 0])**2 +
                (position_quantiles[:, 1, 2] - position_quantiles[:, 1, 0])**2
            ) * 111,  # Convert degrees to km
            intensity_uncertainty=(
                intensity_quantiles[:, 2] - intensity_quantiles[:, 0]
            ) / 2
        )
        
        # Lower bound (10th percentile intensity, 10th position)
        lower_track = self.generate_track(
            position_quantiles[:, :, 0],
            intensity_quantiles[:, 0],
            pressure_quantiles[:, 2],  # Higher pressure = weaker
            init_time,
            f"{storm_id}_q10"
        )
        
        # Upper bound (90th percentile)
        upper_track = self.generate_track(
            position_quantiles[:, :, 2],
            intensity_quantiles[:, 2],
            pressure_quantiles[:, 0],  # Lower pressure = stronger
            init_time,
            f"{storm_id}_q90"
        )
        
        return lower_track, median_track, upper_track


def generate_ensemble_tracks(
    base_track: ForecastTrack,
    config: TrajectoryConfig,
    random_seed: Optional[int] = None
) -> List[ForecastTrack]:
    """Generate ensemble of perturbed tracks.
    
    Parameters
    ----------
    base_track : ForecastTrack
        Base (control) track.
    config : TrajectoryConfig
        Configuration for ensemble generation.
    random_seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    List[ForecastTrack]
        Ensemble of perturbed tracks.
    """
    rng = np.random.default_rng(random_seed)
    ensemble = []
    
    for member_id in range(config.num_ensemble_members):
        perturbed_points = []
        
        for i, point in enumerate(base_track.points):
            # Lead-time dependent uncertainty
            lead_hours = (point.valid_time - base_track.init_time).total_seconds() / 3600
            pos_scale = config.position_uncertainty_km * (1 + lead_hours / 48)
            int_scale = config.intensity_uncertainty_ms * (1 + lead_hours / 72)
            
            # Perturb position
            d_lat = rng.normal(0, pos_scale / 111)  # km to degrees
            d_lon = rng.normal(0, pos_scale / 111 / np.cos(np.radians(point.latitude)))
            
            # Perturb intensity (bounded to be positive)
            d_int = rng.normal(0, int_scale)
            new_int = max(0, point.max_wind_ms + d_int)
            
            # Pressure consistent with intensity
            d_pres = -d_int * 1.1  # Approximate pressure-wind relationship
            
            perturbed_point = TrackPoint(
                valid_time=point.valid_time,
                latitude=point.latitude + d_lat,
                longitude=point.longitude + d_lon,
                max_wind_ms=new_int,
                central_pressure_hPa=point.central_pressure_hPa + d_pres,
                radius_max_wind_km=point.radius_max_wind_km,
            )
            perturbed_points.append(perturbed_point)
        
        ensemble.append(ForecastTrack(
            storm_id=base_track.storm_id,
            init_time=base_track.init_time,
            points=perturbed_points,
            is_ensemble_member=True,
            ensemble_id=member_id
        ))
    
    return ensemble
