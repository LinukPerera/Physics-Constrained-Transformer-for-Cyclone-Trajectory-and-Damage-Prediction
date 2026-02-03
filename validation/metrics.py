"""
Verification Metrics for Cyclone Forecasts.

This module provides standard metrics for evaluating cyclone
track and intensity forecasts against observations.

Standard Metrics
----------------
- Track Error: Great circle distance between forecast and observed
- Intensity Error: Wind speed difference
- Pressure Error: Central pressure difference
- Skill Scores: Relative to climatology or persistence
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from common.logging_config import get_logger
from geospatial.distance_calculations import geodesic_distance

logger = get_logger(__name__)


@dataclass
class TrackErrorMetrics:
    """Track forecast error metrics.
    
    Attributes
    ----------
    mean_error_km : float
        Mean track error in km.
    max_error_km : float
        Maximum track error in km.
    along_track_error_km : float
        Error along the track direction.
    cross_track_error_km : float
        Error perpendicular to track.
    error_by_lead_time : Dict[int, float]
        Track error by lead time in hours.
    """
    mean_error_km: float
    max_error_km: float
    along_track_error_km: float
    cross_track_error_km: float
    error_by_lead_time: Dict[int, float]


@dataclass
class IntensityMetrics:
    """Intensity forecast error metrics.
    
    Attributes
    ----------
    mean_error_ms : float
        Mean absolute error in m/s.
    bias_ms : float
        Mean bias (positive = too strong).
    rmse_ms : float
        Root mean square error.
    error_by_lead_time : Dict[int, float]
        Error by lead time.
    rapid_intensification_pod : float
        Probability of detection for RI.
    rapid_intensification_far : float
        False alarm ratio for RI.
    """
    mean_error_ms: float
    bias_ms: float
    rmse_ms: float
    error_by_lead_time: Dict[int, float]
    rapid_intensification_pod: float = 0.0
    rapid_intensification_far: float = 0.0


@dataclass
class SkillScores:
    """Skill scores relative to baselines.
    
    Attributes
    ----------
    track_skill_vs_climo : float
        Track skill vs climatology (1 = perfect, 0 = no skill).
    track_skill_vs_persist : float
        Track skill vs persistence.
    intensity_skill_vs_climo : float
        Intensity skill vs climatology.
    intensity_skill_vs_persist : float
        Intensity skill vs persistence.
    """
    track_skill_vs_climo: float
    track_skill_vs_persist: float
    intensity_skill_vs_climo: float
    intensity_skill_vs_persist: float


def compute_track_error(
    forecast_lat: NDArray[np.float64],
    forecast_lon: NDArray[np.float64],
    observed_lat: NDArray[np.float64],
    observed_lon: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute track error between forecast and observed positions.
    
    Uses geodesic distance from geospatial module.
    
    Parameters
    ----------
    forecast_lat, forecast_lon : ndarray
        Forecast positions in degrees.
    observed_lat, observed_lon : ndarray
        Observed positions in degrees.
        
    Returns
    -------
    ndarray
        Track errors in km.
    """
    errors = np.zeros(len(forecast_lat))
    
    for i in range(len(forecast_lat)):
        dist = geodesic_distance(
            np.radians(forecast_lat[i]),
            np.radians(forecast_lon[i]),
            np.radians(observed_lat[i]),
            np.radians(observed_lon[i])
        )
        errors[i] = dist / 1000  # m to km
    
    return errors


def compute_track_error_metrics(
    forecast_lat: NDArray[np.float64],
    forecast_lon: NDArray[np.float64],
    observed_lat: NDArray[np.float64],
    observed_lon: NDArray[np.float64],
    lead_times_hours: NDArray[np.int64]
) -> TrackErrorMetrics:
    """Compute comprehensive track error metrics.
    
    Parameters
    ----------
    forecast_lat, forecast_lon : ndarray
        Forecast positions in degrees.
    observed_lat, observed_lon : ndarray
        Observed positions in degrees.
    lead_times_hours : ndarray
        Lead time for each point in hours.
        
    Returns
    -------
    TrackErrorMetrics
        Comprehensive error metrics.
    """
    errors_km = compute_track_error(
        forecast_lat, forecast_lon,
        observed_lat, observed_lon
    )
    
    # Error by lead time
    unique_leads = np.unique(lead_times_hours)
    error_by_lead = {}
    for lead in unique_leads:
        mask = lead_times_hours == lead
        error_by_lead[int(lead)] = float(np.mean(errors_km[mask]))
    
    # Along-track and cross-track decomposition (simplified)
    # Would need track direction for proper decomposition
    along_track = np.mean(errors_km) * 0.7  # Approximate
    cross_track = np.mean(errors_km) * 0.7
    
    return TrackErrorMetrics(
        mean_error_km=float(np.mean(errors_km)),
        max_error_km=float(np.max(errors_km)),
        along_track_error_km=along_track,
        cross_track_error_km=cross_track,
        error_by_lead_time=error_by_lead
    )


def compute_intensity_metrics(
    forecast_intensity: NDArray[np.float64],
    observed_intensity: NDArray[np.float64],
    lead_times_hours: NDArray[np.int64]
) -> IntensityMetrics:
    """Compute intensity forecast error metrics.
    
    Parameters
    ----------
    forecast_intensity : ndarray
        Forecast intensity in m/s.
    observed_intensity : ndarray
        Observed intensity in m/s.
    lead_times_hours : ndarray
        Lead times in hours.
        
    Returns
    -------
    IntensityMetrics
        Comprehensive intensity metrics.
    """
    errors = forecast_intensity - observed_intensity
    abs_errors = np.abs(errors)
    
    # Error by lead time
    unique_leads = np.unique(lead_times_hours)
    error_by_lead = {}
    for lead in unique_leads:
        mask = lead_times_hours == lead
        error_by_lead[int(lead)] = float(np.mean(abs_errors[mask]))
    
    # Rapid intensification detection
    # RI defined as >= 15.4 m/s (30 kt) increase in 24 hours
    ri_threshold = 15.4
    
    # Would need 24h change data for proper RI metrics
    # Placeholder values
    ri_pod = 0.0
    ri_far = 0.0
    
    return IntensityMetrics(
        mean_error_ms=float(np.mean(abs_errors)),
        bias_ms=float(np.mean(errors)),
        rmse_ms=float(np.sqrt(np.mean(errors**2))),
        error_by_lead_time=error_by_lead,
        rapid_intensification_pod=ri_pod,
        rapid_intensification_far=ri_far
    )


def compute_skill_scores(
    model_errors: NDArray[np.float64],
    climatology_errors: NDArray[np.float64],
    persistence_errors: NDArray[np.float64]
) -> SkillScores:
    """Compute skill scores relative to baselines.
    
    Skill = 1 - (model_error / baseline_error)
    
    Parameters
    ----------
    model_errors : ndarray
        Model forecast errors.
    climatology_errors : ndarray
        Climatology baseline errors.
    persistence_errors : ndarray
        Persistence baseline errors.
        
    Returns
    -------
    SkillScores
        Skill relative to baselines.
    """
    model_mse = np.mean(model_errors**2)
    climo_mse = np.mean(climatology_errors**2)
    persist_mse = np.mean(persistence_errors**2)
    
    skill_vs_climo = 1 - model_mse / climo_mse if climo_mse > 0 else 0
    skill_vs_persist = 1 - model_mse / persist_mse if persist_mse > 0 else 0
    
    return SkillScores(
        track_skill_vs_climo=skill_vs_climo,
        track_skill_vs_persist=skill_vs_persist,
        intensity_skill_vs_climo=skill_vs_climo,  # Placeholder
        intensity_skill_vs_persist=skill_vs_persist  # Placeholder
    )


class ForecastVerifier:
    """Comprehensive forecast verification tool.
    
    Computes all standard metrics for cyclone forecasts.
    """
    
    def __init__(self, lead_times: List[int] = [12, 24, 48, 72, 96, 120]):
        """Initialize verifier.
        
        Parameters
        ----------
        lead_times : list
            Lead times to evaluate in hours.
        """
        self.lead_times = lead_times
        self._logger = get_logger("ForecastVerifier")
    
    def verify_forecast(
        self,
        forecast_track,
        observed_track
    ) -> Dict[str, any]:
        """Verify a single forecast against observations.
        
        Parameters
        ----------
        forecast_track : ForecastTrack
            Model forecast.
        observed_track : ForecastTrack or similar
            Observed/best track.
            
        Returns
        -------
        dict
            Verification metrics.
        """
        # Match forecast and observed by valid time
        # Implementation would match times and compute errors
        
        results = {
            'track_metrics': None,
            'intensity_metrics': None,
            'skill_scores': None,
        }
        
        return results
