"""
Monte Carlo Simulation for Uncertainty Propagation.

This module provides Monte Carlo methods for propagating uncertainty
through the cyclone prediction and impact assessment pipeline.

Purpose
-------
Trajectory and intensity forecasts have inherent uncertainty that
grows with lead time. Monte Carlo simulation allows us to:
1. Sample from the forecast distribution
2. Propagate uncertainty through hazard models
3. Produce probabilistic impact estimates
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Any
import numpy as np
from numpy.typing import NDArray
from datetime import datetime

from common.logging_config import get_logger
from trajectory_prediction.track_generator import ForecastTrack, TrackPoint

logger = get_logger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty modeling.
    
    Attributes
    ----------
    num_samples : int
        Number of Monte Carlo samples.
    position_std_km : float
        Position uncertainty standard deviation in km.
    intensity_std_ms : float
        Intensity uncertainty standard deviation in m/s.
    pressure_std_hPa : float
        Pressure uncertainty standard deviation in hPa.
    correlation_length_hours : float
        Temporal correlation length for error persistence.
    random_seed : Optional[int]
        Random seed for reproducibility.
    """
    num_samples: int = 100
    position_std_km: float = 100.0
    intensity_std_ms: float = 5.0
    pressure_std_hPa: float = 5.0
    correlation_length_hours: float = 12.0
    random_seed: Optional[int] = None


@dataclass
class EnsembleStats:
    """Statistics from ensemble of simulations.
    
    Attributes
    ----------
    mean : ndarray
        Mean values.
    std : ndarray
        Standard deviation.
    percentiles : Dict[int, ndarray]
        Percentile values (e.g., 10th, 50th, 90th).
    samples : ndarray
        Raw samples if retained.
    convergence_diagnostic : float
        Metric for assessing convergence.
    """
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    percentiles: Dict[int, NDArray[np.float64]]
    samples: Optional[NDArray[np.float64]] = None
    convergence_diagnostic: float = 0.0


class MonteCarloSimulator:
    """Monte Carlo simulator for cyclone impact uncertainty.
    
    Generates ensemble of possible scenarios by sampling from
    forecast uncertainty distributions.
    """
    
    def __init__(self, config: UncertaintyConfig):
        """Initialize Monte Carlo simulator.
        
        Parameters
        ----------
        config : UncertaintyConfig
            Configuration for uncertainty modeling.
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self._logger = get_logger("MonteCarloSimulator")
    
    def generate_track_ensemble(
        self,
        base_track: ForecastTrack,
        lead_time_factor: bool = True
    ) -> List[ForecastTrack]:
        """Generate ensemble of perturbed tracks.
        
        Parameters
        ----------
        base_track : ForecastTrack
            Central (deterministic) forecast track.
        lead_time_factor : bool
            Whether to scale uncertainty with lead time.
            
        Returns
        -------
        List[ForecastTrack]
            Ensemble of perturbed tracks.
        """
        ensemble = []
        
        for sample_idx in range(self.config.num_samples):
            perturbed_points = []
            
            # Generate correlated perturbations
            num_points = len(base_track.points)
            pos_perturb = self._generate_correlated_noise(
                num_points, self.config.position_std_km / 111  # km to deg
            )
            int_perturb = self._generate_correlated_noise(
                num_points, self.config.intensity_std_ms
            )
            
            for i, point in enumerate(base_track.points):
                # Scale uncertainty with lead time if requested
                if lead_time_factor:
                    lead_hours = (point.valid_time - base_track.init_time).total_seconds() / 3600
                    scale = 1.0 + lead_hours / 48  # Double by 48 hours
                else:
                    scale = 1.0
                
                new_lat = point.latitude + pos_perturb[i, 0] * scale
                new_lon = point.longitude + pos_perturb[i, 1] * scale
                new_int = max(0, point.max_wind_ms + int_perturb[i] * scale)
                
                # Pressure consistent with intensity
                d_int = new_int - point.max_wind_ms
                new_pres = point.central_pressure_hPa - d_int * 1.1
                
                perturbed_points.append(TrackPoint(
                    valid_time=point.valid_time,
                    latitude=new_lat,
                    longitude=new_lon,
                    max_wind_ms=new_int,
                    central_pressure_hPa=new_pres,
                    radius_max_wind_km=point.radius_max_wind_km,
                ))
            
            ensemble.append(ForecastTrack(
                storm_id=f"{base_track.storm_id}_mc{sample_idx:03d}",
                init_time=base_track.init_time,
                points=perturbed_points,
                is_ensemble_member=True,
                ensemble_id=sample_idx
            ))
        
        return ensemble
    
    def _generate_correlated_noise(
        self,
        length: int,
        std: float
    ) -> NDArray[np.float64]:
        """Generate temporally correlated noise.
        
        Uses AR(1) process for temporal correlation.
        
        Parameters
        ----------
        length : int
            Number of time steps.
        std : float
            Standard deviation of noise.
            
        Returns
        -------
        ndarray
            Correlated noise array.
        """
        # Correlation coefficient from correlation length
        dt = 6  # hours per time step
        rho = np.exp(-dt / self.config.correlation_length_hours)
        
        # Generate AR(1) process
        innovation_std = std * np.sqrt(1 - rho**2)
        noise = np.zeros((length, 2))  # (lat, lon) or similar
        noise[0] = self.rng.normal(0, std, 2)
        
        for t in range(1, length):
            noise[t] = rho * noise[t-1] + self.rng.normal(0, innovation_std, 2)
        
        return noise
    
    def run_simulation(
        self,
        base_track: ForecastTrack,
        impact_function: Callable[[ForecastTrack], Any]
    ) -> EnsembleStats:
        """Run Monte Carlo simulation through impact function.
        
        Parameters
        ----------
        base_track : ForecastTrack
            Central forecast.
        impact_function : Callable
            Function that computes impact from a track.
            
        Returns
        -------
        EnsembleStats
            Statistics of impact ensemble.
        """
        self._logger.info(
            f"Running Monte Carlo with {self.config.num_samples} samples"
        )
        
        # Generate ensemble
        ensemble = self.generate_track_ensemble(base_track)
        
        # Compute impacts for each member
        impacts = []
        for track in ensemble:
            try:
                impact = impact_function(track)
                if isinstance(impact, (int, float)):
                    impacts.append(impact)
                else:
                    impacts.append(np.array(impact))
            except Exception as e:
                self._logger.warning(f"Impact calculation failed: {e}")
        
        if not impacts:
            raise RuntimeError("All Monte Carlo samples failed")
        
        impacts = np.array(impacts)
        
        # Compute statistics
        stats = EnsembleStats(
            mean=np.mean(impacts, axis=0),
            std=np.std(impacts, axis=0),
            percentiles={
                10: np.percentile(impacts, 10, axis=0),
                50: np.percentile(impacts, 50, axis=0),
                90: np.percentile(impacts, 90, axis=0),
            },
            samples=impacts if len(impacts) < 1000 else None,
            convergence_diagnostic=self._compute_convergence(impacts)
        )
        
        return stats
    
    def _compute_convergence(self, samples: NDArray) -> float:
        """Compute convergence diagnostic for sample mean.
        
        Uses relative standard error of the mean.
        """
        if len(samples) < 10:
            return float('inf')
        
        se = np.std(samples) / np.sqrt(len(samples))
        mean_abs = np.abs(np.mean(samples))
        
        if mean_abs < 1e-10:
            return 0.0
        
        return se / mean_abs  # Coefficient of variation of mean
