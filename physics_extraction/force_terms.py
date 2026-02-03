"""
Force Terms for Cyclone Trajectory Dynamics.

This module computes the various forcing terms that influence tropical
cyclone motion. These are physics-based calculations used to:
1. Provide interpretable features to the ML model
2. Constrain predictions to physically plausible trajectories
3. Enable diagnostic analysis of model predictions

Physical Framework
------------------
Cyclone motion can be decomposed into:
1. Environmental steering (advection by mean flow)
2. Beta drift (Coriolis gradient forcing)
3. Land interaction effects
4. Trochoidal oscillations (higher-order effects, not modeled here)

References
----------
- Holland, G.J. (1983). Tropical cyclone motion.
- Fiorino, M. & Elsberry, R.L. (1989). Beta drift paper.
- Chan, J.C.L. (2005). Physics of tropical cyclone motion. Ann. Rev.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import xarray as xr

from common.constants import PhysicalConstants
from common.types import GeoCoordinate, CycloneState
from common.logging_config import get_logger
from geospatial.distance_calculations import geodesic_inverse

logger = get_logger(__name__)


@dataclass
class CycloneForces:
    """Collection of forces acting on a tropical cyclone.
    
    All forces are per unit mass (accelerations) in m/s².
    
    Attributes
    ----------
    steering_u : float
        Eastward steering velocity in m/s.
    steering_v : float
        Northward steering velocity in m/s.
    beta_drift_u : float
        Eastward beta drift velocity in m/s.
    beta_drift_v : float
        Northward beta drift velocity in m/s.
    land_friction_u : float
        Eastward friction deceleration effect in m/s/6h.
    land_friction_v : float
        Northward friction deceleration effect in m/s/6h.
    """
    steering_u: float
    steering_v: float
    beta_drift_u: float
    beta_drift_v: float
    land_friction_u: float
    land_friction_v: float
    
    @property
    def total_velocity_tendency(self) -> Tuple[float, float]:
        """Net velocity tendency from all forces."""
        u = self.steering_u + self.beta_drift_u + self.land_friction_u
        v = self.steering_v + self.beta_drift_v + self.land_friction_v
        return u, v


def compute_environmental_steering(
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    storm_center: GeoCoordinate,
    averaging_radius_km: float = 500.0,
    layers: Optional[Dict[float, float]] = None
) -> Tuple[float, float]:
    """Compute environmental steering flow.
    
    The steering flow is the deep-layer mean wind that advects the
    cyclone. Typically computed as a weighted average over multiple
    layers, excluding the cyclone's own circulation.
    
    Parameters
    ----------
    u_wind, v_wind : xr.DataArray
        Wind components on pressure levels.
    storm_center : GeoCoordinate
        Current cyclone center position.
    averaging_radius_km : float
        Radius for spatial averaging (to exclude inner core).
    layers : dict, optional
        Mapping from pressure level (hPa) to weight.
        Default: 850-200 hPa average.
        
    Returns
    -------
    Tuple[float, float]
        (u_steer, v_steer) steering velocity in m/s.
        
    Notes
    -----
    Common steering layer definitions:
    - Shallow systems: 850-700 hPa
    - Deep systems: 850-200 hPa
    - Intensity-weighted: weight by layer depth and vortex depth
    
    References
    ----------
    Velden, C.S. & Leslie, L.M. (1991). The basic relationship between
    tropical cyclone intensity and steering.
    """
    if layers is None:
        # Default deep-layer mean
        layers = {
            850: 0.2,
            700: 0.2,
            500: 0.2,
            300: 0.2,
            200: 0.2,
        }
    
    # Extract storm location
    lat_deg, lon_deg = storm_center.to_degrees()
    
    # Create annulus mask (exclude r < 200 km, include r < averaging_radius)
    lats = u_wind['latitude'].values
    lons = u_wind['longitude'].values
    
    # Compute distances from storm center
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    from geospatial.distance_calculations import geodesic_distance_batch
    
    lat_center_rad = np.full_like(lat_grid, storm_center.latitude)
    lon_center_rad = np.full_like(lon_grid, storm_center.longitude)
    
    distances = geodesic_distance_batch(
        np.radians(lat_grid),
        np.radians(lon_grid),
        lat_center_rad,
        lon_center_rad
    )
    
    # Annulus mask: 200 km < r < averaging_radius
    inner_radius = 200_000  # 200 km in meters
    outer_radius = averaging_radius_km * 1000
    
    mask = (distances >= inner_radius) & (distances <= outer_radius)
    
    # Compute layer-weighted average
    u_steer = 0.0
    v_steer = 0.0
    total_weight = 0.0
    
    if 'level' in u_wind.dims:
        for level_hPa, weight in layers.items():
            if level_hPa in u_wind['level'].values:
                u_level = u_wind.sel(level=level_hPa).values
                v_level = v_wind.sel(level=level_hPa).values
                
                # Apply spatial average within annulus
                u_steer += weight * np.nanmean(u_level[mask])
                v_steer += weight * np.nanmean(v_level[mask])
                total_weight += weight
    else:
        # Single level data
        u_steer = np.nanmean(u_wind.values[mask])
        v_steer = np.nanmean(v_wind.values[mask])
        total_weight = 1.0
    
    if total_weight > 0:
        u_steer /= total_weight
        v_steer /= total_weight
    
    return float(u_steer), float(v_steer)


def compute_beta_drift(
    storm_center: GeoCoordinate,
    max_wind_ms: float,
    radius_max_wind_km: float = 50.0
) -> Tuple[float, float]:
    """Compute beta drift velocity.
    
    Beta drift arises from the interaction of the cyclone's circulation
    with the latitudinal gradient of the Coriolis parameter (β = df/dy).
    
    Parameters
    ----------
    storm_center : GeoCoordinate
        Current cyclone center position.
    max_wind_ms : float
        Maximum sustained wind speed in m/s.
    radius_max_wind_km : float
        Radius of maximum winds in km.
        
    Returns
    -------
    Tuple[float, float]
        (u_beta, v_beta) beta drift velocity in m/s.
        
    Notes
    -----
    Beta drift is typically:
    - Westward and poleward in both hemispheres
    - Magnitude: 1-3 m/s for typical storms
    - Increases with storm size and intensity
    
    The empirical formula used here is from Fiorino & Elsberry (1989):
    
    V_beta ≈ 0.3 × β × R_max² × sin(45°)
    
    with β = (2Ω cos φ)/a
    
    References
    ----------
    Fiorino, M. & Elsberry, R.L. (1989). Some aspects of vortex structure.
    """
    lat_rad = storm_center.latitude
    
    # Beta parameter at storm latitude
    beta = PhysicalConstants.beta_parameter(lat_rad)
    
    # Radius of maximum winds in meters
    R_max = radius_max_wind_km * 1000
    
    # Vortex Rossby number (characterizes nonlinearity)
    # Simplified: use max wind / (f × R_max)
    f = PhysicalConstants.coriolis_parameter(lat_rad)
    
    if np.abs(f) < 1e-10:
        # Near equator: beta drift formulation breaks down
        return 0.0, 0.0
    
    # Beta drift velocity (Fiorino & Elsberry approximation)
    # Typical magnitude: 1-3 m/s
    V_beta = 0.3 * beta * R_max**2 * np.sqrt(2) / 2  # sin(45°) term
    
    # Direction: westward and poleward in both hemispheres
    # In Northern Hemisphere: u < 0 (west), v > 0 (north)
    # In Southern Hemisphere: u < 0 (west), v < 0 (south)
    hemisphere = np.sign(lat_rad)
    
    # Cap at physically reasonable values
    V_beta = min(V_beta, 5.0)  # Max 5 m/s beta drift
    
    u_beta = -V_beta * 0.7  # Westward component
    v_beta = V_beta * 0.7 * hemisphere  # Poleward component
    
    return float(u_beta), float(v_beta)


def compute_land_friction(
    storm_center: GeoCoordinate,
    land_mask: Optional[NDArray[np.bool_]] = None,
    distance_to_coast_km: float = 0.0
) -> Tuple[float, float]:
    """Compute land friction effects on storm motion.
    
    When a cyclone approaches land, surface friction increases and
    can affect both intensity and motion. This function estimates
    the friction-induced deceleration.
    
    Parameters
    ----------
    storm_center : GeoCoordinate
        Current cyclone center position.
    land_mask : ndarray, optional
        Boolean mask where True indicates land.
    distance_to_coast_km : float
        Distance to nearest coastline in km (negative = over land).
        
    Returns
    -------
    Tuple[float, float]
        (du_friction, dv_friction) friction-induced velocity change in m/s.
        
    Notes
    -----
    This is a simplified parameterization. Full treatment requires:
    - Surface roughness fields
    - Boundary layer physics
    - Storm structure details
    
    References
    ----------
    Kaplan & DeMaria (2003). Land decay model.
    """
    # Simplified friction model
    # Only applies when close to or over land
    
    if distance_to_coast_km > 100:
        # Far from land: no friction effect
        return 0.0, 0.0
    
    if distance_to_coast_km > 0:
        # Approaching land: small effect
        friction_factor = 0.1 * (100 - distance_to_coast_km) / 100
    else:
        # Over land: significant friction
        # Friction increases as storm moves inland
        inland_distance = -distance_to_coast_km
        friction_factor = min(0.5, 0.2 + 0.01 * inland_distance)
    
    # Friction acts opposite to motion
    # Without motion information, we can't compute this fully
    # Return zeros as placeholder
    return 0.0, 0.0


class CycloneForceCalculator:
    """Calculator for all forces acting on a tropical cyclone.
    
    This class combines all force terms into a unified interface
    for trajectory prediction.
    """
    
    def __init__(
        self,
        steering_radius_km: float = 500.0,
        steering_layers: Optional[Dict[float, float]] = None
    ):
        """Initialize force calculator.
        
        Parameters
        ----------
        steering_radius_km : float
            Radius for steering flow averaging.
        steering_layers : dict, optional
            Pressure level weights for steering.
        """
        self.steering_radius_km = steering_radius_km
        self.steering_layers = steering_layers
        self._logger = get_logger("CycloneForceCalculator")
    
    def compute_all_forces(
        self,
        state: CycloneState,
        u_wind: xr.DataArray,
        v_wind: xr.DataArray,
        land_mask: Optional[NDArray[np.bool_]] = None
    ) -> CycloneForces:
        """Compute all force terms for a cyclone.
        
        Parameters
        ----------
        state : CycloneState
            Current cyclone state.
        u_wind, v_wind : xr.DataArray
            Environmental wind fields.
        land_mask : ndarray, optional
            Land/ocean mask.
            
        Returns
        -------
        CycloneForces
            All computed force terms.
        """
        # Environmental steering
        u_steer, v_steer = compute_environmental_steering(
            u_wind, v_wind, state.center,
            averaging_radius_km=self.steering_radius_km,
            layers=self.steering_layers
        )
        
        # Beta drift
        rmax = state.radius_max_wind / 1000 if state.radius_max_wind else 50.0
        u_beta, v_beta = compute_beta_drift(
            state.center,
            state.max_wind,
            radius_max_wind_km=rmax
        )
        
        # Land friction (placeholder until coast distance is computed)
        u_fric, v_fric = compute_land_friction(
            state.center,
            land_mask=land_mask
        )
        
        return CycloneForces(
            steering_u=u_steer,
            steering_v=v_steer,
            beta_drift_u=u_beta,
            beta_drift_v=v_beta,
            land_friction_u=u_fric,
            land_friction_v=v_fric
        )
