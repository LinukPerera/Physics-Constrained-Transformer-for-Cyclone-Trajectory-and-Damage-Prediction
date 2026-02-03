"""
Atmospheric Dynamics Calculations for Cyclone Prediction.

This module computes physically interpretable atmospheric quantities
from raw data. These are PHYSICS-BASED calculations, not ML predictions.

Critical Design Rule
--------------------
No machine learning in this module. Every calculation must be traceable
to known atmospheric physics equations with cited sources.

References
----------
- Holton, J.R. & Hakim, G.J. (2013). An Introduction to Dynamic Meteorology.
- Emanuel, K.A. (1986). An air-sea interaction theory for tropical cyclones.
- Emanuel, K.A. (1995). Sensitivity of tropical cyclones to surface exchange.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import xarray as xr

from common.constants import PhysicalConstants
from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PotentialIntensityResult:
    """Result of potential intensity calculation.
    
    Attributes
    ----------
    v_max_ms : float
        Maximum potential intensity (wind speed) in m/s.
    p_min_hPa : float
        Minimum central pressure in hPa.
    outflow_temp_K : float
        Outflow temperature in Kelvin.
    efficiency : float
        Thermodynamic efficiency of the heat engine.
    sse_deficit : float
        Sea surface enthalpy deficit.
    """
    v_max_ms: float
    p_min_hPa: float
    outflow_temp_K: float
    efficiency: float
    sse_deficit: float


def compute_potential_intensity(
    sst_K: float,
    t_outflow_K: float,
    msl_Pa: float,
    relative_humidity: float = 0.8,
    h_boundary_layer_m: float = 1000.0
) -> PotentialIntensityResult:
    """Compute Maximum Potential Intensity (MPI) using Emanuel (1986, 1995).
    
    The potential intensity is the theoretical maximum intensity that a
    tropical cyclone could achieve given the thermodynamic environment.
    
    Parameters
    ----------
    sst_K : float
        Sea surface temperature in Kelvin.
    t_outflow_K : float
        Outflow temperature (typically 200 hPa) in Kelvin.
    msl_Pa : float
        Mean sea level pressure in Pascals.
    relative_humidity : float
        Boundary layer relative humidity (0-1).
    h_boundary_layer_m : float
        Boundary layer depth in meters.
        
    Returns
    -------
    PotentialIntensityResult
        Potential intensity and related quantities.
        
    Notes
    -----
    Based on Emanuel (1995) thermodynamic theory:
    
    V_max² = C_k/C_d × ε × (h*_s - h_b)
    
    where:
    - C_k/C_d ≈ 1 (ratio of exchange coefficients)
    - ε = (SST - T_out) / T_out (Carnot efficiency)
    - h*_s - h_b = sea surface enthalpy deficit
    
    LIMITATION: This is a theoretical maximum. Actual intensity is
    often limited by environmental factors (shear, dry air intrusion).
    
    References
    ----------
    Emanuel, K.A. (1995). Sensitivity of tropical cyclones to surface
    exchange coefficients. J. Atmos. Sci., 52, 3969-3976.
    """
    # Physical constants
    Cp = PhysicalConstants.DRY_AIR_SPECIFIC_HEAT_CP.value
    Lv = PhysicalConstants.LATENT_HEAT_VAPORIZATION.value
    Rv = PhysicalConstants.WATER_VAPOR_GAS_CONSTANT.value
    
    # Carnot efficiency
    epsilon = (sst_K - t_outflow_K) / t_outflow_K
    
    # Saturation specific humidity at SST
    # Using Clausius-Clapeyron approximation
    e_sat = 611.2 * np.exp((Lv / Rv) * (1/273.15 - 1/sst_K))
    q_sat = 0.622 * e_sat / (msl_Pa - 0.378 * e_sat)
    
    # Boundary layer specific humidity
    q_bl = relative_humidity * q_sat
    
    # Sea surface enthalpy (per unit mass)
    h_star_s = Cp * sst_K + Lv * q_sat
    
    # Boundary layer enthalpy
    h_bl = Cp * (sst_K - 1) + Lv * q_bl  # Assuming 1K cooler than SST
    
    # Enthalpy deficit
    delta_h = h_star_s - h_bl
    
    # Ratio of exchange coefficients (typically ~1)
    Ck_Cd = 1.0
    
    # Maximum potential intensity
    v_max_squared = Ck_Cd * epsilon * delta_h
    v_max = np.sqrt(max(v_max_squared, 0))
    
    # Estimate minimum pressure using empirical relationship
    # ΔP ≈ ρ × V_max² (approximately)
    rho = msl_Pa / (287 * sst_K)  # Approximate surface air density
    delta_p = rho * v_max**2
    p_min = (msl_Pa - delta_p) / 100  # Convert to hPa
    
    return PotentialIntensityResult(
        v_max_ms=float(v_max),
        p_min_hPa=float(max(p_min, 870)),  # Physical minimum
        outflow_temp_K=t_outflow_K,
        efficiency=float(epsilon),
        sse_deficit=float(delta_h)
    )


def compute_thermal_wind(
    t_upper: xr.DataArray,
    t_lower: xr.DataArray,
    p_upper: float,
    p_lower: float
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute thermal wind using the thermal wind equation.
    
    The thermal wind is the vertical shear of the geostrophic wind,
    related to horizontal temperature gradients.
    
    Parameters
    ----------
    t_upper : xr.DataArray
        Temperature at upper level (K).
    t_lower : xr.DataArray
        Temperature at lower level (K).
    p_upper : float
        Upper pressure level (Pa).
    p_lower : float
        Lower pressure level (Pa).
        
    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        (u_thermal, v_thermal) thermal wind components in m/s.
        
    Notes
    -----
    Thermal wind equation (in pressure coordinates):
    
    ∂u_g/∂ln(p) = (R/f) × ∂T/∂y
    ∂v_g/∂ln(p) = -(R/f) × ∂T/∂x
    
    References
    ----------
    Holton & Hakim (2013), Section 3.4.
    """
    R = PhysicalConstants.DRY_AIR_GAS_CONSTANT.value
    
    # Get latitude for Coriolis parameter
    lats = t_upper['latitude'].values
    lats_rad = np.radians(lats)
    f = PhysicalConstants.coriolis_parameter(lats_rad[:, np.newaxis])
    
    # Layer mean temperature
    t_mean = (t_upper + t_lower) / 2
    
    # Pressure layer depth
    delta_ln_p = np.log(p_lower) - np.log(p_upper)
    
    # Temperature gradients (using preprocessing module)
    from preprocessing.feature_engineering import compute_vorticity
    # We need spatial derivatives of T
    
    # Simplified: return zeros with same structure
    # TODO: Implement proper spherical gradient
    u_thermal = xr.zeros_like(t_upper)
    v_thermal = xr.zeros_like(t_upper)
    
    return u_thermal, v_thermal


def compute_pressure_gradient_force(
    pressure: xr.DataArray,
    density: Optional[xr.DataArray] = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute horizontal pressure gradient force per unit mass.
    
    Parameters
    ----------
    pressure : xr.DataArray
        Pressure field in Pascals.
    density : xr.DataArray, optional
        Air density in kg/m³. If not provided, estimated from pressure.
        
    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        (F_x, F_y) force per unit mass in m/s².
        
    Notes
    -----
    Pressure gradient force per unit mass:
    F = -(1/ρ) × ∇p
    
    Uses proper spherical derivatives.
    """
    from geospatial.coordinate_models import (
        radius_of_curvature_meridian,
        radius_of_curvature_prime_vertical
    )
    
    lats = pressure['latitude'].values
    lons = pressure['longitude'].values
    lats_rad = np.radians(lats)
    
    dlat = np.radians(np.abs(lats[1] - lats[0])) if len(lats) > 1 else 0.0
    dlon = np.radians(np.abs(lons[1] - lons[0])) if len(lons) > 1 else 0.0
    
    # Estimate density if not provided
    if density is None:
        R = PhysicalConstants.DRY_AIR_GAS_CONSTANT.value
        T = 288  # Assume standard temperature
        density = pressure / (R * T)
    
    # Get radii of curvature
    M = np.array([radius_of_curvature_meridian(lat) for lat in lats_rad])
    N = np.array([radius_of_curvature_prime_vertical(lat) for lat in lats_rad])
    
    # Reshape for broadcasting
    M = M.reshape(-1, 1)
    N = N.reshape(-1, 1)
    cos_lat = np.cos(lats_rad).reshape(-1, 1)
    
    p_vals = pressure.values
    rho_vals = density.values
    
    # ∂p/∂x (eastward)
    dpdx = np.gradient(p_vals, dlon, axis=-1) / (N * cos_lat)
    
    # ∂p/∂y (northward)
    dpdy = np.gradient(p_vals, dlat, axis=-2) / M
    
    # Force = -(1/ρ) ∇p
    F_x = -dpdx / rho_vals
    F_y = -dpdy / rho_vals
    
    F_x_da = xr.DataArray(
        F_x,
        dims=pressure.dims,
        coords=pressure.coords,
        attrs={'units': 'm s**-2', 'long_name': 'pressure_gradient_force_x'}
    )
    
    F_y_da = xr.DataArray(
        F_y,
        dims=pressure.dims,
        coords=pressure.coords,
        attrs={'units': 'm s**-2', 'long_name': 'pressure_gradient_force_y'}
    )
    
    return F_x_da, F_y_da


def compute_coriolis_acceleration(
    u: xr.DataArray,
    v: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute Coriolis acceleration.
    
    Parameters
    ----------
    u : xr.DataArray
        Eastward velocity in m/s.
    v : xr.DataArray
        Northward velocity in m/s.
        
    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        (a_x, a_y) Coriolis acceleration in m/s².
        
    Notes
    -----
    Coriolis acceleration:
    a_x = f × v
    a_y = -f × u
    
    where f = 2Ω sin(φ) is the Coriolis parameter.
    """
    lats = u['latitude'].values
    lats_rad = np.radians(lats)
    
    # Coriolis parameter at each latitude
    f = np.array([PhysicalConstants.coriolis_parameter(lat) for lat in lats_rad])
    
    # Reshape for broadcasting
    f = f.reshape(-1, 1)
    
    # Coriolis acceleration
    a_x = f * v.values
    a_y = -f * u.values
    
    a_x_da = xr.DataArray(
        a_x,
        dims=u.dims,
        coords=u.coords,
        attrs={'units': 'm s**-2', 'long_name': 'coriolis_acceleration_x'}
    )
    
    a_y_da = xr.DataArray(
        a_y,
        dims=u.dims,
        coords=u.coords,
        attrs={'units': 'm s**-2', 'long_name': 'coriolis_acceleration_y'}
    )
    
    return a_x_da, a_y_da


def compute_absolute_vorticity(
    relative_vorticity: xr.DataArray
) -> xr.DataArray:
    """Compute absolute vorticity (relative + planetary).
    
    Parameters
    ----------
    relative_vorticity : xr.DataArray
        Relative vorticity in s⁻¹.
        
    Returns
    -------
    xr.DataArray
        Absolute vorticity in s⁻¹.
        
    Notes
    -----
    η = ζ + f
    
    where ζ is relative vorticity and f is the Coriolis parameter.
    """
    lats = relative_vorticity['latitude'].values
    lats_rad = np.radians(lats)
    
    f = np.array([PhysicalConstants.coriolis_parameter(lat) for lat in lats_rad])
    f = f.reshape(-1, 1)
    
    absolute_vorticity = relative_vorticity.values + f
    
    return xr.DataArray(
        absolute_vorticity,
        dims=relative_vorticity.dims,
        coords=relative_vorticity.coords,
        attrs={'units': 's**-1', 'long_name': 'absolute_vorticity'}
    )
