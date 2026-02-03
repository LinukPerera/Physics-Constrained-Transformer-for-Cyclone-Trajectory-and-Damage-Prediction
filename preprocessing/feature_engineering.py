"""
Feature Engineering for Cyclone Prediction.

This module computes derived physical quantities from raw observations.
Features are explicitly categorized to maintain clarity about their
physical meaning and ML-interpretability.

Feature Categories
------------------
1. OBSERVED: Direct measurements from data sources
2. DERIVED_PHYSICAL: Computed from physics (vorticity, divergence, etc.)
3. ML_ABSTRACTION: Features designed for ML that lack direct physical meaning

Design Principles
-----------------
- All derived features have documented physical meaning
- Features retain unit information
- Computation uses proper spherical derivatives (not Cartesian)

References
----------
- Holton, J.R. (2004). An Introduction to Dynamic Meteorology.
- Emanuel, K.A. (1986). An air-sea interaction theory for tropical cyclones.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import xarray as xr

from common.logging_config import get_logger
from common.constants import PhysicalConstants
from geospatial.coordinate_models import (
    radius_of_curvature_meridian,
    radius_of_curvature_prime_vertical
)

logger = get_logger(__name__)


class FeatureCategory(Enum):
    """Categories of features by their physical nature."""
    
    OBSERVED = auto()
    """Direct measurements from data sources."""
    
    DERIVED_PHYSICAL = auto()
    """Computed from physics-based transformations."""
    
    ML_ABSTRACTION = auto()
    """Features designed for ML without direct physical meaning."""


@dataclass
class FeatureMetadata:
    """Metadata for a feature.
    
    Attributes
    ----------
    name : str
        Feature name.
    category : FeatureCategory
        Physical category.
    unit : str
        Physical unit.
    description : str
        Human-readable description.
    sources : List[str]
        Source variables used to compute this feature.
    """
    name: str
    category: FeatureCategory
    unit: str
    description: str
    sources: List[str]


# Registry of derived features
DERIVED_FEATURES_REGISTRY: Dict[str, FeatureMetadata] = {
    'relative_vorticity': FeatureMetadata(
        name='relative_vorticity',
        category=FeatureCategory.DERIVED_PHYSICAL,
        unit='s⁻¹',
        description='Vertical component of relative vorticity (∂v/∂x - ∂u/∂y)',
        sources=['u', 'v']
    ),
    'divergence': FeatureMetadata(
        name='divergence',
        category=FeatureCategory.DERIVED_PHYSICAL,
        unit='s⁻¹',
        description='Horizontal divergence (∂u/∂x + ∂v/∂y)',
        sources=['u', 'v']
    ),
    'wind_shear_magnitude': FeatureMetadata(
        name='wind_shear_magnitude',
        category=FeatureCategory.DERIVED_PHYSICAL,
        unit='m/s',
        description='Magnitude of vertical wind shear between levels',
        sources=['u', 'v']
    ),
    'potential_intensity': FeatureMetadata(
        name='potential_intensity',
        category=FeatureCategory.DERIVED_PHYSICAL,
        unit='m/s',
        description='Maximum potential intensity (Emanuel 1986)',
        sources=['sst', 't', 'q', 'msl']
    ),
}


def compute_vorticity(
    u: xr.DataArray,
    v: xr.DataArray,
    lat_dim: str = 'latitude',
    lon_dim: str = 'longitude'
) -> xr.DataArray:
    """Compute relative vorticity on a sphere.
    
    Uses proper spherical derivatives, not Cartesian approximations.
    
    Parameters
    ----------
    u : xr.DataArray
        Eastward wind component in m/s.
    v : xr.DataArray
        Northward wind component in m/s.
    lat_dim, lon_dim : str
        Names of latitude and longitude dimensions.
        
    Returns
    -------
    xr.DataArray
        Relative vorticity in s⁻¹.
        
    Notes
    -----
    On a sphere:
    ζ = (1/R cos φ) * (∂v/∂λ) - (1/R) * (∂u/∂φ) + (u tan φ)/R
    
    where R is Earth's radius, φ is latitude, λ is longitude.
    
    This implementation uses the proper ellipsoidal radii of curvature.
    """
    # Get coordinates
    lats = u[lat_dim].values
    lons = u[lon_dim].values
    lats_rad = np.radians(lats)
    
    # Grid spacing
    dlat = np.radians(np.abs(lats[1] - lats[0])) if len(lats) > 1 else 0.0
    dlon = np.radians(np.abs(lons[1] - lons[0])) if len(lons) > 1 else 0.0
    
    # Compute radii of curvature at each latitude
    M = np.array([radius_of_curvature_meridian(lat) for lat in lats_rad])
    N = np.array([radius_of_curvature_prime_vertical(lat) for lat in lats_rad])
    
    # Reshape for broadcasting
    if u.ndim > 2:
        # Handle additional dimensions (time, level)
        M = M.reshape((1,) * (u.ndim - 2) + (-1, 1))
        N = N.reshape((1,) * (u.ndim - 2) + (-1, 1))
        cos_lat = np.cos(lats_rad).reshape((1,) * (u.ndim - 2) + (-1, 1))
        tan_lat = np.tan(lats_rad).reshape((1,) * (u.ndim - 2) + (-1, 1))
    else:
        M = M.reshape(-1, 1)
        N = N.reshape(-1, 1)
        cos_lat = np.cos(lats_rad).reshape(-1, 1)
        tan_lat = np.tan(lats_rad).reshape(-1, 1)
    
    u_vals = u.values
    v_vals = v.values
    
    # Compute derivatives using central differences
    # ∂v/∂λ (east-west derivative)
    dvdlon = np.gradient(v_vals, dlon, axis=-1)
    
    # ∂u/∂φ (north-south derivative)
    dudlat = np.gradient(u_vals, dlat, axis=-2)
    
    # Vorticity on sphere
    # ζ = (1/(N cos φ)) ∂v/∂λ - (1/M) ∂u/∂φ + (u tan φ)/N
    vorticity = (
        dvdlon / (N * cos_lat) 
        - dudlat / M 
        + u_vals * tan_lat / N
    )
    
    # Create output DataArray
    result = xr.DataArray(
        vorticity,
        dims=u.dims,
        coords=u.coords,
        attrs={
            'units': 's**-1',
            'long_name': 'relative_vorticity',
            'description': 'Vertical component of relative vorticity',
            'computation': 'spherical_derivatives',
        }
    )
    
    return result


def compute_divergence(
    u: xr.DataArray,
    v: xr.DataArray,
    lat_dim: str = 'latitude',
    lon_dim: str = 'longitude'
) -> xr.DataArray:
    """Compute horizontal divergence on a sphere.
    
    Parameters
    ----------
    u : xr.DataArray
        Eastward wind component in m/s.
    v : xr.DataArray
        Northward wind component in m/s.
    lat_dim, lon_dim : str
        Names of latitude and longitude dimensions.
        
    Returns
    -------
    xr.DataArray
        Divergence in s⁻¹.
        
    Notes
    -----
    On a sphere:
    D = (1/R cos φ) * (∂u/∂λ) + (1/R) * (∂(v cos φ)/∂φ)
    """
    lats = u[lat_dim].values
    lons = u[lon_dim].values
    lats_rad = np.radians(lats)
    
    dlat = np.radians(np.abs(lats[1] - lats[0])) if len(lats) > 1 else 0.0
    dlon = np.radians(np.abs(lons[1] - lons[0])) if len(lons) > 1 else 0.0
    
    M = np.array([radius_of_curvature_meridian(lat) for lat in lats_rad])
    N = np.array([radius_of_curvature_prime_vertical(lat) for lat in lats_rad])
    
    if u.ndim > 2:
        M = M.reshape((1,) * (u.ndim - 2) + (-1, 1))
        N = N.reshape((1,) * (u.ndim - 2) + (-1, 1))
        cos_lat = np.cos(lats_rad).reshape((1,) * (u.ndim - 2) + (-1, 1))
    else:
        M = M.reshape(-1, 1)
        N = N.reshape(-1, 1)
        cos_lat = np.cos(lats_rad).reshape(-1, 1)
    
    u_vals = u.values
    v_vals = v.values
    
    # ∂u/∂λ
    dudlon = np.gradient(u_vals, dlon, axis=-1)
    
    # ∂(v cos φ)/∂φ
    v_cos = v_vals * cos_lat
    d_vcos_dlat = np.gradient(v_cos, dlat, axis=-2)
    
    # Divergence
    divergence = dudlon / (N * cos_lat) + d_vcos_dlat / (M * cos_lat)
    
    result = xr.DataArray(
        divergence,
        dims=u.dims,
        coords=u.coords,
        attrs={
            'units': 's**-1',
            'long_name': 'divergence',
            'description': 'Horizontal divergence',
            'computation': 'spherical_derivatives',
        }
    )
    
    return result


def compute_wind_shear(
    u_upper: xr.DataArray,
    v_upper: xr.DataArray,
    u_lower: xr.DataArray,
    v_lower: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute vertical wind shear between two levels.
    
    Parameters
    ----------
    u_upper, v_upper : xr.DataArray
        Wind components at upper level (e.g., 200 hPa).
    u_lower, v_lower : xr.DataArray
        Wind components at lower level (e.g., 850 hPa).
        
    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        (shear_magnitude, shear_direction) in m/s and radians.
        
    Notes
    -----
    Deep-layer shear (200-850 hPa) is a key predictor of cyclone
    intensity change. High shear (>10 m/s) generally inhibits
    intensification.
    """
    du = u_upper - u_lower
    dv = v_upper - v_lower
    
    magnitude = np.sqrt(du**2 + dv**2)
    direction = np.arctan2(du, dv)  # Direction shear is pointing TO
    
    magnitude.attrs = {
        'units': 'm s**-1',
        'long_name': 'vertical_wind_shear_magnitude',
        'description': 'Magnitude of deep-layer vertical wind shear',
    }
    
    direction.attrs = {
        'units': 'radians',
        'long_name': 'vertical_wind_shear_direction',
        'description': 'Direction of wind shear vector (from lower to upper)',
    }
    
    return magnitude, direction


class DerivedFeatureCalculator:
    """Calculator for derived physical features.
    
    This class computes physics-based derived features from raw
    atmospheric data and tracks the category of each feature.
    """
    
    def __init__(self):
        self._logger = get_logger("DerivedFeatureCalculator")
        self._feature_registry: Dict[str, FeatureMetadata] = {}
    
    def compute_all(
        self,
        ds: xr.Dataset,
        features: Optional[List[str]] = None
    ) -> xr.Dataset:
        """Compute all requested derived features.
        
        Parameters
        ----------
        ds : xr.Dataset
            Input dataset with raw variables.
        features : list, optional
            List of feature names to compute. If None, compute all available.
            
        Returns
        -------
        xr.Dataset
            Dataset with additional derived features.
        """
        output = ds.copy()
        
        if features is None:
            features = list(DERIVED_FEATURES_REGISTRY.keys())
        
        for feature_name in features:
            if feature_name not in DERIVED_FEATURES_REGISTRY:
                self._logger.warning(f"Unknown feature: {feature_name}")
                continue
            
            metadata = DERIVED_FEATURES_REGISTRY[feature_name]
            
            # Check if sources are available
            sources_available = all(s in ds.data_vars for s in metadata.sources)
            if not sources_available:
                self._logger.warning(
                    f"Cannot compute {feature_name}: missing sources {metadata.sources}"
                )
                continue
            
            # Compute the feature
            try:
                if feature_name == 'relative_vorticity':
                    output[feature_name] = compute_vorticity(ds['u'], ds['v'])
                elif feature_name == 'divergence':
                    output[feature_name] = compute_divergence(ds['u'], ds['v'])
                else:
                    self._logger.warning(
                        f"Computation not implemented for {feature_name}"
                    )
                    continue
                
                self._feature_registry[feature_name] = metadata
                self._logger.info(f"Computed {feature_name}")
                
            except Exception as e:
                self._logger.error(f"Error computing {feature_name}: {e}")
        
        return output
    
    def get_feature_categories(self) -> Dict[str, FeatureCategory]:
        """Get categories for all computed features."""
        return {
            name: meta.category
            for name, meta in self._feature_registry.items()
        }
    
    def get_observed_features(self, ds: xr.Dataset) -> List[str]:
        """Identify which features in a dataset are observed (not derived)."""
        derived = set(DERIVED_FEATURES_REGISTRY.keys())
        return [var for var in ds.data_vars if var not in derived]
