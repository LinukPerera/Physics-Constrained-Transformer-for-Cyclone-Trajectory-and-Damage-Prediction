"""
Physics-Aware Normalization for Cyclone Prediction.

This module provides normalization operations that preserve dimensional
interpretability. Unlike standard ML normalization, these transforms
are designed to be reversible and to maintain physical meaning.

Design Principles
-----------------
1. All transformations are reversible
2. Normalized values have defined physical interpretation
3. Different treatment for intensive vs extensive quantities
4. Anomaly-based normalization for climate-dependent variables

References
----------
- Weyn, J.A. et al. (2020). Improving data-driven global weather prediction
  using deep convolutional neural networks on a cubed sphere. JAMES.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray
import xarray as xr
import json
from pathlib import Path

from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NormalizationParams:
    """Parameters for reversible normalization.
    
    Attributes
    ----------
    method : str
        Normalization method used.
    mean : float
        Mean value (for standardization).
    std : float
        Standard deviation (for standardization).
    min_val : float
        Minimum value (for min-max scaling).
    max_val : float
        Maximum value (for min-max scaling).
    climatology_mean : float, optional
        Climatological mean (for anomaly normalization).
    climatology_std : float, optional
        Climatological standard deviation.
    unit : str
        Physical unit of the original data.
    """
    method: str
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    climatology_mean: Optional[float] = None
    climatology_std: Optional[float] = None
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method,
            'mean': float(self.mean),
            'std': float(self.std),
            'min_val': float(self.min_val),
            'max_val': float(self.max_val),
            'climatology_mean': float(self.climatology_mean) if self.climatology_mean else None,
            'climatology_std': float(self.climatology_std) if self.climatology_std else None,
            'unit': self.unit,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NormalizationParams':
        """Create from dictionary."""
        return cls(**d)


class ReversibleTransform(ABC):
    """Abstract base class for reversible transforms."""
    
    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        """Apply forward transform."""
        pass
    
    @abstractmethod
    def inverse(self, x: NDArray) -> NDArray:
        """Apply inverse transform."""
        pass
    
    @abstractmethod
    def get_params(self) -> NormalizationParams:
        """Get parameters for persistence."""
        pass


class StandardScaler(ReversibleTransform):
    """Standard (z-score) normalization: (x - mean) / std."""
    
    def __init__(
        self,
        mean: float,
        std: float,
        unit: str = ""
    ):
        self.mean = mean
        self.std = std
        self.unit = unit
        
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
    
    def forward(self, x: NDArray) -> NDArray:
        return (x - self.mean) / self.std
    
    def inverse(self, x: NDArray) -> NDArray:
        return x * self.std + self.mean
    
    def get_params(self) -> NormalizationParams:
        return NormalizationParams(
            method='standard',
            mean=self.mean,
            std=self.std,
            unit=self.unit
        )
    
    @classmethod
    def fit(cls, data: NDArray, unit: str = "") -> 'StandardScaler':
        """Fit scaler to data."""
        return cls(
            mean=float(np.nanmean(data)),
            std=float(np.nanstd(data)),
            unit=unit
        )


class MinMaxScaler(ReversibleTransform):
    """Min-max normalization to [0, 1] range."""
    
    def __init__(
        self,
        min_val: float,
        max_val: float,
        unit: str = ""
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        
        if max_val <= min_val:
            raise ValueError("max_val must be greater than min_val")
    
    def forward(self, x: NDArray) -> NDArray:
        return (x - self.min_val) / (self.max_val - self.min_val)
    
    def inverse(self, x: NDArray) -> NDArray:
        return x * (self.max_val - self.min_val) + self.min_val
    
    def get_params(self) -> NormalizationParams:
        return NormalizationParams(
            method='minmax',
            min_val=self.min_val,
            max_val=self.max_val,
            unit=self.unit
        )
    
    @classmethod
    def fit(cls, data: NDArray, unit: str = "") -> 'MinMaxScaler':
        """Fit scaler to data."""
        return cls(
            min_val=float(np.nanmin(data)),
            max_val=float(np.nanmax(data)),
            unit=unit
        )


class AnomalyNormalizer(ReversibleTransform):
    """Anomaly-based normalization: (x - climatology) / climatology_std.
    
    This is preferred for quantities with strong climatological signals,
    such as sea surface temperature where the pattern of anomalies
    is more relevant than absolute values.
    """
    
    def __init__(
        self,
        climatology_mean: float,
        climatology_std: float,
        unit: str = ""
    ):
        self.climatology_mean = climatology_mean
        self.climatology_std = climatology_std
        self.unit = unit
        
        if climatology_std <= 0:
            raise ValueError("Climatological std must be positive")
    
    def forward(self, x: NDArray) -> NDArray:
        return (x - self.climatology_mean) / self.climatology_std
    
    def inverse(self, x: NDArray) -> NDArray:
        return x * self.climatology_std + self.climatology_mean
    
    def get_params(self) -> NormalizationParams:
        return NormalizationParams(
            method='anomaly',
            climatology_mean=self.climatology_mean,
            climatology_std=self.climatology_std,
            unit=self.unit
        )


class PhysicsAwareNormalizer:
    """Normalizer that respects physical properties of variables.
    
    This class applies appropriate normalization strategies based on
    the physical nature of each variable:
    
    1. Intensive quantities (T, P): Standard or anomaly normalization
    2. Extensive quantities (precipitation): Preserve zero, scale by max
    3. Vector quantities (wind): Normalize components together
    4. Bounded quantities (humidity): Min-max to [0, 1]
    
    Attributes
    ----------
    scalers : dict
        Mapping from variable names to scalers.
    """
    
    # Default normalization strategies by variable type
    VARIABLE_STRATEGIES = {
        # Temperature-like: anomaly normalization
        'temperature': 'anomaly',
        't': 'anomaly',
        't2m': 'anomaly',
        'sst': 'anomaly',
        
        # Pressure: standard normalization
        'pressure': 'standard',
        'msl': 'standard',
        
        # Wind: standard normalization (components normalized together)
        'u': 'standard',
        'v': 'standard',
        'u10': 'standard',
        'v10': 'standard',
        
        # Humidity: min-max to [0, 1]
        'q': 'minmax',
        'r': 'minmax',
        
        # Precipitation: positive, log-scale often useful
        'tp': 'positive',
        'precipitation': 'positive',
        
        # Geopotential: standard
        'z': 'standard',
    }
    
    def __init__(self):
        self.scalers: Dict[str, ReversibleTransform] = {}
        self._logger = get_logger("PhysicsAwareNormalizer")
    
    def fit(
        self,
        ds: xr.Dataset,
        climatology: Optional[xr.Dataset] = None
    ) -> 'PhysicsAwareNormalizer':
        """Fit normalizers to dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            Training data to fit normalizers.
        climatology : xr.Dataset, optional
            Climatological means and stds for anomaly normalization.
            
        Returns
        -------
        self
        """
        for var_name in ds.data_vars:
            data = ds[var_name].values
            unit = ds[var_name].attrs.get('units', '')
            
            strategy = self.VARIABLE_STRATEGIES.get(var_name, 'standard')
            
            if strategy == 'anomaly' and climatology is not None:
                if var_name in climatology:
                    self.scalers[var_name] = AnomalyNormalizer(
                        climatology_mean=float(climatology[var_name].mean()),
                        climatology_std=float(climatology[var_name].std()),
                        unit=unit
                    )
                else:
                    # Fall back to standard
                    self.scalers[var_name] = StandardScaler.fit(data, unit)
            elif strategy == 'minmax':
                self.scalers[var_name] = MinMaxScaler.fit(data, unit)
            else:  # standard or default
                self.scalers[var_name] = StandardScaler.fit(data, unit)
            
            self._logger.debug(
                f"Fitted {strategy} normalizer for {var_name}"
            )
        
        return self
    
    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply normalization to dataset.
        
        Parameters
        ----------
        ds : xr.Dataset
            Data to normalize.
            
        Returns
        -------
        xr.Dataset
            Normalized data.
        """
        output = ds.copy()
        
        for var_name in ds.data_vars:
            if var_name in self.scalers:
                output[var_name].values = self.scalers[var_name].forward(
                    ds[var_name].values
                )
                # Store normalization info in attributes
                params = self.scalers[var_name].get_params()
                output[var_name].attrs['normalization'] = params.method
                output[var_name].attrs['original_unit'] = params.unit
            else:
                self._logger.warning(
                    f"No scaler fitted for {var_name}, leaving unchanged"
                )
        
        return output
    
    def inverse_transform(self, ds: xr.Dataset) -> xr.Dataset:
        """Reverse normalization.
        
        Parameters
        ----------
        ds : xr.Dataset
            Normalized data.
            
        Returns
        -------
        xr.Dataset
            Data in original units.
        """
        output = ds.copy()
        
        for var_name in ds.data_vars:
            if var_name in self.scalers:
                output[var_name].values = self.scalers[var_name].inverse(
                    ds[var_name].values
                )
                # Restore original unit
                params = self.scalers[var_name].get_params()
                output[var_name].attrs['units'] = params.unit
        
        return output
    
    def save_params(self, path: Path) -> None:
        """Save normalization parameters to file.
        
        Parameters
        ----------
        path : Path
            Path to save JSON file.
        """
        params = {
            var_name: scaler.get_params().to_dict()
            for var_name, scaler in self.scalers.items()
        }
        
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        
        self._logger.info(f"Saved normalization params to {path}")
    
    def load_params(self, path: Path) -> 'PhysicsAwareNormalizer':
        """Load normalization parameters from file.
        
        Parameters
        ----------
        path : Path
            Path to JSON file.
            
        Returns
        -------
        self
        """
        with open(path, 'r') as f:
            params = json.load(f)
        
        for var_name, param_dict in params.items():
            np_params = NormalizationParams.from_dict(param_dict)
            
            if np_params.method == 'standard':
                self.scalers[var_name] = StandardScaler(
                    np_params.mean, np_params.std, np_params.unit
                )
            elif np_params.method == 'minmax':
                self.scalers[var_name] = MinMaxScaler(
                    np_params.min_val, np_params.max_val, np_params.unit
                )
            elif np_params.method == 'anomaly':
                self.scalers[var_name] = AnomalyNormalizer(
                    np_params.climatology_mean,
                    np_params.climatology_std,
                    np_params.unit
                )
        
        self._logger.info(f"Loaded normalization params from {path}")
        return self
