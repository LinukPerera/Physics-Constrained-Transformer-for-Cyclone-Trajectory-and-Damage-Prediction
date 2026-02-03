"""
Prediction Heads for Cyclone Trajectory and Intensity.

This module provides the output heads that convert fused features
into specific predictions (position, intensity, pressure, structure).

Each head is designed with:
1. Appropriate output constraints
2. Uncertainty quantification
3. Physical reasonability checks
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger

logger = get_logger(__name__)


class PositionHead(nn.Module):
    """Output head for position (latitude, longitude) prediction.
    
    Predicts displacement from current position rather than absolute
    position, which is more physically meaningful and easier to learn.
    
    Output
    ------
    - Position as (lat, lon) displacement in degrees
    - Can output quantiles for uncertainty
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_quantiles: int = 3,
        max_displacement_deg: float = 10.0
    ):
        """Initialize position head.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dim : int
            Hidden layer dimension.
        num_quantiles : int
            Number of quantiles to predict.
        max_displacement_deg : float
            Maximum displacement per time step in degrees.
        """
        super().__init__()
        
        self.num_quantiles = num_quantiles
        self.max_displacement = max_displacement_deg
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * num_quantiles),  # (lat, lon) × quantiles
        )
    
    def forward(
        self,
        features: torch.Tensor,
        current_position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features (B, T, input_dim).
        current_position : torch.Tensor, optional
            Current position (B, 2) in degrees for displacement calculation.
            
        Returns
        -------
        torch.Tensor
            Position predictions (B, T, 2, num_quantiles).
        """
        B, T, _ = features.shape
        
        # Predict displacements
        output = self.mlp(features)  # (B, T, 2 * num_quantiles)
        output = output.view(B, T, 2, self.num_quantiles)
        
        # Apply tanh to bound displacements
        displacement = torch.tanh(output) * self.max_displacement
        
        # If current position provided, convert to absolute
        if current_position is not None:
            # Cumulative sum for trajectory
            position = current_position.unsqueeze(1).unsqueeze(-1) + displacement.cumsum(dim=1)
        else:
            position = displacement
        
        return position


class IntensityHead(nn.Module):
    """Output head for intensity (maximum sustained wind) prediction.
    
    Intensity is bounded to be positive and below physical maximum.
    Uses log-space prediction for better numerical properties.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_quantiles: int = 3,
        max_intensity_ms: float = 95.0
    ):
        super().__init__()
        
        self.num_quantiles = num_quantiles
        self.max_intensity = max_intensity_ms
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_quantiles),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features (B, T, input_dim).
            
        Returns
        -------
        torch.Tensor
            Intensity predictions (B, T, num_quantiles) in m/s.
        """
        output = self.mlp(features)
        
        # Softplus for positivity, then scale and clip
        intensity = F.softplus(output) * 20  # Scale factor for reasonable range
        intensity = torch.clamp(intensity, 0, self.max_intensity)
        
        # Ensure quantile ordering (q10 < q50 < q90)
        intensity = self._enforce_quantile_order(intensity)
        
        return intensity
    
    def _enforce_quantile_order(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce monotonic ordering of quantiles."""
        if self.num_quantiles <= 1:
            return x
        
        # Sort along quantile dimension
        return x.sort(dim=-1)[0]


class PressureHead(nn.Module):
    """Output head for central pressure prediction.
    
    Pressure is bounded between physical minimum (~870 hPa) and
    ambient (~1013 hPa).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_quantiles: int = 3,
        min_pressure_hPa: float = 870.0,
        max_pressure_hPa: float = 1013.0
    ):
        super().__init__()
        
        self.num_quantiles = num_quantiles
        self.min_pressure = min_pressure_hPa
        self.max_pressure = max_pressure_hPa
        self.pressure_range = max_pressure_hPa - min_pressure_hPa
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_quantiles),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features (B, T, input_dim).
            
        Returns
        -------
        torch.Tensor
            Pressure predictions (B, T, num_quantiles) in hPa.
        """
        output = self.mlp(features)
        
        # Sigmoid to [0, 1], then scale to pressure range
        normalized = torch.sigmoid(output)
        pressure = self.min_pressure + normalized * self.pressure_range
        
        # For pressure, lower quantile = stronger storm = lower pressure
        # So we reverse the quantile ordering
        pressure = self._enforce_reversed_quantile_order(pressure)
        
        return pressure
    
    def _enforce_reversed_quantile_order(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce reversed ordering (q10 > q50 > q90 for pressure)."""
        if self.num_quantiles <= 1:
            return x
        
        # Sort descending
        return x.sort(dim=-1, descending=True)[0]


class StructureHead(nn.Module):
    """Output head for storm structure predictions.
    
    Predicts:
    - Radius of maximum winds (RMW)
    - Radii of 34, 50, 64 kt winds in quadrants
    
    These are important for hazard modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        predict_quadrants: bool = True
    ):
        """Initialize structure head.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dim : int
            Hidden dimension.
        predict_quadrants : bool
            Whether to predict per-quadrant radii.
        """
        super().__init__()
        
        self.predict_quadrants = predict_quadrants
        
        # RMW head
        self.rmw_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        if predict_quadrants:
            # Wind radii: 3 thresholds × 4 quadrants = 12 outputs
            self.wind_radii_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 12),
            )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features (B, T, input_dim).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with:
            - 'rmw': Radius of maximum winds (B, T) in km
            - 'wind_radii': Wind radii (B, T, 3, 4) in km if quadrants
        """
        # RMW prediction (bounded to 10-200 km)
        rmw = self.rmw_head(features).squeeze(-1)
        rmw = 10 + F.softplus(rmw) * 50  # Typical range
        rmw = torch.clamp(rmw, 10, 200)
        
        outputs = {'rmw': rmw}
        
        if self.predict_quadrants:
            # Wind radii prediction
            radii = self.wind_radii_head(features)
            radii = F.softplus(radii) * 100  # Scale to reasonable km
            radii = radii.view(features.shape[0], features.shape[1], 3, 4)
            outputs['wind_radii'] = radii
        
        return outputs


class CombinedPredictionHead(nn.Module):
    """Combined prediction head for all outputs.
    
    Combines position, intensity, pressure, and structure heads
    with optional cross-consistency enforcement.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_quantiles: int = 3,
        predict_structure: bool = True
    ):
        super().__init__()
        
        self.position_head = PositionHead(input_dim, hidden_dim, num_quantiles)
        self.intensity_head = IntensityHead(input_dim, hidden_dim, num_quantiles)
        self.pressure_head = PressureHead(input_dim, hidden_dim, num_quantiles)
        
        if predict_structure:
            self.structure_head = StructureHead(input_dim, hidden_dim)
        else:
            self.structure_head = None
    
    def forward(
        self,
        features: torch.Tensor,
        current_position: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        Dict[str, torch.Tensor]
            All predictions in a dictionary.
        """
        outputs = {
            'position': self.position_head(features, current_position),
            'intensity': self.intensity_head(features),
            'pressure': self.pressure_head(features),
        }
        
        if self.structure_head is not None:
            structure = self.structure_head(features)
            outputs.update(structure)
        
        return outputs
