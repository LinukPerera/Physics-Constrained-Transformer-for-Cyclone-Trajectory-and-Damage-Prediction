"""
Physics-Informed Loss Functions for Cyclone Prediction.

This module provides loss functions that incorporate physical constraints
into the training objective. These losses ensure that predictions respect
known physical laws and constraints.

Loss Components
---------------
1. Data loss: Standard prediction error
2. Physics loss: Violation of physical constraints
3. Conservation loss: Violation of conservation laws
4. Consistency loss: Internal consistency of predictions

References
----------
- Raissi, M. et al. (2019). Physics-informed neural networks.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PhysicsConstraints:
    """Physical constraints for cyclone predictions.
    
    Attributes
    ----------
    max_translation_speed_ms : float
        Maximum realistic cyclone translation speed.
    max_intensity_ms : float
        Maximum realistic wind speed (physical limit).
    min_pressure_hPa : float
        Minimum realistic central pressure.
    max_intensification_rate : float
        Maximum realistic intensification rate in m/s per 6h.
    max_weakening_rate : float
        Maximum realistic weakening rate in m/s per 6h.
    pressure_wind_slope : float
        Approximate pressure-wind relationship slope.
    """
    max_translation_speed_ms: float = 35.0  # ~70 knots
    max_intensity_ms: float = 95.0  # ~185 knots (theoretical max)
    min_pressure_hPa: float = 870.0  # Lowest ever recorded
    max_intensification_rate: float = 35.0  # Rapid intensification
    max_weakening_rate: float = 50.0  # Over land decay
    pressure_wind_slope: float = 1.1  # hPa per m/s approximately


class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty estimation."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted quantiles (B, T, num_quantiles).
        targets : torch.Tensor
            True values (B, T).
            
        Returns
        -------
        torch.Tensor
            Quantile loss value.
        """
        targets_expanded = targets.unsqueeze(-1)
        quantiles = self.quantiles.to(predictions.device)
        
        errors = targets_expanded - predictions
        
        loss = torch.max(
            quantiles * errors,
            (quantiles - 1) * errors
        )
        
        return loss.mean()


class ConstraintViolationPenalty(nn.Module):
    """Penalty for physical constraint violations.
    
    This loss penalizes predictions that violate known physical
    constraints. It can be used as either a soft penalty (added to
    loss) or a hard constraint (projection).
    """
    
    def __init__(
        self,
        constraints: PhysicsConstraints,
        violation_weight: float = 1.0
    ):
        super().__init__()
        self.constraints = constraints
        self.violation_weight = violation_weight
        self._logger = get_logger("ConstraintViolationPenalty")
    
    def forward(
        self,
        position: torch.Tensor,
        intensity: torch.Tensor,
        pressure: torch.Tensor,
        dt_hours: float = 6.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        position : torch.Tensor
            Predicted positions (B, T, 2) in degrees.
        intensity : torch.Tensor
            Predicted intensity (B, T) in m/s.
        pressure : torch.Tensor
            Predicted pressure (B, T) in hPa.
        dt_hours : float
            Time step in hours.
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (total_penalty, violation_counts)
        """
        penalties = []
        violations = {}
        
        # 1. Translation speed constraint
        if position.shape[1] > 1:
            # Approximate distance from position differences
            # Using degrees as proxy (rough, should use geodesic)
            d_pos = position[:, 1:, :] - position[:, :-1, :]
            d_deg = torch.sqrt((d_pos ** 2).sum(dim=-1))
            
            # Convert degrees to m/s (very rough: 111 km per degree)
            d_km = d_deg * 111
            speed_ms = d_km * 1000 / (dt_hours * 3600)
            
            max_speed = self.constraints.max_translation_speed_ms
            speed_violation = F.relu(speed_ms - max_speed)
            penalties.append(speed_violation.mean())
            violations['translation_speed'] = (speed_ms > max_speed).float().mean().item()
        
        # 2. Intensity bounds
        max_int = self.constraints.max_intensity_ms
        intensity_violation = F.relu(intensity - max_int) + F.relu(-intensity)
        penalties.append(intensity_violation.mean())
        violations['intensity_bounds'] = (
            (intensity > max_int) | (intensity < 0)
        ).float().mean().item()
        
        # 3. Pressure bounds
        min_p = self.constraints.min_pressure_hPa
        pressure_violation = F.relu(min_p - pressure) + F.relu(pressure - 1020)
        penalties.append(pressure_violation.mean())
        violations['pressure_bounds'] = (
            (pressure < min_p) | (pressure > 1020)
        ).float().mean().item()
        
        # 4. Intensity change rate
        if intensity.shape[1] > 1:
            d_intensity = intensity[:, 1:] - intensity[:, :-1]
            
            max_intensify = self.constraints.max_intensification_rate
            max_weaken = self.constraints.max_weakening_rate
            
            intensification_violation = F.relu(d_intensity - max_intensify)
            weakening_violation = F.relu(-d_intensity - max_weaken)
            
            penalties.append(intensification_violation.mean())
            penalties.append(weakening_violation.mean())
            violations['intensity_change'] = (
                (d_intensity > max_intensify) | (d_intensity < -max_weaken)
            ).float().mean().item()
        
        # 5. Pressure-wind consistency
        # Higher winds should correlate with lower pressure
        # V ~= slope * (1013 - P)
        expected_pressure = 1013 - intensity / self.constraints.pressure_wind_slope
        pressure_wind_error = torch.abs(pressure - expected_pressure)
        # Allow some slack (±20 hPa is common scatter)
        pressure_wind_violation = F.relu(pressure_wind_error - 20)
        penalties.append(pressure_wind_violation.mean())
        violations['pressure_wind_consistency'] = (
            pressure_wind_error > 20
        ).float().mean().item()
        
        total_penalty = self.violation_weight * sum(penalties)
        
        return total_penalty, violations


class ConservationLoss(nn.Module):
    """Losses based on conservation laws.
    
    While tropical cyclones are not closed systems, certain approximate
    conservation properties can be used as soft constraints.
    """
    
    def __init__(
        self,
        angular_momentum_weight: float = 0.1,
        energy_weight: float = 0.1
    ):
        super().__init__()
        self.angular_momentum_weight = angular_momentum_weight
        self.energy_weight = energy_weight
    
    def forward(
        self,
        intensity: torch.Tensor,
        radius_max_wind: torch.Tensor,
        dt_hours: float = 6.0
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        intensity : torch.Tensor
            Maximum wind speed (B, T) in m/s.
        radius_max_wind : torch.Tensor
            Radius of maximum winds (B, T) in km.
        dt_hours : float
            Time step in hours.
            
        Returns
        -------
        torch.Tensor
            Conservation penalty.
        """
        penalties = []
        
        if intensity.shape[1] > 1:
            # Approximate angular momentum: M ~ r × v
            M = radius_max_wind * intensity
            
            # Change in angular momentum should be bounded
            dM = M[:, 1:] - M[:, :-1]
            
            # Penalize large sudden changes
            M_violation = F.relu(torch.abs(dM) - 0.3 * M[:, :-1])
            penalties.append(self.angular_momentum_weight * M_violation.mean())
        
        return sum(penalties) if penalties else torch.tensor(0.0)


class PhysicsInformedLoss(nn.Module):
    """Combined physics-informed loss function.
    
    This loss combines:
    1. Data fidelity (quantile loss for predictions)
    2. Physics constraints (violation penalties)
    3. Conservation properties (soft constraints)
    
    Attributes
    ----------
    quantile_loss : QuantileLoss
        Loss for quantile predictions.
    constraint_penalty : ConstraintViolationPenalty
        Penalty for constraint violations.
    conservation_loss : ConservationLoss
        Penalty for conservation violations.
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        constraint_weight: float = 1.0,
        conservation_weight: float = 0.1,
        constraints: Optional[PhysicsConstraints] = None
    ):
        super().__init__()
        
        self.quantile_loss = QuantileLoss(quantiles)
        self.constraint_penalty = ConstraintViolationPenalty(
            constraints or PhysicsConstraints(),
            violation_weight=constraint_weight
        )
        self.conservation_loss = ConservationLoss(
            angular_momentum_weight=conservation_weight
        )
        
        self._logger = get_logger("PhysicsInformedLoss")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        radius_max_wind: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        predictions : dict
            Model predictions with keys: 'position', 'intensity', 'pressure'.
        targets : dict
            Ground truth with same keys.
        radius_max_wind : torch.Tensor, optional
            Radius of max winds for conservation loss.
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (total_loss, loss_breakdown)
        """
        losses = {}
        
        # Data fidelity losses
        if 'position' in predictions and 'position' in targets:
            # Position loss for latitude and longitude
            pos_pred = predictions['position'][:, :, :, 1]  # Median quantile
            pos_loss = F.mse_loss(pos_pred, targets['position'])
            losses['position'] = pos_loss
        
        if 'intensity' in predictions and 'intensity' in targets:
            int_pred = predictions['intensity']
            int_loss = self.quantile_loss(int_pred, targets['intensity'])
            losses['intensity'] = int_loss
        
        if 'pressure' in predictions and 'pressure' in targets:
            p_pred = predictions['pressure']
            p_loss = self.quantile_loss(p_pred, targets['pressure'])
            losses['pressure'] = p_loss
        
        # Physics constraint penalty
        pos_median = predictions['position'][:, :, :, 1]
        int_median = predictions['intensity'][:, :, 1]
        p_median = predictions['pressure'][:, :, 1]
        
        constraint_penalty, violations = self.constraint_penalty(
            pos_median, int_median, p_median
        )
        losses['physics_constraints'] = constraint_penalty
        
        # Conservation loss
        if radius_max_wind is not None:
            cons_loss = self.conservation_loss(int_median, radius_max_wind)
            losses['conservation'] = cons_loss
        
        # Total loss
        total_loss = sum(losses.values())
        
        # Create breakdown for logging
        breakdown = {k: v.item() for k, v in losses.items()}
        breakdown.update({f'violation_{k}': v for k, v in violations.items()})
        breakdown['total'] = total_loss.item()
        
        return total_loss, breakdown
