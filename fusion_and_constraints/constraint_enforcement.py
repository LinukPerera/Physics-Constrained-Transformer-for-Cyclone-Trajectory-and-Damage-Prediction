"""
Constraint Enforcement for Cyclone Predictions.

This module provides mechanisms for enforcing physical constraints on
model predictions, both as hard constraints (projection) and soft
constraints (penalty).

Constraint Types
----------------
1. HARD: Violated predictions are projected back to feasible space
2. SOFT: Violations are penalized but allowed

The choice depends on:
- Certainty in the constraint
- Importance of the constraint
- Whether projection is well-defined
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn

from common.logging_config import get_logger, AuditLogger
from common.types import GeoCoordinate

logger = get_logger(__name__)


@dataclass
class ConstraintViolation:
    """Record of a constraint violation.
    
    Attributes
    ----------
    constraint_name : str
        Name of the violated constraint.
    violation_magnitude : float
        Magnitude of the violation.
    original_value : float
        Value before enforcement.
    corrected_value : float
        Value after enforcement.
    timestamp : datetime
        When the violation occurred.
    """
    constraint_name: str
    violation_magnitude: float
    original_value: float
    corrected_value: float
    timestamp: datetime


class HardConstraintEnforcer(nn.Module):
    """Enforces hard physical constraints by projection.
    
    Hard constraints are enforced by projecting predictions to the
    nearest feasible point. This guarantees constraint satisfaction
    but may introduce discontinuities.
    
    Constraints
    -----------
    1. Position bounds (valid lat/lon)
    2. Intensity bounds (0 to max physical)
    3. Pressure bounds (min to ambient)
    4. Motion smoothness (max acceleration)
    """
    
    def __init__(
        self,
        max_intensity_ms: float = 95.0,
        min_pressure_hPa: float = 870.0,
        max_pressure_hPa: float = 1020.0,
        max_translation_ms: float = 35.0
    ):
        super().__init__()
        
        self.max_intensity_ms = max_intensity_ms
        self.min_pressure_hPa = min_pressure_hPa
        self.max_pressure_hPa = max_pressure_hPa
        self.max_translation_ms = max_translation_ms
        
        self._logger = get_logger("HardConstraintEnforcer")
        self._violations: List[ConstraintViolation] = []
    
    def forward(
        self,
        position: torch.Tensor,
        intensity: torch.Tensor,
        pressure: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enforce hard constraints on predictions.
        
        Parameters
        ----------
        position : torch.Tensor
            Predicted positions (B, T, 2) in degrees [lat, lon].
        intensity : torch.Tensor
            Predicted intensity (B, T) in m/s.
        pressure : torch.Tensor
            Predicted pressure (B, T) in hPa.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (position, intensity, pressure) after constraint enforcement.
        """
        # 1. Enforce position bounds
        lat = position[:, :, 0]
        lon = position[:, :, 1]
        
        lat_orig = lat.clone()
        lon_orig = lon.clone()
        
        lat = torch.clamp(lat, -60, 60)  # Tropical cyclone domain
        lon = torch.clamp(lon, -180, 180)
        
        # Handle longitude wraparound
        lon = torch.where(lon > 180, lon - 360, lon)
        lon = torch.where(lon < -180, lon + 360, lon)
        
        position_out = torch.stack([lat, lon], dim=-1)
        
        # Log position violations
        pos_violations = (lat != lat_orig) | (lon != lon_orig)
        if pos_violations.any():
            self._log_violation('position_bounds', pos_violations.sum().item())
        
        # 2. Enforce intensity bounds
        intensity_orig = intensity.clone()
        intensity_out = torch.clamp(intensity, 0, self.max_intensity_ms)
        
        int_violations = intensity != intensity_out
        if int_violations.any():
            self._log_violation('intensity_bounds', int_violations.sum().item())
        
        # 3. Enforce pressure bounds
        pressure_orig = pressure.clone()
        pressure_out = torch.clamp(
            pressure, 
            self.min_pressure_hPa, 
            self.max_pressure_hPa
        )
        
        p_violations = pressure != pressure_out
        if p_violations.any():
            self._log_violation('pressure_bounds', p_violations.sum().item())
        
        # 4. Enforce motion smoothness (implicit via position changes)
        if position.shape[1] > 1:
            position_out = self._enforce_smooth_motion(position_out)
        
        return position_out, intensity_out, pressure_out
    
    def _enforce_smooth_motion(
        self,
        position: torch.Tensor
    ) -> torch.Tensor:
        """Enforce maximum translation speed constraint.
        
        Uses iterative projection to ensure all sequential position
        differences correspond to speeds below the maximum.
        """
        max_delta_deg = self.max_translation_ms * 6 * 3600 / 111000  # 6h in deg
        
        position_out = position.clone()
        
        for t in range(1, position.shape[1]):
            delta = position_out[:, t] - position_out[:, t-1]
            delta_mag = torch.sqrt((delta ** 2).sum(dim=-1, keepdim=True))
            
            # Clip excessive motion
            scale = torch.where(
                delta_mag > max_delta_deg,
                max_delta_deg / (delta_mag + 1e-8),
                torch.ones_like(delta_mag)
            )
            
            position_out[:, t] = position_out[:, t-1] + delta * scale
        
        return position_out
    
    def _log_violation(self, constraint_name: str, count: int):
        """Log a constraint violation."""
        self._logger.warning(
            f"Hard constraint violation: {constraint_name}, count={count}"
        )
    
    def get_violations(self) -> List[ConstraintViolation]:
        """Get list of recorded violations."""
        return self._violations


class SoftConstraintEnforcer(nn.Module):
    """Computes penalties for soft constraint violations.
    
    Soft constraints are not strictly enforced but contribute to the
    loss function. This allows violations in extreme cases while
    discouraging them during training.
    """
    
    def __init__(
        self,
        pressure_wind_consistency_weight: float = 1.0,
        intensity_change_weight: float = 1.0,
        trajectory_smoothness_weight: float = 0.5
    ):
        super().__init__()
        
        self.pressure_wind_weight = pressure_wind_consistency_weight
        self.intensity_change_weight = intensity_change_weight
        self.smoothness_weight = trajectory_smoothness_weight
        
        self._logger = get_logger("SoftConstraintEnforcer")
    
    def forward(
        self,
        position: torch.Tensor,
        intensity: torch.Tensor,
        pressure: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute soft constraint penalties.
        
        Parameters
        ----------
        position : torch.Tensor
            Predicted positions (B, T, 2).
        intensity : torch.Tensor
            Predicted intensity (B, T).
        pressure : torch.Tensor
            Predicted pressure (B, T).
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            (total_penalty, penalty_breakdown)
        """
        penalties = {}
        
        # 1. Pressure-wind consistency
        # Atkinson & Holliday (1977): V = 3.92 * (1010 - P)^0.644
        expected_intensity = 3.92 * torch.relu(1010 - pressure) ** 0.644
        pw_error = torch.abs(intensity - expected_intensity)
        pw_penalty = self.pressure_wind_weight * torch.relu(pw_error - 10).mean()
        penalties['pressure_wind'] = pw_penalty
        
        # 2. Intensity change rate
        if intensity.shape[1] > 1:
            d_int = intensity[:, 1:] - intensity[:, :-1]
            
            # Penalize rapid intensification beyond 35 m/s / 24h = 8.75 m/s / 6h
            rapid_int = torch.relu(d_int - 8.75)
            
            # Penalize rapid weakening beyond 50 m/s / 24h = 12.5 m/s / 6h  
            rapid_weak = torch.relu(-d_int - 12.5)
            
            int_change_penalty = self.intensity_change_weight * (
                rapid_int.mean() + rapid_weak.mean()
            )
            penalties['intensity_change'] = int_change_penalty
        
        # 3. Trajectory smoothness (penalize jerk/acceleration changes)
        if position.shape[1] > 2:
            velocity = position[:, 1:] - position[:, :-1]
            acceleration = velocity[:, 1:] - velocity[:, :-1]
            
            # Penalize large acceleration
            acc_mag = torch.sqrt((acceleration ** 2).sum(dim=-1))
            smoothness_penalty = self.smoothness_weight * acc_mag.mean()
            penalties['smoothness'] = smoothness_penalty
        
        total_penalty = sum(penalties.values())
        breakdown = {k: v.item() for k, v in penalties.items()}
        breakdown['total'] = total_penalty.item()
        
        return total_penalty, breakdown


class ConstraintViolationLogger:
    """Logger for constraint violations with audit trail.
    
    Tracks all constraint violations for post-hoc analysis and
    system improvement.
    """
    
    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        log_threshold: float = 0.1
    ):
        """Initialize violation logger.
        
        Parameters
        ----------
        audit_logger : AuditLogger, optional
            Audit logger for persistence.
        log_threshold : float
            Minimum violation magnitude to log.
        """
        self.audit_logger = audit_logger
        self.log_threshold = log_threshold
        self._violations: List[Dict[str, Any]] = []
        self._logger = get_logger("ConstraintViolationLogger")
    
    def log_violation(
        self,
        constraint_name: str,
        violation_magnitude: float,
        original_value: float,
        corrected_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a constraint violation.
        
        Parameters
        ----------
        constraint_name : str
            Name of the violated constraint.
        violation_magnitude : float
            Magnitude of the violation.
        original_value : float
            Value before correction.
        corrected_value : float, optional
            Value after correction (for hard constraints).
        metadata : dict, optional
            Additional context.
        """
        if violation_magnitude < self.log_threshold:
            return
        
        record = {
            'constraint': constraint_name,
            'magnitude': violation_magnitude,
            'original': original_value,
            'corrected': corrected_value,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
        }
        
        self._violations.append(record)
        
        self._logger.info(
            f"Constraint violation: {constraint_name}, "
            f"magnitude={violation_magnitude:.4f}"
        )
        
        if self.audit_logger:
            self.audit_logger.log_constraint_violation(
                constraint_name,
                violation_magnitude,
                {
                    'original': original_value,
                    'corrected': corrected_value,
                }
            )
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of all violations."""
        if not self._violations:
            return {'total_violations': 0}
        
        summary = {
            'total_violations': len(self._violations),
            'by_constraint': {},
        }
        
        for v in self._violations:
            name = v['constraint']
            if name not in summary['by_constraint']:
                summary['by_constraint'][name] = {
                    'count': 0,
                    'avg_magnitude': 0,
                    'max_magnitude': 0,
                }
            
            summary['by_constraint'][name]['count'] += 1
            summary['by_constraint'][name]['avg_magnitude'] += v['magnitude']
            summary['by_constraint'][name]['max_magnitude'] = max(
                summary['by_constraint'][name]['max_magnitude'],
                v['magnitude']
            )
        
        # Compute averages
        for name in summary['by_constraint']:
            count = summary['by_constraint'][name]['count']
            summary['by_constraint'][name]['avg_magnitude'] /= count
        
        return summary
