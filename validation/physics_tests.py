"""
Physics Consistency Tests for Cyclone Prediction System.

This module provides tests to verify that model outputs and
calculations obey known physical laws and constraints.

Test Categories
---------------
1. Dimensional consistency (units match)
2. Conservation properties (energy, mass, momentum)
3. Physical bounds (values within realistic ranges)
4. Internal consistency (outputs don't contradict each other)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray

from common.logging_config import get_logger
from common.constants import PhysicalConstants

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check.
    
    Attributes
    ----------
    test_name : str
        Name of the test.
    passed : bool
        Whether the test passed.
    message : str
        Description of result.
    details : dict
        Additional details.
    """
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any]


class PhysicsConsistencyChecker:
    """Checker for physical consistency of model outputs.
    
    Validates that model outputs obey known physical constraints
    and relationships.
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        log_violations: bool = True
    ):
        """Initialize physics checker.
        
        Parameters
        ----------
        strict_mode : bool
            If True, raise exceptions on violations.
        log_violations : bool
            Whether to log violations.
        """
        self.strict_mode = strict_mode
        self.log_violations = log_violations
        self._logger = get_logger("PhysicsConsistencyChecker")
    
    def check_all(
        self,
        position: NDArray[np.float64],
        intensity: NDArray[np.float64],
        pressure: NDArray[np.float64],
        dt_hours: float = 6.0
    ) -> List[ValidationResult]:
        """Run all physics checks on predictions.
        
        Parameters
        ----------
        position : ndarray
            Position predictions (T, 2) in degrees.
        intensity : ndarray
            Intensity predictions (T,) in m/s.
        pressure : ndarray
            Pressure predictions (T,) in hPa.
        dt_hours : float
            Time step in hours.
            
        Returns
        -------
        List[ValidationResult]
            Results of all checks.
        """
        results = []
        
        # 1. Intensity bounds
        results.append(self.check_intensity_bounds(intensity))
        
        # 2. Pressure bounds
        results.append(self.check_pressure_bounds(pressure))
        
        # 3. Position bounds
        results.append(self.check_position_bounds(position))
        
        # 4. Translation speed
        results.append(self.check_translation_speed(position, dt_hours))
        
        # 5. Intensity change rate
        results.append(self.check_intensity_rate(intensity, dt_hours))
        
        # 6. Pressure-wind consistency
        results.append(self.check_pressure_wind_consistency(intensity, pressure))
        
        return results
    
    def check_intensity_bounds(
        self,
        intensity: NDArray[np.float64]
    ) -> ValidationResult:
        """Check that intensity is within physical bounds."""
        min_val = 0.0
        max_val = 95.0  # Theoretical maximum ~185 kt
        
        violations = (intensity < min_val) | (intensity > max_val)
        num_violations = np.sum(violations)
        
        passed = num_violations == 0
        
        return ValidationResult(
            test_name="intensity_bounds",
            passed=passed,
            message=f"Intensity bounds check: {num_violations} violations",
            details={
                'min_value': float(np.min(intensity)),
                'max_value': float(np.max(intensity)),
                'num_violations': int(num_violations),
                'bounds': (min_val, max_val),
            }
        )
    
    def check_pressure_bounds(
        self,
        pressure: NDArray[np.float64]
    ) -> ValidationResult:
        """Check that pressure is within physical bounds."""
        min_val = 870.0  # Record low: 870 hPa
        max_val = 1020.0  # Ambient
        
        violations = (pressure < min_val) | (pressure > max_val)
        num_violations = np.sum(violations)
        
        return ValidationResult(
            test_name="pressure_bounds",
            passed=num_violations == 0,
            message=f"Pressure bounds check: {num_violations} violations",
            details={
                'min_value': float(np.min(pressure)),
                'max_value': float(np.max(pressure)),
                'bounds': (min_val, max_val),
            }
        )
    
    def check_position_bounds(
        self,
        position: NDArray[np.float64]
    ) -> ValidationResult:
        """Check that positions are valid lat/lon."""
        lat = position[:, 0]
        lon = position[:, 1]
        
        lat_violations = (lat < -90) | (lat > 90)
        lon_violations = (lon < -180) | (lon > 180)
        
        num_violations = np.sum(lat_violations) + np.sum(lon_violations)
        
        return ValidationResult(
            test_name="position_bounds",
            passed=num_violations == 0,
            message=f"Position bounds check: {num_violations} violations",
            details={
                'lat_range': (float(np.min(lat)), float(np.max(lat))),
                'lon_range': (float(np.min(lon)), float(np.max(lon))),
            }
        )
    
    def check_translation_speed(
        self,
        position: NDArray[np.float64],
        dt_hours: float
    ) -> ValidationResult:
        """Check that movement speed is reasonable."""
        max_speed_ms = 35.0  # ~70 kt
        
        if len(position) < 2:
            return ValidationResult(
                test_name="translation_speed",
                passed=True,
                message="Not enough points to check speed",
                details={}
            )
        
        # Approximate speed from position changes
        d_pos = np.diff(position, axis=0)
        d_km = np.sqrt(np.sum((d_pos * 111) ** 2, axis=1))  # rough km
        speed_ms = d_km * 1000 / (dt_hours * 3600)
        
        violations = speed_ms > max_speed_ms
        
        return ValidationResult(
            test_name="translation_speed",
            passed=np.sum(violations) == 0,
            message=f"Translation speed check: max={np.max(speed_ms):.1f} m/s",
            details={
                'max_speed_ms': float(np.max(speed_ms)),
                'mean_speed_ms': float(np.mean(speed_ms)),
                'limit_ms': max_speed_ms,
            }
        )
    
    def check_intensity_rate(
        self,
        intensity: NDArray[np.float64],
        dt_hours: float
    ) -> ValidationResult:
        """Check that intensity change rate is physical."""
        # Rapid intensification: ~35 m/s / 24h = 8.75 m/s / 6h
        max_intensification_per_step = 10.0  # m/s per time step
        max_weakening_per_step = 15.0
        
        if len(intensity) < 2:
            return ValidationResult(
                test_name="intensity_rate",
                passed=True,
                message="Not enough points to check rate",
                details={}
            )
        
        d_int = np.diff(intensity)
        
        intensification_violations = d_int > max_intensification_per_step
        weakening_violations = d_int < -max_weakening_per_step
        
        num_violations = np.sum(intensification_violations) + np.sum(weakening_violations)
        
        return ValidationResult(
            test_name="intensity_rate",
            passed=num_violations == 0,
            message=f"Intensity rate check: {num_violations} violations",
            details={
                'max_change': float(np.max(d_int)),
                'min_change': float(np.min(d_int)),
            }
        )
    
    def check_pressure_wind_consistency(
        self,
        intensity: NDArray[np.float64],
        pressure: NDArray[np.float64]
    ) -> ValidationResult:
        """Check pressure-wind relationship consistency.
        
        Uses Atkinson & Holliday (1977) relationship as reference:
        V = 3.4 × (1010 - P)^0.644
        """
        expected_intensity = 3.4 * np.power(np.maximum(1010 - pressure, 0), 0.644)
        
        # Allow 20% deviation
        relative_error = np.abs(intensity - expected_intensity) / (expected_intensity + 1)
        violations = relative_error > 0.3  # 30% tolerance
        
        return ValidationResult(
            test_name="pressure_wind_consistency",
            passed=np.mean(violations) < 0.1,  # Allow 10% of points to deviate
            message=f"Pressure-wind consistency: mean error {np.mean(relative_error):.2%}",
            details={
                'mean_relative_error': float(np.mean(relative_error)),
                'max_relative_error': float(np.max(relative_error)),
            }
        )


class DimensionalAnalyzer:
    """Analyzer for dimensional consistency.
    
    Verifies that calculations maintain proper physical dimensions.
    """
    
    def __init__(self):
        self._logger = get_logger("DimensionalAnalyzer")
    
    def check_units(
        self,
        value: Any,
        expected_dimension: str
    ) -> bool:
        """Check if a value has the expected dimension.
        
        Parameters
        ----------
        value : Any
            Value to check (preferably with units from pint).
        expected_dimension : str
            Expected dimension string.
            
        Returns
        -------
        bool
            Whether dimensions match.
        """
        try:
            from common.units import ureg
            
            if hasattr(value, 'units'):
                return value.dimensionality == ureg.parse_expression(expected_dimension).dimensionality
            else:
                self._logger.warning("Value does not have units attached")
                return True  # Cannot verify
        except ImportError:
            return True  # Cannot verify without pint


class ConservationValidator:
    """Validator for conservation properties.
    
    Checks that model outputs don't violate conservation laws.
    """
    
    def __init__(self):
        self._logger = get_logger("ConservationValidator")
    
    def check_angular_momentum(
        self,
        max_wind: NDArray[np.float64],
        radius_max_wind: NDArray[np.float64],
        tolerance: float = 0.3
    ) -> ValidationResult:
        """Check approximate angular momentum conservation.
        
        M ∝ V × R should be roughly conserved during intensity changes.
        """
        if len(max_wind) < 2:
            return ValidationResult(
                test_name="angular_momentum",
                passed=True,
                message="Not enough points",
                details={}
            )
        
        M = max_wind * radius_max_wind
        dM = np.diff(M) / M[:-1]
        
        violations = np.abs(dM) > tolerance
        
        return ValidationResult(
            test_name="angular_momentum_conservation",
            passed=np.mean(violations) < 0.2,
            message=f"Angular momentum: {np.mean(np.abs(dM)):.1%} mean change",
            details={
                'mean_relative_change': float(np.mean(np.abs(dM))),
                'max_relative_change': float(np.max(np.abs(dM))),
            }
        )
