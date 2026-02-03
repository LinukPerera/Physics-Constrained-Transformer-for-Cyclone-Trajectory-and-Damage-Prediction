"""
Unit Registry and Dimensional Analysis for Physics-Constrained Cyclone Prediction.

This module provides a centralized unit system using the `pint` library to ensure
dimensional consistency across all calculations. Every physical quantity in the
system should be tagged with units, and operations between incompatible units
will raise errors at runtime.

Scientific Context
------------------
Dimensional analysis is a fundamental validation technique in physics. By enforcing
units at the computation level, we catch errors that would otherwise propagate
silently through the system.

Example Usage
-------------
>>> from common.units import ureg, Q_
>>> distance = Q_(100, 'km')
>>> time = Q_(2, 'hour')
>>> speed = distance / time
>>> speed.to('m/s')
<Quantity(13.8889, 'meter / second')>
"""

from functools import wraps
from typing import Any, Callable, TypeVar, Union
import warnings

import pint
from pint import UnitRegistry as PintUnitRegistry

# Create the global unit registry
ureg = PintUnitRegistry()
ureg.setup_matplotlib(True)  # Enable matplotlib integration

# Convenience alias for creating quantities
Q_ = ureg.Quantity


class UnitRegistry:
    """Wrapper around pint UnitRegistry with cyclone-specific extensions.
    
    This class provides access to the global unit registry and defines
    additional units and contexts specific to atmospheric science.
    
    Attributes
    ----------
    registry : pint.UnitRegistry
        The underlying pint unit registry.
        
    Examples
    --------
    >>> units = UnitRegistry()
    >>> pressure = units.quantity(1013.25, 'hPa')
    >>> pressure.to('Pa')
    <Quantity(101325.0, 'pascal')>
    """
    
    def __init__(self):
        """Initialize the unit registry with atmospheric science extensions."""
        self._registry = ureg
        self._setup_atmospheric_units()
        self._setup_cyclone_contexts()
    
    def _setup_atmospheric_units(self) -> None:
        """Define additional units common in atmospheric science."""
        # Only define if not already defined
        try:
            # Pressure units
            self._registry.define("hPa = hectopascal")
            self._registry.define("mbar = millibar = hPa")
            
            # Wind speed units
            self._registry.define("knot = nautical_mile / hour = kt = kn")
            
            # Temperature contexts
            self._registry.define("degC = kelvin; offset: 273.15 = celsius")
            
        except pint.errors.RedefinitionError:
            # Units already defined, skip
            pass
    
    def _setup_cyclone_contexts(self) -> None:
        """Set up unit contexts for cyclone-specific conversions."""
        # Context for cyclone intensity comparisons
        pass  # Future: Add context for wind averaging periods
    
    @property
    def registry(self) -> PintUnitRegistry:
        """Access the underlying pint registry."""
        return self._registry
    
    def quantity(self, value: float, unit: str) -> pint.Quantity:
        """Create a quantity with units.
        
        Parameters
        ----------
        value : float
            The numerical value.
        unit : str
            The unit string (e.g., 'm/s', 'hPa', 'degC').
            
        Returns
        -------
        pint.Quantity
            A quantity object with associated units.
        """
        return self._registry.Quantity(value, unit)
    
    def validate_dimensionality(
        self, 
        quantity: pint.Quantity, 
        expected_dim: str
    ) -> bool:
        """Check if a quantity has the expected dimensionality.
        
        Parameters
        ----------
        quantity : pint.Quantity
            The quantity to check.
        expected_dim : str
            The expected dimensionality (e.g., '[length]', '[velocity]').
            
        Returns
        -------
        bool
            True if dimensionality matches.
            
        Raises
        ------
        pint.DimensionalityError
            If dimensionality does not match.
        """
        expected = self._registry.parse_expression(expected_dim).dimensionality
        if quantity.dimensionality != expected:
            raise pint.DimensionalityError(
                quantity.units, 
                expected,
                quantity.dimensionality,
                expected
            )
        return True


def validate_units(expected_units: dict[str, str]):
    """Decorator to validate units of function arguments and return values.
    
    Parameters
    ----------
    expected_units : dict[str, str]
        Mapping from argument names to expected unit strings.
        Use 'return' key for return value validation.
        
    Examples
    --------
    >>> @validate_units({'distance': 'm', 'time': 's', 'return': 'm/s'})
    ... def calculate_speed(distance, time):
    ...     return distance / time
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate input arguments
            for param_name, expected_unit in expected_units.items():
                if param_name == 'return':
                    continue
                    
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if isinstance(value, pint.Quantity):
                        try:
                            # Check if units are compatible
                            value.to(expected_unit)
                        except pint.DimensionalityError as e:
                            raise ValueError(
                                f"Parameter '{param_name}' has incompatible units. "
                                f"Expected {expected_unit}, got {value.units}"
                            ) from e
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Validate return value
            if 'return' in expected_units and isinstance(result, pint.Quantity):
                try:
                    result.to(expected_units['return'])
                except pint.DimensionalityError as e:
                    raise ValueError(
                        f"Return value has incompatible units. "
                        f"Expected {expected_units['return']}, got {result.units}"
                    ) from e
            
            return result
        return wrapper
    return decorator


def ensure_quantity(value: Union[float, pint.Quantity], default_unit: str) -> pint.Quantity:
    """Ensure a value is a pint Quantity, applying default unit if necessary.
    
    Parameters
    ----------
    value : float or pint.Quantity
        The value to convert.
    default_unit : str
        The unit to apply if value is a bare number.
        
    Returns
    -------
    pint.Quantity
        The value with units.
        
    Warnings
    --------
    Issues a warning if a bare number is provided without units.
    """
    if isinstance(value, pint.Quantity):
        return value
    else:
        warnings.warn(
            f"Bare number {value} provided without units. "
            f"Assuming {default_unit}. Consider using explicit units.",
            UserWarning,
            stacklevel=2
        )
        return ureg.Quantity(value, default_unit)


# Standard unit definitions for the system
STANDARD_UNITS = {
    # Spatial
    "latitude": "radian",
    "longitude": "radian",
    "distance": "meter",
    "altitude": "meter",
    
    # Temporal
    "time": "second",
    "forecast_hour": "hour",
    
    # Atmospheric state
    "pressure": "pascal",
    "temperature": "kelvin",
    "wind_speed": "meter/second",
    "wind_direction": "radian",
    
    # Cyclone-specific
    "vorticity": "1/second",
    "divergence": "1/second",
    
    # Derived
    "velocity": "meter/second",
    "acceleration": "meter/second**2",
    "force_per_mass": "meter/second**2",
}
