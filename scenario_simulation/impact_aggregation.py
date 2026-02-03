"""
Impact Aggregation for Regional Assessment.

This module aggregates hazard impacts across different dimensions:
- Spatial: by region, grid cell, or administrative boundary
- Temporal: peak impacts, cumulative exposure, duration
- Sectoral: residential, commercial, infrastructure
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from numpy.typing import NDArray
from datetime import datetime

from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RegionalImpactSummary:
    """Summary of impacts for a region.
    
    Attributes
    ----------
    region_id : str
        Region identifier.
    region_name : str
        Human-readable region name.
    population_exposed : int
        Estimated population exposed to hazard.
    area_affected_km2 : float
        Area affected in square kilometers.
    peak_wind_ms : float
        Peak wind speed in m/s.
    max_surge_m : float
        Maximum storm surge in meters.
    total_rainfall_mm : float
        Total accumulated rainfall in mm.
    damage_estimate_usd : float
        Estimated economic damage.
    affected_infrastructure : Dict[str, int]
        Count of affected infrastructure by type.
    uncertainty_bounds : Dict[str, tuple]
        Uncertainty ranges for key metrics.
    """
    region_id: str
    region_name: str
    population_exposed: int = 0
    area_affected_km2: float = 0.0
    peak_wind_ms: float = 0.0
    max_surge_m: float = 0.0
    total_rainfall_mm: float = 0.0
    damage_estimate_usd: float = 0.0
    affected_infrastructure: Dict[str, int] = field(default_factory=dict)
    uncertainty_bounds: Dict[str, tuple] = field(default_factory=dict)


class ImpactAggregator:
    """Aggregates spatially distributed impacts.
    
    Takes gridded impact data and aggregates to regions, computes
    totals, and summarizes for reporting.
    """
    
    def __init__(
        self,
        region_definitions: Optional[Dict[str, NDArray[np.bool_]]] = None,
        population_grid: Optional[NDArray[np.float64]] = None,
        grid_resolution_km: float = 1.0
    ):
        """Initialize impact aggregator.
        
        Parameters
        ----------
        region_definitions : dict, optional
            Mapping from region ID to boolean mask array.
        population_grid : ndarray, optional
            Grid of population counts.
        grid_resolution_km : float
            Grid cell size in km.
        """
        self.regions = region_definitions or {}
        self.population = population_grid
        self.cell_area_km2 = grid_resolution_km ** 2
        self._logger = get_logger("ImpactAggregator")
    
    def aggregate_wind_impact(
        self,
        wind_speed: NDArray[np.float64],
        damage_ratio: NDArray[np.float64],
        region_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregate wind impact metrics.
        
        Parameters
        ----------
        wind_speed : ndarray
            Wind speed grid in m/s.
        damage_ratio : ndarray
            Damage ratio grid (0-1).
        region_id : str, optional
            Specific region to aggregate; if None, aggregate all.
            
        Returns
        -------
        dict
            Aggregated impact metrics.
        """
        if region_id and region_id in self.regions:
            mask = self.regions[region_id]
        else:
            mask = np.ones_like(wind_speed, dtype=bool)
        
        metrics = {
            'peak_wind_ms': float(np.max(wind_speed[mask])),
            'mean_wind_ms': float(np.mean(wind_speed[mask])),
            'area_exposed_km2': float(np.sum(mask) * self.cell_area_km2),
        }
        
        # Population exposure if available
        if self.population is not None:
            # Exposed = wind > 17 m/s (tropical storm)
            exposed_mask = mask & (wind_speed > 17)
            metrics['population_exposed'] = int(np.sum(self.population[exposed_mask]))
            
            # Severely exposed = wind > 43 m/s (Cat 2)
            severe_mask = mask & (wind_speed > 43)
            metrics['population_severe_exposure'] = int(np.sum(self.population[severe_mask]))
        
        # Damage statistics
        metrics['mean_damage_ratio'] = float(np.mean(damage_ratio[mask]))
        metrics['total_damage_area_km2'] = float(
            np.sum(damage_ratio[mask] > 0.1) * self.cell_area_km2
        )
        
        return metrics
    
    def aggregate_flood_impact(
        self,
        flood_depth: NDArray[np.float64],
        region_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregate flood impact metrics.
        
        Parameters
        ----------
        flood_depth : ndarray
            Flood depth grid in meters.
        region_id : str, optional
            Specific region.
            
        Returns
        -------
        dict
            Aggregated flood metrics.
        """
        if region_id and region_id in self.regions:
            mask = self.regions[region_id]
        else:
            mask = np.ones_like(flood_depth, dtype=bool)
        
        flooded = mask & (flood_depth > 0)
        
        metrics = {
            'max_depth_m': float(np.max(flood_depth[mask])),
            'mean_depth_m': float(np.mean(flood_depth[flooded])) if flooded.any() else 0,
            'flooded_area_km2': float(np.sum(flooded) * self.cell_area_km2),
        }
        
        # Depth thresholds
        metrics['area_depth_gt_0.5m'] = float(
            np.sum(mask & (flood_depth > 0.5)) * self.cell_area_km2
        )
        metrics['area_depth_gt_1m'] = float(
            np.sum(mask & (flood_depth > 1.0)) * self.cell_area_km2
        )
        metrics['area_depth_gt_2m'] = float(
            np.sum(mask & (flood_depth > 2.0)) * self.cell_area_km2
        )
        
        # Population exposure
        if self.population is not None:
            metrics['population_flooded'] = int(np.sum(self.population[flooded]))
        
        return metrics
    
    def create_regional_summary(
        self,
        region_id: str,
        region_name: str,
        wind_metrics: Dict[str, Any],
        flood_metrics: Optional[Dict[str, Any]] = None
    ) -> RegionalImpactSummary:
        """Create comprehensive regional impact summary.
        
        Parameters
        ----------
        region_id : str
            Region identifier.
        region_name : str
            Human-readable name.
        wind_metrics : dict
            Aggregated wind metrics.
        flood_metrics : dict, optional
            Aggregated flood metrics.
            
        Returns
        -------
        RegionalImpactSummary
            Complete regional summary.
        """
        summary = RegionalImpactSummary(
            region_id=region_id,
            region_name=region_name,
            peak_wind_ms=wind_metrics.get('peak_wind_ms', 0),
            area_affected_km2=wind_metrics.get('area_exposed_km2', 0),
            population_exposed=wind_metrics.get('population_exposed', 0),
        )
        
        if flood_metrics:
            summary.max_surge_m = flood_metrics.get('max_depth_m', 0)
        
        return summary
    
    def combine_ensemble_results(
        self,
        ensemble_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine results from ensemble members.
        
        Parameters
        ----------
        ensemble_results : list
            List of result dictionaries from ensemble.
            
        Returns
        -------
        dict
            Combined statistics with uncertainty bounds.
        """
        if not ensemble_results:
            return {}
        
        # Get all keys
        keys = set()
        for result in ensemble_results:
            keys.update(result.keys())
        
        combined = {}
        
        for key in keys:
            values = [r.get(key) for r in ensemble_results if key in r]
            
            if not values:
                continue
            
            if isinstance(values[0], (int, float)):
                values = np.array(values)
                combined[f'{key}_mean'] = float(np.mean(values))
                combined[f'{key}_std'] = float(np.std(values))
                combined[f'{key}_p10'] = float(np.percentile(values, 10))
                combined[f'{key}_p50'] = float(np.percentile(values, 50))
                combined[f'{key}_p90'] = float(np.percentile(values, 90))
        
        return combined
