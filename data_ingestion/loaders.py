"""
Data Loaders for Cyclone Prediction System.

This module provides standardized interfaces for loading various data sources
required for cyclone trajectory and impact prediction. Each loader validates
metadata, preserves provenance, and ensures unit consistency.

Supported Data Sources
----------------------
1. Satellite imagery (GOES, Himawari, Meteosat)
2. Atmospheric reanalysis (ERA5, GFS, MERRA-2)
3. Terrain DEMs (SRTM, ASTER GDEM, Copernicus)
4. Best track data (IBTrACS, HURDAT2)

Design Principles
-----------------
- Lazy loading for memory efficiency
- Metadata preservation for traceability
- Unit validation on ingest
- Consistent output formats via xarray
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import logging

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from common.logging_config import get_logger
from common.types import GeoCoordinate, CycloneState
from common.units import ureg, Q_

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion.
    
    Attributes
    ----------
    data_root : Path
        Root directory for data files.
    satellite_subdir : str
        Subdirectory for satellite data.
    reanalysis_subdir : str
        Subdirectory for atmospheric reanalysis.
    terrain_subdir : str
        Subdirectory for terrain DEMs.
    track_subdir : str
        Subdirectory for best track data.
    cache_enabled : bool
        Whether to cache loaded data.
    validate_units : bool
        Whether to validate units on ingest.
    """
    data_root: Path
    satellite_subdir: str = "satellite"
    reanalysis_subdir: str = "reanalysis"
    terrain_subdir: str = "terrain"
    track_subdir: str = "tracks"
    cache_enabled: bool = True
    validate_units: bool = True


@dataclass
class DataProvenance:
    """Metadata tracking data provenance.
    
    Attributes
    ----------
    source : str
        Data source identifier (e.g., "ERA5", "GOES-16").
    version : str
        Version of the dataset.
    download_time : datetime
        When the data was downloaded/accessed.
    spatial_coverage : Tuple[float, float, float, float]
        Bounding box (min_lat, max_lat, min_lon, max_lon) in degrees.
    temporal_coverage : Tuple[datetime, datetime]
        Time range coverage.
    original_crs : str
        Original coordinate reference system.
    processing_steps : List[str]
        List of processing steps applied.
    """
    source: str
    version: str
    download_time: datetime
    spatial_coverage: Tuple[float, float, float, float]
    temporal_coverage: Tuple[datetime, datetime]
    original_crs: str = "EPSG:4326"
    processing_steps: List[str] = field(default_factory=list)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders.
    
    All data loaders must implement this interface to ensure consistent
    handling of data sources throughout the system.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """Initialize the loader.
        
        Parameters
        ----------
        config : DataIngestionConfig
            Configuration for data ingestion.
        """
        self.config = config
        self._cache: Dict[str, Any] = {}
        self._logger = get_logger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def data_type(self) -> str:
        """Identifier for this data type."""
        pass
    
    @abstractmethod
    def load(
        self,
        start_time: datetime,
        end_time: datetime,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> xr.Dataset:
        """Load data for a given time range and spatial extent.
        
        Parameters
        ----------
        start_time, end_time : datetime
            Time range to load.
        bounds : tuple, optional
            Spatial bounds (min_lat, max_lat, min_lon, max_lon) in degrees.
            
        Returns
        -------
        xr.Dataset
            Loaded data with standardized dimensions and coordinates.
        """
        pass
    
    @abstractmethod
    def get_provenance(self) -> DataProvenance:
        """Get provenance information for loaded data."""
        pass
    
    def _validate_units(self, ds: xr.Dataset, expected_units: Dict[str, str]) -> None:
        """Validate that variables have expected units.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate.
        expected_units : dict
            Mapping of variable names to expected unit strings.
        """
        if not self.config.validate_units:
            return
            
        for var_name, expected in expected_units.items():
            if var_name in ds:
                actual = ds[var_name].attrs.get('units', 'unknown')
                if actual != expected:
                    self._logger.warning(
                        f"Unit mismatch for {var_name}: expected {expected}, got {actual}"
                    )


class SatelliteDataLoader(BaseDataLoader):
    """Loader for satellite imagery data.
    
    Supports:
    - GOES-16/17 (NetCDF, HDF5)
    - Himawari-8/9 (HSD, NetCDF)
    - Meteosat (NetCDF)
    
    Notes
    -----
    Satellite data is typically provided in native satellite projection.
    This loader reprojects to WGS84 lat/lon grid for consistency.
    """
    
    @property
    def data_type(self) -> str:
        return "satellite"
    
    def load(
        self,
        start_time: datetime,
        end_time: datetime,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        channels: Optional[List[str]] = None
    ) -> xr.Dataset:
        """Load satellite imagery.
        
        Parameters
        ----------
        start_time, end_time : datetime
            Time range to load.
        bounds : tuple, optional
            Spatial bounds (min_lat, max_lat, min_lon, max_lon).
        channels : list, optional
            Specific channels to load (e.g., ['IR', 'VIS', 'WV']).
            
        Returns
        -------
        xr.Dataset
            Dataset with dimensions (time, lat, lon) and channel variables.
            
        Notes
        -----
        Standard channels:
        - IR: Infrared (brightness temperature)
        - VIS: Visible (reflectance)
        - WV: Water vapor (brightness temperature)
        """
        data_dir = self.config.data_root / self.config.satellite_subdir
        
        # Find relevant files
        # TODO: Implement file discovery based on time range
        
        # Placeholder: Create empty dataset with expected structure
        self._logger.info(
            f"Loading satellite data from {start_time} to {end_time}"
        )
        
        # Define standard output structure
        times = np.array([start_time])
        lats = np.arange(-60, 60, 0.1)  # Placeholder
        lons = np.arange(-180, 180, 0.1)  # Placeholder
        
        ds = xr.Dataset(
            coords={
                'time': ('time', times),
                'latitude': ('latitude', lats),
                'longitude': ('longitude', lons),
            },
            attrs={
                'source': 'satellite',
                'loader': 'SatelliteDataLoader',
                'load_time': datetime.now().isoformat(),
            }
        )
        
        return ds
    
    def get_provenance(self) -> DataProvenance:
        return DataProvenance(
            source="satellite",
            version="1.0",
            download_time=datetime.now(),
            spatial_coverage=(-60, 60, -180, 180),
            temporal_coverage=(datetime.now(), datetime.now()),
            original_crs="native_satellite"
        )


class AtmosphericReanalysisLoader(BaseDataLoader):
    """Loader for atmospheric reanalysis data.
    
    Supports:
    - ERA5 (ECMWF)
    - GFS (NOAA)
    - MERRA-2 (NASA)
    
    Variables
    ---------
    Standard variables loaded:
    - u10, v10: 10m wind components (m/s)
    - msl: Mean sea level pressure (Pa)
    - t2m: 2m temperature (K)
    - sst: Sea surface temperature (K)
    - z: Geopotential height (m²/s²)
    - u, v: Wind components at pressure levels (m/s)
    - q: Specific humidity (kg/kg)
    - t: Temperature at pressure levels (K)
    """
    
    EXPECTED_UNITS = {
        'u10': 'm s**-1',
        'v10': 'm s**-1',
        'msl': 'Pa',
        't2m': 'K',
        'sst': 'K',
        'z': 'm**2 s**-2',
        'u': 'm s**-1',
        'v': 'm s**-1',
        'q': 'kg kg**-1',
        't': 'K',
    }
    
    @property
    def data_type(self) -> str:
        return "reanalysis"
    
    def load(
        self,
        start_time: datetime,
        end_time: datetime,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        pressure_levels: Optional[List[int]] = None,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """Load atmospheric reanalysis data.
        
        Parameters
        ----------
        start_time, end_time : datetime
            Time range to load.
        bounds : tuple, optional
            Spatial bounds (min_lat, max_lat, min_lon, max_lon).
        pressure_levels : list, optional
            Pressure levels to load in hPa (e.g., [1000, 850, 500, 200]).
        variables : list, optional
            Specific variables to load.
            
        Returns
        -------
        xr.Dataset
            Dataset with dimensions (time, level, lat, lon).
        """
        data_dir = self.config.data_root / self.config.reanalysis_subdir
        
        if pressure_levels is None:
            pressure_levels = [1000, 925, 850, 700, 500, 300, 200]
        
        self._logger.info(
            f"Loading reanalysis data from {start_time} to {end_time}"
        )
        
        # Placeholder: Create empty dataset with expected structure
        times = np.array([start_time])
        levels = np.array(pressure_levels)
        lats = np.arange(-90, 90.1, 0.25)
        lons = np.arange(-180, 180, 0.25)
        
        ds = xr.Dataset(
            coords={
                'time': ('time', times),
                'level': ('level', levels),
                'latitude': ('latitude', lats),
                'longitude': ('longitude', lons),
            },
            attrs={
                'source': 'reanalysis',
                'loader': 'AtmosphericReanalysisLoader',
                'load_time': datetime.now().isoformat(),
                'pressure_level_units': 'hPa',
            }
        )
        
        return ds
    
    def get_provenance(self) -> DataProvenance:
        return DataProvenance(
            source="ERA5",
            version="1.0",
            download_time=datetime.now(),
            spatial_coverage=(-90, 90, -180, 180),
            temporal_coverage=(datetime.now(), datetime.now()),
            original_crs="EPSG:4326"
        )
    
    def load_single_level(
        self,
        start_time: datetime,
        end_time: datetime,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """Load single-level (surface) reanalysis data.
        
        This is for variables that don't vary with pressure level,
        such as sea surface temperature, 10m winds, mean sea level pressure.
        """
        return self.load(
            start_time, end_time, bounds,
            pressure_levels=None,
            variables=variables or ['u10', 'v10', 'msl', 't2m', 'sst']
        )


class TerrainDEMLoader(BaseDataLoader):
    """Loader for terrain Digital Elevation Models.
    
    Supports:
    - SRTM (NASA)
    - ASTER GDEM (NASA/METI)
    - Copernicus DEM (ESA)
    
    Output
    ------
    Elevation in meters above WGS84 ellipsoid.
    Also computes derived products:
    - Slope (degrees)
    - Aspect (degrees from north)
    - Roughness (standard deviation of elevation)
    """
    
    @property
    def data_type(self) -> str:
        return "terrain"
    
    def load(
        self,
        start_time: datetime = None,  # Not used for terrain
        end_time: datetime = None,    # Not used for terrain
        bounds: Optional[Tuple[float, float, float, float]] = None,
        resolution_m: float = 90.0
    ) -> xr.Dataset:
        """Load terrain DEM data.
        
        Parameters
        ----------
        bounds : tuple
            Spatial bounds (min_lat, max_lat, min_lon, max_lon).
        resolution_m : float
            Target resolution in meters.
            
        Returns
        -------
        xr.Dataset
            Dataset with variables: elevation, slope, aspect.
        """
        if bounds is None:
            raise ValueError("Bounds are required for terrain loading")
            
        data_dir = self.config.data_root / self.config.terrain_subdir
        
        self._logger.info(
            f"Loading terrain DEM for bounds {bounds} at {resolution_m}m resolution"
        )
        
        # Placeholder structure
        min_lat, max_lat, min_lon, max_lon = bounds
        lats = np.arange(min_lat, max_lat, 0.001)  # ~100m
        lons = np.arange(min_lon, max_lon, 0.001)
        
        ds = xr.Dataset(
            coords={
                'latitude': ('latitude', lats),
                'longitude': ('longitude', lons),
            },
            attrs={
                'source': 'terrain_dem',
                'loader': 'TerrainDEMLoader',
                'load_time': datetime.now().isoformat(),
                'resolution_m': resolution_m,
            }
        )
        
        return ds
    
    def get_provenance(self) -> DataProvenance:
        return DataProvenance(
            source="SRTM_V3",
            version="1.0",
            download_time=datetime.now(),
            spatial_coverage=(-60, 60, -180, 180),
            temporal_coverage=(datetime(2000, 1, 1), datetime(2000, 12, 31)),
            original_crs="EPSG:4326"
        )


class BestTrackLoader(BaseDataLoader):
    """Loader for tropical cyclone best track data.
    
    Supports:
    - IBTrACS (International Best Track Archive)
    - HURDAT2 (NHC Atlantic/Pacific)
    - JTWC (Joint Typhoon Warning Center)
    
    Output
    ------
    List of CycloneState objects with position, intensity, and structure.
    """
    
    @property
    def data_type(self) -> str:
        return "best_track"
    
    def load(
        self,
        start_time: datetime,
        end_time: datetime,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        storm_id: Optional[str] = None,
        basin: Optional[str] = None
    ) -> List[CycloneState]:
        """Load best track data.
        
        Parameters
        ----------
        start_time, end_time : datetime
            Time range to load.
        bounds : tuple, optional
            Spatial bounds (min_lat, max_lat, min_lon, max_lon).
        storm_id : str, optional
            Specific storm ID (e.g., "AL092017" for Irma).
        basin : str, optional
            Basin code: 'AL' (Atlantic), 'EP' (East Pacific),
            'WP' (West Pacific), 'IO' (Indian Ocean), etc.
            
        Returns
        -------
        List[CycloneState]
            List of cyclone states ordered by time.
        """
        data_dir = self.config.data_root / self.config.track_subdir
        
        self._logger.info(
            f"Loading best track data from {start_time} to {end_time}"
        )
        
        # Placeholder: Return empty list
        return []
    
    def load_storm(self, storm_id: str) -> List[CycloneState]:
        """Load all track points for a specific storm.
        
        Parameters
        ----------
        storm_id : str
            Storm identifier (e.g., "AL092017").
            
        Returns
        -------
        List[CycloneState]
            Complete track of the storm.
        """
        return self.load(
            start_time=datetime(1850, 1, 1),
            end_time=datetime.now(),
            storm_id=storm_id
        )
    
    def get_provenance(self) -> DataProvenance:
        return DataProvenance(
            source="IBTrACS",
            version="v04r00",
            download_time=datetime.now(),
            spatial_coverage=(-90, 90, -180, 180),
            temporal_coverage=(datetime(1850, 1, 1), datetime.now()),
            original_crs="EPSG:4326"
        )
    
    def list_storms(
        self,
        year: int,
        basin: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all storms in a given year.
        
        Parameters
        ----------
        year : int
            Year to query.
        basin : str, optional
            Basin code to filter by.
            
        Returns
        -------
        List[dict]
            List of storm metadata (id, name, max_intensity, etc.).
        """
        self._logger.info(f"Listing storms for {year}, basin={basin}")
        return []
