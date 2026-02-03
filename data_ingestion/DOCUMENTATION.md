# Data Ingestion Module Documentation

## Purpose & Scope

This module provides **standardized data loading interfaces** for all input data sources required by the cyclone prediction system.

### What This Module Does
- Load satellite imagery (GOES, Himawari, Meteosat)
- Load atmospheric reanalysis (ERA5, GFS, MERRA-2)
- Load terrain DEMs (SRTM, ASTER GDEM, Copernicus)
- Load best track data (IBTrACS, HURDAT2)
- Preserve metadata and provenance
- Validate units on ingest

### What This Module Does NOT Do
- Data interpolation (see `preprocessing/alignment.py`)
- Feature derivation (see `preprocessing/feature_engineering.py`)
- Physical calculations (see `physics_extraction/`)

---

## Scientific Context

### Domain
Data management, geospatial data handling, meteorological data formats.

### Data Model Families

| Data Type | Format | Temporal Resolution | Spatial Resolution |
|-----------|--------|--------------------|--------------------|
| Satellite | NetCDF, HDF5 | 10-30 min | 1-4 km |
| Reanalysis | GRIB, NetCDF | 1-6 hours | 0.25° (~25 km) |
| Terrain | GeoTIFF | Static | 30-90 m |
| Best Track | CSV, NetCDF | 6 hours | Point locations |

### Why These Data Sources

1. **Satellite imagery**: Direct observation of cloud structure and eye characteristics
2. **Reanalysis**: Spatially complete atmospheric state reconstruction
3. **Terrain**: Required for landfall impact and landslide modeling
4. **Best track**: Ground truth for validation and historical analysis

---

## Physical Assumptions & Limits

### Reanalysis Limitations
- Analysis products, not direct observations
- Tropical cyclone intensity often underestimated
- Resolution insufficient for inner-core structure

### Satellite Limitations
- Cloud-top only (no vertical structure)
- Parallax errors away from nadir
- Gaps due to scan geometry

### Best Track Limitations
- 6-hourly temporal resolution
- Subjective intensity estimates
- Post-season reanalysis may differ from operational

---

## Units & Dimensional Integrity

### Standard Units After Loading

| Variable | Unit | Notes |
|----------|------|-------|
| Wind components | m/s | ERA5 uses m s**-1 notation |
| Pressure | Pa | Converted from hPa if needed |
| Temperature | K | Always Kelvin, not Celsius |
| Geopotential | m²/s² | Divide by g for height in m |
| SST | K | Sea surface temperature |
| Elevation | m | Above WGS84 ellipsoid |

### Unit Validation

The `validate_units` flag enables automatic checking of variable units against expected values. Mismatches are logged as warnings.

---

## Verification & Validation Strategy

### Data Integrity Checks
1. No NaN/Inf values in critical variables
2. Values within physical bounds (e.g., pressure > 0)
3. Temporal monotonicity
4. Spatial coverage completeness

### Provenance Tracking
Every loaded dataset includes:
- Source identifier
- Version number
- Download timestamp
- Original CRS
- Processing steps applied

---

## Traceability

### Data Flow

```
Raw files (NetCDF, GRIB, GeoTIFF)
    │
    ├── SatelliteDataLoader.load()
    │       └── xr.Dataset (time, lat, lon, channel)
    │
    ├── AtmosphericReanalysisLoader.load()
    │       └── xr.Dataset (time, level, lat, lon)
    │
    ├── TerrainDEMLoader.load()
    │       └── xr.Dataset (lat, lon) + slope, aspect
    │
    └── BestTrackLoader.load()
            └── List[CycloneState]
```

### Downstream Dependencies
- `preprocessing/`: All loaders
- `vision_perception/`: Satellite data
- `physics_extraction/`: Reanalysis data
- `hazard_modeling/`: Terrain data
- `validation/`: Best track data

---

## References

1. Hersbach, H. et al. (2020). The ERA5 global reanalysis. QJRMS, 146(730), 1999-2049.
2. Knapp, K.R. et al. (2010). The International Best Track Archive for Climate Stewardship. BAMS.
3. Farr, T.G. et al. (2007). The Shuttle Radar Topography Mission. Reviews of Geophysics.
