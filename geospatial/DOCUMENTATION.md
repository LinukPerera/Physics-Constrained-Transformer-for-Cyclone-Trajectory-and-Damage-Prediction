# Geospatial Module Documentation

## Purpose & Scope

This module provides **all Earth-surface geometry calculations** for the cyclone prediction system. It is the single source of truth for:

- Coordinate transformations (geodetic, ECEF, ENU)
- Distance calculations (geodesic/ellipsoidal)
- Map projections with distortion quantification

### What This Module Does NOT Model

- Atmospheric dynamics (see `physics_extraction/`)
- Ocean surface currents
- Terrain-atmosphere interactions

---

## Scientific Context

### Domain
Geodesy, mathematical cartography, differential geometry on curved surfaces.

### Model Family
**Ellipsoidal Earth geometry** using the WGS84 reference ellipsoid.

### Why These Models Are Appropriate

Tropical cyclones travel thousands of kilometers across Earth's surface. The choice of Earth geometry model directly impacts:

1. **Trajectory accuracy**: Planar approximations introduce cumulative errors
2. **Wind field mapping**: Cyclone wind fields are defined relative to the surface
3. **Hazard extent calculation**: Damage footprints require accurate area calculations

For a 5000 km trans-Pacific cyclone track, the choice of model matters:

| Model | Error | Impact |
|-------|-------|--------|
| Planar/Euclidean | >100 km | Catastrophic |
| Spherical | ~25 km | Significant |
| Ellipsoidal | <15 nm | Negligible |

---

## Mathematical Model Class

### Coordinate Models (`coordinate_models.py`)

**Type**: Algebraic transformations, globally valid

**Model**: WGS84 reference ellipsoid

The WGS84 ellipsoid is defined by:
- Semi-major axis: a = 6,378,137.0 m (exact)
- Flattening: f = 1/298.257223563 (exact)

**Key transformations**:
1. Geodetic (φ, λ, h) ↔ ECEF (X, Y, Z)
2. Geodetic ↔ Local ENU (East, North, Up)

**Why simpler models are invalid**:
- Spherical assumption ignores 21 km polar flattening
- Planar models fail for any distance > 100 km

### Distance Calculations (`distance_calculations.py`)

**Type**: Geodesic (shortest path on ellipsoid)

**Algorithm**: Karney's method (via GeographicLib/pyproj)

This algorithm:
- Converges for all point pairs including antipodal
- Provides 15-nanometer precision
- Returns both distance and azimuths

**Why simpler models are invalid**:
- Haversine (spherical great-circle): 0.3-0.5% error
- Rhumb line: Not shortest path, longer distances
- Planar: Error grows quadratically

### Projections (`projections.py`)

**Type**: Conformal map projections with distortion tracking

**Models**:
1. Transverse Mercator (for narrow N-S extent)
2. Lambert Conformal Conic (for mid-latitude E-W extent)

**Key feature**: Tissot's indicatrix computation quantifies local distortion, enabling informed use of projected coordinates.

---

## Physical Assumptions & Limits

### Assumptions
1. **Rigid Earth**: No tectonic motion during forecast period
2. **WGS84 validity**: The reference ellipsoid is appropriate for all cyclone basins
3. **Negligible altitude**: Most calculations assume sea-level (h ≈ 0)

### Valid Ranges
- Latitude: -90° to +90° (but ENU accuracy degrades > 85°)
- Longitude: -180° to +180°
- Distance: 0 to ~20,000 km (half Earth circumference)

### Known Breakdown Conditions
1. **Near poles (|φ| > 85°)**: Coordinate singularities require special handling
2. **Antipodal points**: Geodesic is not unique (handled by Karney algorithm)
3. **Very short distances (< 1 m)**: Numerical precision limits apply

---

## Units & Dimensional Integrity

### Expected Units

| Quantity | Internal Unit | Notes |
|----------|---------------|-------|
| Latitude | radians | Range: [-π/2, π/2] |
| Longitude | radians | Range: [-π, π] |
| Altitude | meters | Above WGS84 ellipsoid |
| Distance | meters | Geodesic (ellipsoidal) |
| Azimuth | radians | Clockwise from north |
| Projected coords | meters | In projection plane |

### Unit Consistency Enforcement

1. All public functions accept radians (internal standard)
2. Convenience functions provide degree conversions (`GeoCoordinate.from_degrees()`)
3. Return values document units in function signatures

### Violation Detection

- `GeoCoordinate.__post_init__()` validates latitude range
- Large degree values (> 2π) trigger automatic detection

---

## Verification & Validation Strategy

### Independent Reference Comparisons

All functions are validated against:

1. **GeographicLib**: Charles Karney's reference implementation
2. **NGS Geodetic Test Points**: US National Geodetic Survey reference data
3. **PROJ Library**: Widely-used cartographic library

### Synthetic Test Cases

| Test | Input | Expected | Tolerance |
|------|-------|----------|-----------|
| Equator circumference | 0°,0° to 0°,180° | 20,003,931.46 m | 1 mm |
| Pole distance | 0°,0° to 90°,0° | 10,001,965.73 m | 1 mm |
| ECEF round-trip | Any valid coord | Original coord | 1e-10 rad |

### Expected Numerical Tolerances

- Geodesic distance: < 15 nm for any input
- Coordinate round-trip: < 1e-10 radians
- Projection distortion: < 0.1% within valid domain

### Failure Mode Detection

1. **NaN/Inf outputs**: Explicit check in all functions
2. **Convergence failure**: Maximum iteration limits with warnings
3. **Out-of-bounds inputs**: Validation with clear error messages

---

## Traceability

### Input → Transformation → Output

```
GeoCoordinate (lat, lon, alt)
    │
    ├── geodetic_to_ecef() → (X, Y, Z) ECEF
    │
    ├── geodesic_inverse() → GeodesicResult (distance, azimuths)
    │
    └── projection.to_projected() → (x, y) + TissotIndicatrix
```

### Audit Requirements

All geospatial calculations used in trajectory prediction are logged with:
- Input coordinates
- Calculation type
- Output values
- Timestamp

### Downstream Module Dependencies

Modules that MUST import from `geospatial/`:
- `preprocessing/alignment.py` (for spatial regridding)
- `physics_extraction/` (for derivative calculations)
- `trajectory_prediction/` (for track generation)
- `hazard_modeling/` (for damage extent mapping)

---

## References

1. NIMA TR8350.2 (2000). Department of Defense World Geodetic System 1984.
2. Karney, C.F.F. (2013). Algorithms for geodesics. *Journal of Geodesy*, 87(1), 43-55.
3. Snyder, J.P. (1987). Map Projections - A Working Manual. USGS Professional Paper 1395.
4. Torge, W. (2001). *Geodesy* (3rd ed.). de Gruyter.
5. Hofmann-Wellenhof, B. et al. (2008). *GNSS: GPS, GLONASS, Galileo*.
