# Preprocessing Module Documentation

## Purpose & Scope

This module provides **physics-aware data preprocessing** that preserves physical meaning and dimensional interpretability throughout all transformations.

### What This Module Does
- Temporal alignment with physical validity checks
- Spatial regridding with conservation properties
- Storm-following coordinate transformation
- Reversible normalization with stored parameters
- Feature engineering with spherical derivatives

### What This Module Does NOT Do
- Raw data loading (see `data_ingestion/`)
- Physical dynamics calculations (see `physics_extraction/`)
- Machine learning operations (see `temporal_model/`)

---

## Scientific Context

### Domain
Geophysical data preprocessing, numerical meteorology, feature engineering.

### Key Mathematical Operations

| Operation | Mathematical Model | Why Chosen |
|-----------|-------------------|------------|
| Vorticity | Spherical curl | Cartesian curl invalid on sphere |
| Divergence | Spherical divergence | Proper for atmospheric mass flux |
| Regridding | Area-weighted | Preserves integral quantities |
| Normalization | Anomaly-based | Climate signal removal |

---

## Mathematical Model Class

### Alignment (`alignment.py`)

**Type**: Interpolation with physics constraints

**Key transformations**:
1. Temporal interpolation with monotonicity checks
2. Conservative spatial regridding
3. Storm-following polar coordinate transform

**Why simpler models are invalid**:
- Bilinear interpolation can create non-physical values (negative pressure)
- Cartesian regridding distorts at high latitudes
- Earth-fixed coordinates lose storm-relative patterns

### Normalization (`normalization.py`)

**Type**: Reversible statistical transforms

**Strategies by variable type**:
- Intensive (T, P): Standard z-score or anomaly
- Extensive (precipitation): Positive-preserving
- Bounded (humidity): Min-max to [0, 1]
- Vector (wind): Components normalized together

### Feature Engineering (`feature_engineering.py`)

**Type**: Spherical differential operators

**Mathematical forms**:

Vorticity on ellipsoid:
```
ζ = (1/(N cos φ)) ∂v/∂λ - (1/M) ∂u/∂φ + (u tan φ)/N
```

Divergence on ellipsoid:
```
D = (1/(N cos φ)) ∂u/∂λ + (1/(M cos φ)) ∂(v cos φ)/∂φ
```

where M, N are meridian and prime vertical radii of curvature.

---

## Physical Assumptions & Limits

### Alignment
- Interpolation assumes smooth fields
- Conservative regridding requires closed boundaries
- Storm-following assumes circular structure

### Normalization
- Climatology from training period representative
- Standard deviation is a good uncertainty measure
- Variables are approximately Gaussian (or transformable)

### Feature Engineering
- Hydrostatic balance for pressure-height conversion
- Beta-plane valid within ±60° latitude
- Finite differences accurate for grid spacing << radius of curvature

---

## Units & Dimensional Integrity

### Feature Units

| Feature | Unit | Typical Range |
|---------|------|---------------|
| Vorticity | s⁻¹ | ±10⁻⁴ |
| Divergence | s⁻¹ | ±10⁻⁵ |
| Wind shear | m/s | 0-30 |
| Normalized T | dimensionless | ±3 |

### Reversibility Guarantee

All normalization transforms store parameters and can be exactly reversed:
```python
normalizer.transform(data)  # Forward
normalizer.inverse_transform(data)  # Back to original
```

---

## Verification & Validation Strategy

### Numerical Tests
1. Vorticity of solid-body rotation = 2Ω sin φ (Coriolis)
2. Divergence of uniform field = 0
3. Normalization round-trip error < 1e-10

### Physical Tests
1. Vorticity sign correct for cyclonic rotation
2. Shear magnitude matches operational products
3. Conservation of integral quantities after regridding

---

## Traceability

### Data Flow

```
Raw xr.Dataset
    │
    ├── TemporalAligner.align()
    │       └── Common time points
    │
    ├── SpatialRegridder.regrid()
    │       └── Common spatial grid
    │
    ├── PhysicsAwareNormalizer.transform()
    │       └── Normalized values
    │
    └── DerivedFeatureCalculator.compute_all()
            └── Additional derived features
```

### Metadata Propagation
- Original units stored in attributes
- Normalization method recorded
- Feature category tracked

---

## References

1. Jones, P.W. (1999). Conservative remapping schemes for grids in spherical coordinates. MWR.
2. Holton, J.R. (2004). An Introduction to Dynamic Meteorology.
3. Weyn, J.A. et al. (2020). Data-driven global weather prediction using deep CNNs. JAMES.
