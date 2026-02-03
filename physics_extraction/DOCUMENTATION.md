# Physics Extraction Module Documentation

## Purpose & Scope

This module converts raw observations and model outputs into **physically interpretable quantities**. This is a pure physics module with NO machine learning.

### What This Module Does
- Compute potential intensity (theoretical maximum)
- Compute environmental steering flow
- Compute beta drift velocity
- Compute pressure gradient and Coriolis forces
- Provide physically interpretable features for ML

### What This Module Does NOT Do
- Make machine learning predictions
- Interpolate or regrid data
- Process satellite imagery

---

## Scientific Context

### Domain
Tropical meteorology, atmospheric dynamics, cyclone physics.

### Physical Framework

Tropical cyclone motion is governed by:
```
dX/dt = V_steering + V_beta + V_land + higher_order_terms
```

where:
- **V_steering**: Deep-layer mean environmental wind
- **V_beta**: Coriolis gradient (β-effect) drift
- **V_land**: Land interaction effects

---

## Mathematical Model Class

### Potential Intensity (`atmospheric_dynamics.py`)

**Type**: Thermodynamic, algebraic
**Model**: Emanuel (1986, 1995)

Maximum potential intensity:
```
V_max² = (C_k/C_d) × ε × (h*_s - h_b)
```

where:
- ε = (T_s - T_out)/T_out is Carnot efficiency
- h*_s - h_b is sea surface enthalpy deficit

**Why simpler models are invalid**:
- Empirical pressure-wind relationships lack physical basis
- Statistical models don't capture environmental dependencies

### Environmental Steering (`force_terms.py`)

**Type**: Kinematic, empirical
**Model**: Deep-layer mean flow averaging

Steering flow computed as mass-weighted layer average:
```
V_steer = Σ(w_k × V_k) / Σ(w_k)
```

over annulus 200-500 km from center.

### Beta Drift (`force_terms.py`)

**Type**: Dynamic, approximate
**Model**: Fiorino & Elsberry (1989)

Beta drift arises from:
1. Ventilation of anticyclonic gyres by β
2. Nonlinear vortex-Rossby wave dynamics

Approximate velocity:
```
V_beta ≈ 0.3 × β × R_max² × sin(45°)
```

Typical magnitude: 1-3 m/s (westward and poleward).

---

## Physical Assumptions & Limits

### Potential Intensity
- Axisymmetric storm
- Quasi-steady state
- No environmental shear
- Sufficient ocean heat content

### Steering Flow
- Storm has minimal impact on environment beyond 500 km
- Deep-layer mean captures steering at all levels
- Beta-plane approximation valid

### Beta Drift
- Barotropic vortex approximation
- Beta-plane (df/dy = constant)
- Isolated vortex (no other storms nearby)

### Breakdown Conditions
- Extratropical transition (asymmetric structure)
- Binary interaction (multiple storms)
- Strong shear environments
- Rapidly changing environments

---

## Units & Dimensional Integrity

### Output Units

| Quantity | Unit | Typical Range |
|----------|------|---------------|
| Potential intensity | m/s | 30-85 |
| Steering velocity | m/s | 0-15 |
| Beta drift velocity | m/s | 0-5 |
| Pressure gradient force | m/s² | ±10⁻³ |
| Coriolis acceleration | m/s² | ±10⁻³ |

### Dimensional Checks
- All velocities in m/s
- All accelerations in m/s²
- All temperatures in K
- All pressures in Pa

---

## Verification & Validation Strategy

### Potential Intensity
- Compare against operational PI products (SHIPS/LGEM)
- Verify thermodynamic efficiency ε ∈ [0.3, 0.6]
- Check V_max correlates with SST - T_outflow

### Steering Flow
- Compare against reanalysis deep-layer mean
- Verify annular averaging excludes inner core
- Check layer weights sum to 1.0

### Beta Drift
- Magnitude 1-3 m/s for typical storms
- Direction: westward + poleward (NH) / equatorward (SH)
- Increases with storm size

---

## Traceability

### Data Flow
```
Atmospheric reanalysis (T, q, u, v, p)
    │
    ├── compute_potential_intensity()
    │       └── V_max, P_min, efficiency
    │
    ├── compute_environmental_steering()
    │       └── (u_steer, v_steer)
    │
    └── compute_beta_drift()
            └── (u_beta, v_beta)
                    │
                    └── CycloneForces (complete force budget)
```

### Downstream Use
- `temporal_model/`: Features for TFT
- `fusion_and_constraints/`: Physical bounds for predictions
- `validation/`: Physics consistency checks

---

## References

1. Emanuel, K.A. (1986). An air-sea interaction theory for tropical cyclones. J. Atmos. Sci., 43, 585-604.
2. Emanuel, K.A. (1995). Sensitivity of tropical cyclones to surface exchange coefficients. J. Atmos. Sci., 52, 3969-3976.
3. Fiorino, M. & Elsberry, R.L. (1989). Some aspects of vortex structure related to tropical cyclone motion. J. Atmos. Sci., 46, 975-990.
4. Holland, G.J. (1983). Tropical cyclone motion: Environmental interaction plus a beta effect. J. Atmos. Sci., 40, 328-342.
5. Chan, J.C.L. (2005). The physics of tropical cyclone motion. Annu. Rev. Fluid Mech., 37, 99-128.
