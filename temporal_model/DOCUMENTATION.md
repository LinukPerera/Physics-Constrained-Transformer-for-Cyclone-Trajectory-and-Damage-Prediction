# Temporal Model Module Documentation

## Purpose & Scope

This module provides the **physics-constrained Temporal Fusion Transformer** for cyclone trajectory and intensity prediction.

### What This Module Does
- Temporal pattern learning for trajectory prediction
- Uncertainty quantification via quantile outputs
- Physics constraint integration in architecture and loss
- Interpretable attention mechanisms

### What This Module Does NOT Do
- Physical calculations (see `physics_extraction/`)
- Spatial pattern extraction (see `vision_perception/`)
- Final trajectory generation (see `trajectory_prediction/`)

---

## Scientific Context

### Domain
Time series forecasting, physics-informed machine learning, deep learning.

### Architecture

**Base**: Temporal Fusion Transformer (Lim et al., 2021)

**Modifications**:
1. Physics Enrichment Layers
2. Constraint-aware loss functions
3. Output bounds for physical validity

---

## Mathematical Model Class

### Temporal Fusion Transformer

**Type**: Deep learning, encoder-decoder with attention

**Components**:
1. **Variable Selection Networks**: Identify important features
2. **LSTM Encoders**: Capture local temporal patterns
3. **Multi-Head Attention**: Learn long-range dependencies
4. **Gated Linear Units**: Control information flow
5. **Quantile Outputs**: Predict distribution, not point estimate

### Physics Enrichment

Physics-derived features are integrated via gated connections:

```
h_enriched = LayerNorm(h_temporal + GLU([h_temporal, W × f_physics]))
```

This allows the model to use physics guidance while maintaining flexibility.

### Physics-Informed Loss

Total loss combines:
```
L = L_data + λ₁ L_constraints + λ₂ L_conservation
```

where:
- L_data: Quantile loss on predictions
- L_constraints: Penalty for physical violations
- L_conservation: Soft conservation properties

---

## Physical Constraints

### Hard Constraints (enforced in architecture)
- Wind speed ≥ 0 (ReLU activation)
- Pressure ≤ 1013 hPa (clipping)

### Soft Constraints (penalized in loss)
| Constraint | Limit | Justification |
|------------|-------|---------------|
| Translation speed | < 35 m/s | Fastest recorded ~30 m/s |
| Max intensity | < 95 m/s | Theoretical limit |
| Min pressure | > 870 hPa | Record: 870 hPa (Tip) |
| Intensification | < 35 m/s / 6h | Rapid intensification |
| Weakening | < 50 m/s / 6h | Land decay |

---

## Physical Assumptions & Limits

### Model Assumptions
- Past observations are representative of current state
- Physics features capture relevant environmental forcing
- 6-hour time steps resolve important dynamics

### Breakdown Conditions
- Extratropical transition (structure change)
- Binary interaction (multiple storms)
- Rapid intensity change (RI/RW)
- Data gaps in observations

---

## Units & Dimensional Integrity

### Input/Output Units

| Quantity | Unit | Notes |
|----------|------|-------|
| Position | degrees | Latitude, longitude |
| Intensity | m/s | Maximum sustained wind |
| Pressure | hPa | Central pressure |
| Time step | 6 hours | Standard best track |

### Quantile Outputs
Model predicts 10th, 50th, 90th percentiles for uncertainty.

---

## Verification & Validation Strategy

### Training Validation
- Track error at lead times (12h, 24h, 48h, 72h, 96h, 120h)  
- Intensity MAE at same lead times
- Calibration of quantile predictions

### Physical Consistency Checks
- Constraint violation rate < 1%
- Pressure-wind consistency
- Translation speed reasonableness

### Interpretability Analysis
- Attention map inspection
- Variable importance weights
- Physics feature attribution

---

## Traceability

### Data Flow
```
Past observations + Physics features
    │
    └── PhysicsConstrainedTFT
            │
            ├── Variable Selection → Feature importance
            │
            ├── Physics Enrichment → Constraint integration
            │
            ├── Attention → Temporal patterns
            │
            └── Output Heads
                    ├── Position (lat, lon, quantiles)
                    ├── Intensity (wind, quantiles)
                    └── Pressure (p_central, quantiles)
```

### Downstream Use
- `fusion_and_constraints/`: Combine with other branches
- `trajectory_prediction/`: Generate full tracks

---

## References

1. Lim, B. et al. (2021). Temporal Fusion Transformers. IJCAI.
2. Raissi, M. et al. (2019). Physics-informed neural networks. JCP.
3. Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.
