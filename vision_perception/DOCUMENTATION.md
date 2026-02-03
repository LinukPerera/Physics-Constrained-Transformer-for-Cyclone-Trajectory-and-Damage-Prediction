# Vision Perception Module Documentation

## Purpose & Scope

This module provides **visual feature extraction** from satellite imagery using deep learning. It serves as a pattern extraction system, NOT a physics calculator.

### What This Module Does
- Extract spatial patterns from satellite imagery (EfficientNet)
- Estimate motion between consecutive frames (RAFT optical flow)
- Convert pixel motion to physical displacement

### What This Module Does NOT Do
- Provide physical interpretations of visual features
- Replace atmospheric observations
- Compute atmospheric dynamics (see `physics_extraction/`)

---

## Scientific Context

### Domain
Computer vision, optical flow estimation, remote sensing.

### CRITICAL DISTINCTION

| Quantity | Definition | Units |
|----------|------------|-------|
| **Image motion** | Pixel displacement between frames | pixels/frame |
| **Atmospheric motion** | Wind velocity at cloud level | m/s |

**These are NOT the same.** The conversion requires:
1. Satellite viewing geometry (pixel scale varies with view angle)
2. Parallax correction (clouds at different heights appear to move differently)
3. Temporal calibration (frame interval to seconds)
4. Height assignment (which pressure level does the motion represent?)

### Model Family

**EfficientNet Encoder**:
- Convolutional neural network for spatial pattern extraction
- NO physical meaning assumed from features
- Treats satellite imagery as texture/pattern detection problem

**RAFT Motion Estimator**:
- Recurrent optical flow estimation
- Outputs pixel-space motion, not physical velocity
- Must be converted before physics use

---

## Mathematical Model Class

### Feature Extraction
**Type**: Learned visual representations
**Model**: Deep convolutional network (EfficientNet B4)

**Explicit limitation**: Features are learned patterns optimized for discrimination, not physical quantities. The 256-dimensional feature vector has NO defined physical interpretation.

### Motion Estimation
**Type**: Dense optical flow
**Model**: RAFT (Recurrent All-Pairs Field Transforms)

**Conversion from pixels to physics**:
```
velocity_m_s = pixel_displacement × pixel_scale_m / frame_interval_s
```

where `pixel_scale_m` accounts for:
- Latitude-dependent distortion
- Satellite view angle
- Parallax from cloud height

---

## Physical Assumptions & Limits

### EfficientNet Encoder
- Image statistics similar to training distribution
- Cloud structures are resolved at input resolution
- Multi-channel (IR, VIS, WV) combinations are meaningful

### RAFT Motion
- Brightness constancy between frames (approximately holds for IR)
- Motion is smooth (valid for synoptic-scale features)
- No occlusion handling (cloud overlap ignored)

### Motion to Physical Conversion
- Geostationary satellite geometry
- Known pixel scale at nadir
- Cloud heights available for parallax correction
- Temporal sampling resolves motion (Nyquist criteria)

### Breakdown Conditions
- Very high latitudes (>60°) where satellite view angle is extreme
- Rapidly evolving convection where brightness constancy fails
- Mixed-layer clouds where height assignment is ambiguous

---

## Units & Dimensional Integrity

### Feature Outputs
| Output | Type | Shape | Meaning |
|--------|------|-------|---------|
| Features | float32 | (B, C, H, W) | Visual patterns (NO physical units) |
| Attention | float32 | (B, H, W) | Salience [0, 1] |

### Motion Outputs
| Output | Unit | Notes |
|--------|------|-------|
| PixelMotion.u_pixels | pixels | Horizontal displacement |
| PixelMotion.v_pixels | pixels | Vertical displacement |
| PhysicalMotion.u_ms | m/s | Eastward velocity |
| PhysicalMotion.v_ms | m/s | Northward velocity |
| PhysicalMotion.uncertainty_ms | m/s | Estimated error |

---

## Verification & Validation Strategy

### EfficientNet Validation
- Attention maps should highlight cyclone eye and outer bands
- Feature similarity high for same storm, low across storms
- No quantitative physics validation (by design)

### RAFT Motion Validation
1. Compare against Atmospheric Motion Vectors (AMV) products
2. Synthetic displacement tests with known translation
3. Cross-validation against reanalysis winds

### Expected Errors
- Optical flow accuracy: ~1-2 pixels RMS
- Physical velocity error: 2-5 m/s (depending on geometry)
- Height assignment error: largest uncertainty source

---

## Traceability

### Data Flow
```
Satellite imagery (C, H, W)
    │
    ├── EfficientNetEncoder
    │       └── Feature pyramid (visual patterns)
    │
    └── RAFTMotionEstimator
            └── PixelMotion
                    │
                    └── MotionToPhysicalConverter
                            └── PhysicalMotion (m/s, with uncertainty)
```

### Validation Against Atmospheric Products
Physical motion estimates should be compared to:
- ERA5/GFS analysis winds at appropriate level
- Operational AMV products (GOES, Himawari)
- Radiosonde observations near storm

---

## References

1. Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking model scaling. ICML.
2. Teed, Z. & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms. ECCV.
3. Velden, C.S. et al. (2005). Recent innovations in deriving AMVs. BAMS.
4. Bresky, W.C. et al. (2012). New methods toward minimizing AMV height assignment errors. JAMC.
