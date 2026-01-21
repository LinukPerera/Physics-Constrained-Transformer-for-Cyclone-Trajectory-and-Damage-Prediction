# Physics Constrained Transformer for Cyclone Trajectory and Damage Prediction
This framework fuses satellite imagery, atmospheric data, and terrain information with a physics-constrained transformer to predict cyclone trajectories. By integrating hazard models and scenario simulations, it delivers interpretable, physically grounded forecasts of cyclone impacts.

![Cyclone Transformer](images/cyclone_transformer.png)

---

### **Methodology**

#### **1. Input Data Acquisition**

The model leverages multiple heterogeneous data sources, capturing the complex spatiotemporal dynamics of cyclones:

* **Satellite Imagery & Remote Sensing Data**: Provides high-resolution visual and infrared snapshots of cloud structures, storm systems, and ocean surface conditions.
* **Atmospheric Reanalysis Data**: Historical gridded datasets including wind vectors, pressure, temperature, and humidity fields, providing a comprehensive representation of atmospheric conditions.
* **Terrain Digital Elevation Model (DEM)**: High-resolution terrain data capturing topographic variations that influence cyclone behavior, rainfall runoff, and potential landslide risk.

> **Note:** These datasets are processed as **four parallel layers**, allowing simultaneous extraction of spatial and temporal correlations.

---

#### **2. Data Preprocessing**

Raw input data undergoes preprocessing to ensure compatibility and quality for downstream modeling:

* **Normalization & Scaling**: Each input modality is normalized according to its physical units to reduce bias.
* **Resampling & Alignment**: All inputs are temporally and spatially aligned to the same grid resolution and timestamps.
* **Feature Augmentation**: Derived features like vorticity, divergence, and terrain slope are calculated to enrich model input.

---

#### **3. Vision Perceptron**

To capture spatial patterns from the inputs, a hybrid vision transformer architecture is applied in **two parallel branches**:

1. **EfficientNet (EFFNet) Backbone**: Extracts hierarchical features from satellite imagery, capturing mesoscale cloud structures and cyclone eye characteristics.
2. **Recurrent All-Pairs Field Transformer (RAFT)**: Processes sequential data to model spatiotemporal interactions in atmospheric fields, effectively encoding motion vectors and evolving storm dynamics.

> **Output**: High-dimensional feature embeddings representing both the visual and physical evolution of cyclones.

---

#### **4. Physics Calculation Extraction**

Prior to trajectory modeling, physically meaningful features are explicitly computed from the input data:

* Wind field gradients, pressure tendencies, and vorticity.
* Momentum flux, energy dissipation, and terrain-influenced runoff factors.

This step ensures the model retains interpretable, physics-based information alongside learned representations.

---

#### **5. Temporal Fusion Transformer (Physics-Constrained)**

The core of the model is a **Temporal Fusion Transformer (TFT)** augmented with **physics constraints**, designed to forecast cyclone trajectories:

* **Parallel Processing**:

  1. **Physics Model Loss**: Ensures predictions adhere to conservation laws (mass, momentum, energy).
  2. **Physics-Informed Mini Models**: Small surrogate models that provide upper/lower bounds for predictions (e.g., maximum wind speed or storm surge limits).
  3. **Raw Atmospheric Physics Calculations**: Direct integration of observed physical quantities to guide model attention.

* **Fusion Layer**: Outputs of the three parallel branches are fused while respecting physical constraints.

* **Limit Enforcement**: The fused representation is constrained to stay within physically plausible boundaries defined by mini-models, reducing unphysical predictions.

---

#### **6. Cyclone Path Prediction**

The transformer output is decoded into predicted cyclone trajectories, providing:

* **Position (latitude, longitude) at future time steps**
* **Intensity metrics**: Maximum sustained winds, central pressure estimates
* **Trajectory uncertainty bounds** derived from physics-informed constraints.

---

#### **7. Hazard Modeling**

To translate trajectory predictions into real-world risk, the predicted path is combined with hazard models:

* **Wind Damage Models** (e.g., Holland wind model)
* **Flood Risk Models** (incorporating soil infiltration, slope, and drainage)
* **Landslide Susceptibility Models** (using terrain DEM, soil cohesion, and rainfall intensity)

This step allows assessment of pre-existing vulnerabilities, enhancing impact estimation.

---

#### **8. Scenario Simulation**

Finally, the integrated framework generates multiple potential cyclone impact scenarios:

* **Flooding extents**
* **Landslide occurrences**
* **Overall structural and population exposure**

By combining **trajectory forecasts**, **hazard maps**, and **physics constraints**, the system outputs probabilistic risk assessments for emergency planning and mitigation.

---

### **Summary of the Workflow**

```
Inputs (Satellite + Atmospheric Reanalysis + Terrain DEM) 
        ↓
Preprocessing (Normalization, Alignment, Feature Engineering)
        ↓
Vision Perceptron (EfficientNet + RAFT)
        ↓
Physics Calculation Extraction
        ↓
Temporal Fusion Transformer (Physics-Constrained)
    ├─ Physics Model Loss
    ├─ Physics-Informed Mini Models
    └─ Raw Atmospheric Physics
        ↓
Fuse Outputs → Enforce Physics Limits
        ↓
Predict Cyclone Trajectory
        ↓
Hazard Modeling (Wind, Flood, Landslide)
        ↓
Scenario Simulation (Impact Assessment)
```

---

This framework emphasizes **physics-informed deep learning**, combining learned spatiotemporal patterns with explicit physical constraints to produce **interpretable, robust, and actionable cyclone forecasts**.

---
