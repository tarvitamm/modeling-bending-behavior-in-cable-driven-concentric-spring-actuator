# Modeling bending behavior in cable-driven concentric spring actuator
## Overview

This repository contains the implementation code and analysis tools for the paper:

> **"Modeling Bending Behavior in Cable-Driven Continuum Spring Actuators"**
> 
> Tarvi Tamm, Brayan Fimberg, Ritesh Soni, and Alvo Aabloo, 2026

The work presents a hybrid kinematic model that combines constant curvature geometry with physically-derived stiffness corrections for predicting bending angle and direction in spring-backbone continuum actuators.

## Key Results

| Model | Bending Angle RMSE | Bending Direction RMSE |
|-------|-------------------|----------------------|
| Geometric (baseline) | 18.41° | 11.10° |
| **Hybrid (proposed)** | **6.44°** | **7.75°** |
| Improvement | **65.0%** | **30.2%** |

## Repository Structure

```
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── src/
│   ├── models.py            # Core model implementations
│   └── measurement/
│       ├── bending_angle.py      # Bending angle extraction tool
│       └── bending_direction.py  # Bending direction extraction tool
├── figures/
│   ├── sideview_frontview.py       # Side and front view visualization
│   ├── visual_model_comparison.py  # Model comparison visualization
│   └── output/              # Generated figures (PNG/PDF)
└── docs/
    └── parameters.md        # Physical parameter documentation
```


## Physical Parameters

The default parameters correspond to the experimental prototype:

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Spring outer radius | R | 2.5 | mm |
| Active length | L | 40 | mm |
| Spring Young's modulus | E | 193 | GPa |
| Wire diameter | d | 0.5 | mm |
| Tendon diameter | d_t | 0.28 | mm |
| Tendon Young's modulus | y | 200 | GPa |
| Tendon free length | L_s | 180 | mm |

### Derived Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Area moment of inertia | I | 0.00307 | mm⁴ |
| Bending rigidity | EI | 592.12 | N·mm² |
| Tendon stiffness | k_c | 68.42 | N/mm |
| Stiffness factor | 1 + EI/(k_c·L) | 1.216 | - |
| Interaction factor | k_c·L/EI | 4.62 | - |

## Model Equations

### Geometric Model (Baseline)

**Bending angle:**
```
α = (1/√6R) × √[(l₁-l₂)² + (l₂-l₃)² + (l₃-l₁)²]
```

**Bending direction:**
```
ρ = arctan2(√3(l₂-l₃), 2l₁-l₂-l₃) + θ₁
```

### Stiffness-Scaled Model

**Corrected bending angle:**
```
α_corrected = α_geometric / (1 + EI/(k_c·L))
```

### Asymmetry-Corrected Model

**Effective weights:**
```
Wᵢ = lᵢ × (1 + (k_c·L/EI) × lᵢ)
```

**Corrected direction:**
```
ρ = arctan2(Σ Wᵢ·sin(θᵢ), Σ Wᵢ·cos(θᵢ))
```


## Contact
Project Link: https://github.com/tarvitamm/modeling-bending-behavior-in-cable-driven-concentric-spring-actuator
