# Appendix A: Supplementary Materials and Code Repository

## A.1 Code Availability

The complete source code for the mathematical models, measurement tools, and figure
generation scripts developed in this work is publicly available in a GitHub repository:

    https://github.com/[yourusername]/continuum-actuator-model

A permanent archived version with a Digital Object Identifier (DOI) is available through Zenodo:

    DOI: 10.5281/zenodo.XXXXXXX

## A.2 Repository Contents

The repository includes the following components:

### A.2.1 Model Implementation (`src/models.py`)

Python implementation of all kinematic models presented in Section 2:

- **Geometric constant curvature model** (Equations 1-2)
  - `geometric_bending_angle()`: Baseline angle prediction
  - `geometric_bending_direction()`: Baseline direction prediction

- **Stiffness-scaled model** (Equation 3)
  - `stiffness_scaled_bending_angle()`: Energy-balance corrected angle

- **Asymmetry-corrected model** (Equations 4-5)
  - `asymmetry_corrected_direction()`: Interaction-weighted direction

- **Physical parameter dataclass** (`ActuatorParameters`)
  - Encapsulates all actuator geometry and material properties
  - Automatically computes derived quantities (EI, k_c, correction factors)

### A.2.2 Measurement Tools (`src/measurement/`)

OpenCV-based interactive tools for extracting bending measurements from experimental images:

- **`angle_measurement.py`**: Bending angle extraction
  - Interactive curve tracing via mouse input
  - Arc-length parameterized resampling for uniform point distribution
  - Central difference tangent computation
  - Automated angle calculation at sampled points

- **`direction_measurement.py`**: Bending direction extraction
  - Three-point angle measurement interface
  - Reference axis overlay for consistent measurements

### A.2.3 Figure Generation (`figures/`)

Scripts to reproduce all publication figures:

- Parity plots comparing measured vs. predicted values
- Error distribution histograms
- Model comparison across actuation types

## A.3 Data Availability

The experimental measurement data (27 actuation configurations) used for model validation
is proprietary and cannot be publicly released. However, the repository includes:

- Complete model implementations that can be applied to new data
- Example usage demonstrating model predictions
- Physical parameters of the experimental prototype (Table 1)

Researchers wishing to reproduce the validation results may contact the author
to discuss data sharing arrangements under appropriate agreements.

## A.4 Usage Example

```python
from src.models import ActuatorParameters, hybrid_model

# Initialize with prototype parameters
params = ActuatorParameters(
    R=2.5,      # mm
    L=40.0,     # mm  
    E=193e3,    # N/mm²
    d_wire=0.5  # mm
)

# Predict bending for unequal dual-tendon actuation
l1, l2, l3 = 2.84, 5.67, 0.0  # mm
alpha, rho = hybrid_model(l1, l2, l3, params)

print(f"Predicted bending angle: {alpha:.2f}°")
print(f"Predicted bending direction: {rho:.2f}°")
```

## A.5 Citation

When using this code, please cite both the paper and the repository:

```bibtex
@article{[yourname]2025continuum,
  title={Modeling Bending Behavior in Cable-Driven Continuum Spring Actuators},
  author={[Your Name]},
  journal={[Journal]},
  year={2025}
}

@software{[yourname]2025repo,
  author={[Your Name]},
  title={Continuum Actuator Model - Code Repository},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXXX}
}
```
