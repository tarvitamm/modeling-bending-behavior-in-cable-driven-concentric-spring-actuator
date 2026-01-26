"""
Hybrid Kinematic Models for Cable-Driven Continuum Spring Actuators
====================================================================

This module implements the mathematical models for predicting bending angle (α)
and bending direction (ρ) in cable-driven continuum spring actuators.

Models implemented:
    1. Geometric (constant curvature) - baseline model
    2. Stiffness-scaled - energy balance correction for bending angle
    3. Asymmetry-corrected - stiffness-displacement interaction for direction
    4. Combined hybrid model - integrates both corrections
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================

@dataclass
class ActuatorParameters:
    """
    Physical parameters of the cable-driven continuum spring actuator.
    
    Attributes:
        R: Spring outer radius (mm)
        L: Active spring length (mm)
        E: Young's modulus of spring material (N/mm²)
        d_wire: Spring wire diameter (mm)
        d_tendon: Tendon diameter (mm)
        y_tendon: Young's modulus of tendon material (N/mm²)
        L_tendon: Free tendon length (mm)
        theta_1, theta_2, theta_3: Tendon angular positions (degrees)
    """
    R: float = 2.5          # mm
    L: float = 40.0         # mm
    E: float = 193e3        # N/mm² (stainless steel)
    d_wire: float = 0.5     # mm
    d_tendon: float = 0.28  # mm
    y_tendon: float = 200e3 # N/mm²
    L_tendon: float = 180.0 # mm
    theta_1: float = 30.0   # degrees
    theta_2: float = 150.0  # degrees
    theta_3: float = 270.0  # degrees
    
    @property
    def I(self) -> float:
        """Area moment of inertia of spring wire (mm⁴)"""
        return np.pi * self.d_wire**4 / 64
    
    @property
    def A_tendon(self) -> float:
        """Cross-sectional area of tendon (mm²)"""
        return np.pi * (self.d_tendon / 2)**2
    
    @property
    def k_c(self) -> float:
        """Tendon stiffness (N/mm)"""
        return self.y_tendon * self.A_tendon / self.L_tendon
    
    @property
    def EI(self) -> float:
        """Bending rigidity (N·mm²)"""
        return self.E * self.I
    
    @property
    def stiffness_factor(self) -> float:
        """
        Stiffness correction factor for bending angle.
        
        Derived from energy balance: W_tendon = W_bending
        α_corrected = α_geometric / stiffness_factor
        """
        return 1 + self.EI / (self.k_c * self.L)
    
    @property
    def interaction_factor(self) -> float:
        """
        Stiffness-displacement interaction factor for direction correction.
        
        Quantifies how tendon effectiveness scales with displacement
        when working against spring bending resistance.
        """
        return self.k_c * self.L / self.EI


# Default parameters (can be customized)
DEFAULT_PARAMS = ActuatorParameters()


# =============================================================================
# GEOMETRIC MODEL (Constant Curvature Baseline)
# =============================================================================

def geometric_bending_angle(l1: float, l2: float, l3: float, 
                            params: ActuatorParameters = DEFAULT_PARAMS) -> float:
    """
    Compute bending angle using geometric constant curvature model.
    
    This baseline model assumes the actuator bends uniformly along its length
    with no internal resistance. Valid for ideal conditions only.
    
    Args:
        l1, l2, l3: Tendon displacements (mm)
        params: Actuator physical parameters
        
    Returns:
        Bending angle α in degrees
        
    Mathematical formulation:
        α = (1/√6R) × √[(l₁-l₂)² + (l₂-l₃)² + (l₃-l₁)²]
        
    Reference:
        Webster & Jones (2010), "Design and Kinematic Modeling of 
        Constant Curvature Continuum Robots: A Review"
    """
    delta_sq = (l1 - l2)**2 + (l2 - l3)**2 + (l3 - l1)**2
    delta = np.sqrt(delta_sq)
    alpha_rad = delta / (np.sqrt(6.0) * params.R)
    return np.degrees(alpha_rad)


def geometric_bending_direction(l1: float, l2: float, l3: float,
                                params: ActuatorParameters = DEFAULT_PARAMS) -> float:
    """
    Compute bending direction using geometric model.
    
    Determines the angular orientation of tip displacement based on
    relative tendon length changes and their angular positions.
    
    Args:
        l1, l2, l3: Tendon displacements (mm)
        params: Actuator physical parameters
        
    Returns:
        Bending direction ρ in degrees [0, 360)
        
    Mathematical formulation:
        ρ = arctan2(√3(l₂-l₃), 2l₁-l₂-l₃) + θ₁
    """
    numerator = np.sqrt(3.0) * (l2 - l3)
    denominator = 2.0 * l1 - l2 - l3
    
    # Handle degenerate case (no actuation)
    if abs(denominator) < 1e-10 and abs(numerator) < 1e-10:
        return params.theta_1
    
    rho = np.degrees(np.arctan2(numerator, denominator)) + params.theta_1
    
    # Normalize to [0, 360)
    rho = rho % 360
    return rho


# =============================================================================
# STIFFNESS-SCALED MODEL (Energy Balance Correction)
# =============================================================================

def stiffness_scaled_bending_angle(l1: float, l2: float, l3: float,
                                   params: ActuatorParameters = DEFAULT_PARAMS) -> float:
    """
    Compute bending angle with stiffness correction.
    
    This model incorporates the mechanical stiffness of both the spring
    backbone and tendons through an energy balance derivation, providing
    physically-grounded predictions without empirical tuning.
    
    Args:
        l1, l2, l3: Tendon displacements (mm)
        params: Actuator physical parameters
        
    Returns:
        Corrected bending angle α in degrees
        
    Mathematical formulation:
        α_corrected = α_geometric / (1 + EI/(k_c·L))
        
    Derivation:
        Energy stored in tendons: W_c = ½k_c·l²
        Energy stored in bending: W_b = ½(EI/L)·α²
        Equating and solving: α = α_geo / stiffness_factor
        
    Reference:
        Li & Rahn (2002), "Design of Continuous Backbone, 
        Cable-Driven Robots"
    """
    alpha_geo = geometric_bending_angle(l1, l2, l3, params)
    return alpha_geo / params.stiffness_factor


# =============================================================================
# ASYMMETRY-CORRECTED MODEL (Direction Correction)
# =============================================================================

def asymmetry_corrected_direction(l1: float, l2: float, l3: float,
                                  params: ActuatorParameters = DEFAULT_PARAMS) -> float:
    """
    Compute bending direction with asymmetry correction.
    
    This model accounts for the stiffness-displacement interaction effect,
    where tendons with larger displacements overcome spring resistance
    more effectively, gaining disproportionate influence on direction.
    
    Args:
        l1, l2, l3: Tendon displacements (mm)
        params: Actuator physical parameters
        
    Returns:
        Corrected bending direction ρ in degrees [0, 360)
        
    Mathematical formulation:
        Wᵢ = lᵢ × (1 + (k_c·L/EI) × lᵢ)
        ρ = arctan2(Σ Wᵢ·sin(θᵢ), Σ Wᵢ·cos(θᵢ))
        
    Physical interpretation:
        The weight function combines linear displacement contribution
        with a quadratic correction term that amplifies the influence
        of more displaced tendons.
    """
    # Convert angles to radians
    t1 = np.radians(params.theta_1)
    t2 = np.radians(params.theta_2)
    t3 = np.radians(params.theta_3)
    
    # Compute effective weights (only positive displacements contribute)
    k = params.interaction_factor
    W1 = max(0, l1) * (1 + max(0, l1) * k)
    W2 = max(0, l2) * (1 + max(0, l2) * k)
    W3 = max(0, l3) * (1 + max(0, l3) * k)
    
    # Vector summation
    sum_sin = W1 * np.sin(t1) + W2 * np.sin(t2) + W3 * np.sin(t3)
    sum_cos = W1 * np.cos(t1) + W2 * np.cos(t2) + W3 * np.cos(t3)
    
    # Handle degenerate case
    if abs(sum_cos) < 1e-10 and abs(sum_sin) < 1e-10:
        return params.theta_1
    
    rho = np.degrees(np.arctan2(sum_sin, sum_cos))
    
    # Normalize to [0, 360)
    if rho < 0:
        rho += 360
    return rho


# =============================================================================
# COMBINED HYBRID MODEL
# =============================================================================

def hybrid_model(l1: float, l2: float, l3: float,
                 params: ActuatorParameters = DEFAULT_PARAMS) -> Tuple[float, float]:
    """
    Compute both bending angle and direction using the full hybrid model.
    
    Combines stiffness-scaled angle prediction with asymmetry-corrected
    direction prediction for comprehensive kinematic modeling.
    
    Args:
        l1, l2, l3: Tendon displacements (mm)
        params: Actuator physical parameters
        
    Returns:
        Tuple of (α, ρ) in degrees
    """
    alpha = stiffness_scaled_bending_angle(l1, l2, l3, params)
    rho = asymmetry_corrected_direction(l1, l2, l3, params)
    return alpha, rho


# =============================================================================
# ERROR METRICS
# =============================================================================

def compute_rmse(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return np.sqrt(np.mean((predicted - measured)**2))


def compute_mae(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(predicted - measured))


def compute_r_squared(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Compute coefficient of determination (R²)."""
    ss_res = np.sum((measured - predicted)**2)
    ss_tot = np.sum((measured - np.mean(measured))**2)
    if ss_tot < 1e-10:
        return 0.0
    return 1 - ss_res / ss_tot


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize with default parameters
    params = ActuatorParameters()
    
    print("=" * 60)
    print("Cable-Driven Continuum Spring Actuator Model")
    print("=" * 60)
    print(f"\nPhysical Parameters:")
    print(f"  Spring radius (R):     {params.R} mm")
    print(f"  Active length (L):     {params.L} mm")
    print(f"  Bending rigidity (EI): {params.EI:.2f} N·mm²")
    print(f"  Tendon stiffness (kc): {params.k_c:.2f} N/mm")
    print(f"\nDerived Factors:")
    print(f"  Stiffness factor:      {params.stiffness_factor:.4f}")
    print(f"  Interaction factor:    {params.interaction_factor:.4f}")
    
    # Example: Single tendon actuation
    print(f"\n{'='*60}")
    print("Example: Single tendon actuation (l1=5mm, l2=0, l3=0)")
    print("=" * 60)
    
    l1, l2, l3 = 5.0, 0.0, 0.0
    
    alpha_geo = geometric_bending_angle(l1, l2, l3, params)
    alpha_stiff = stiffness_scaled_bending_angle(l1, l2, l3, params)
    rho_geo = geometric_bending_direction(l1, l2, l3, params)
    rho_asym = asymmetry_corrected_direction(l1, l2, l3, params)
    
    print(f"\nBending Angle (α):")
    print(f"  Geometric model:       {alpha_geo:.2f}°")
    print(f"  Stiffness-scaled:      {alpha_stiff:.2f}°")
    print(f"  Improvement:           {(1 - alpha_stiff/alpha_geo)*100:.1f}% reduction")
    
    print(f"\nBending Direction (ρ):")
    print(f"  Geometric model:       {rho_geo:.2f}°")
    print(f"  Asymmetry-corrected:   {rho_asym:.2f}°")
    
    # Example: Unequal dual tendon actuation
    print(f"\n{'='*60}")
    print("Example: Unequal dual tendon (l1=2mm, l2=6mm, l3=0)")
    print("=" * 60)
    
    l1, l2, l3 = 2.0, 6.0, 0.0
    
    alpha_geo = geometric_bending_angle(l1, l2, l3, params)
    alpha_stiff = stiffness_scaled_bending_angle(l1, l2, l3, params)
    rho_geo = geometric_bending_direction(l1, l2, l3, params)
    rho_asym = asymmetry_corrected_direction(l1, l2, l3, params)
    
    print(f"\nBending Angle (α):")
    print(f"  Geometric model:       {alpha_geo:.2f}°")
    print(f"  Stiffness-scaled:      {alpha_stiff:.2f}°")
    
    print(f"\nBending Direction (ρ):")
    print(f"  Geometric model:       {rho_geo:.2f}°")
    print(f"  Asymmetry-corrected:   {rho_asym:.2f}°")
    print(f"  Shift toward l2:       {rho_asym - rho_geo:.2f}°")
