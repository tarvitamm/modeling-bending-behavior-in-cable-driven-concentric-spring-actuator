"""
Figure for Section 4.3.2: Visual Model Comparison
=================================================
Shows measured vs predicted for both α and ρ across all models
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
R = 2.5  # mm
L = 40.0  # mm
E = 193e3  # N/mm²
I = np.pi * 0.5**4 / 64  # mm⁴
d_tendon = 0.28  # mm
y = 200e3  # N/mm²
A_s = np.pi * (d_tendon/2)**2
L_s = 180.0  # mm
k_c = y * A_s / L_s
EI = E * I
STIFFNESS_FACTOR = 1 + EI / (k_c * L)
INTERACTION_FACTOR = k_c * L / EI

# Tendon angles
THETA_1, THETA_2, THETA_3 = 30, 150, 270
t1, t2, t3 = np.radians(THETA_1), np.radians(THETA_2), np.radians(THETA_3)

# Model functions
def geometric_angle(l1, l2, l3):
    Delta = np.sqrt((l1 - l2)**2 + (l2 - l3)**2 + (l3 - l1)**2)
    return np.degrees(Delta / (np.sqrt(6.0) * R))

def stiffness_angle(l1, l2, l3):
    return geometric_angle(l1, l2, l3) / STIFFNESS_FACTOR

def geometric_direction(l1, l2, l3):
    num = np.sqrt(3.0) * (l2 - l3)
    den = 2.0 * l1 - l2 - l3
    if abs(den) < 1e-10 and abs(num) < 1e-10:
        return THETA_1
    rho = np.degrees(np.arctan2(num, den)) + THETA_1
    if rho < 0: rho += 360
    if rho >= 360: rho -= 360
    return rho

def asymmetry_direction(l1, l2, l3):
    W1 = max(0, l1) * (1 + max(0, l1) * INTERACTION_FACTOR)
    W2 = max(0, l2) * (1 + max(0, l2) * INTERACTION_FACTOR)
    W3 = max(0, l3) * (1 + max(0, l3) * INTERACTION_FACTOR)
    sum_sin = W1*np.sin(t1) + W2*np.sin(t2) + W3*np.sin(t3)
    sum_cos = W1*np.cos(t1) + W2*np.cos(t2) + W3*np.cos(t3)
    if abs(sum_cos) < 1e-10 and abs(sum_sin) < 1e-10:
        return THETA_1
    rho = np.degrees(np.arctan2(sum_sin, sum_cos))
    if rho < 0: rho += 360
    return rho

# Experimental data: (l1, l2, l3, measured_α, measured_ρ)
single_data = [
    (1.42, 0, 0, 23.246, 42.03),
    (2.84, 0, 0, 31.974, 44.00),
    (4.25, 0, 0, 45.982, 42.80),
    (5.67, 0, 0, 54.826, 43.30),
    (7.09, 0, 0, 66.748, 44.69),
    (8.51, 0, 0, 80.870, 46.08),
]

equal_data = [
    (1.42, 1.42, 0, 27.576, 89.60),
    (2.84, 2.84, 0, 36.560, 90.19),
    (4.25, 4.25, 0, 49.494, 89.20),
    (5.67, 5.67, 0, 60.652, 86.50),
    (7.09, 7.09, 0, 73.868, 85.33),
    (8.51, 8.51, 0, 85.950, 86.79),
]

unequal_data = [
    (1.42, 2.84, 0, 30.682, 129.65),
    (1.42, 4.25, 0, 37.302, 135.44),
    (1.42, 5.67, 0, 47.276, 138.63),
    (1.42, 7.09, 0, 58.704, 137.82),
    (1.42, 8.51, 0, 74.558, 140.22),
    (2.84, 4.25, 0, 42.534, 125.32),
    (2.84, 5.67, 0, 47.864, 135.34),
    (2.84, 7.09, 0, 59.056, 136.51),
    (2.84, 8.51, 0, 80.786, 138.98),
    (4.25, 5.67, 0, 55.670, 120.53),
    (4.25, 7.09, 0, 63.122, 130.44),
    (4.25, 8.51, 0, 78.412, 135.41),
    (5.67, 7.09, 0, 71.760, 111.44),
    (5.67, 8.51, 0, 80.004, 127.81),
    (7.09, 8.51, 0, 80.560, 107.64),
]

all_data = single_data + equal_data + unequal_data

# Set up publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Colors
c_single = '#E94F37'    # Red for single
c_equal = '#4CAF50'     # Green for equal
c_unequal = '#2E86AB'   # Blue for unequal

# ============================================================================
# FIGURE: Measured vs Predicted Parity Plots (2x2)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

# Compute all predictions
meas_a = [d[3] for d in all_data]
meas_r = [d[4] for d in all_data]
geo_a = [geometric_angle(d[0], d[1], d[2]) for d in all_data]
stiff_a = [stiffness_angle(d[0], d[1], d[2]) for d in all_data]
geo_r = [geometric_direction(d[0], d[1], d[2]) for d in all_data]
asym_r = [asymmetry_direction(d[0], d[1], d[2]) for d in all_data]

# Separate by type
single_idx = list(range(6))
equal_idx = list(range(6, 12))
unequal_idx = list(range(12, 27))

# --- Top Left: Geometric α ---
ax = axes[0, 0]
ax.scatter([geo_a[i] for i in single_idx], [meas_a[i] for i in single_idx],
           color=c_single, s=50, label='Single-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([geo_a[i] for i in equal_idx], [meas_a[i] for i in equal_idx],
           color=c_equal, s=50, label='Equal dual-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([geo_a[i] for i in unequal_idx], [meas_a[i] for i in unequal_idx],
           color=c_unequal, s=50, label='Unequal dual-tendon', edgecolors='black', linewidth=0.5)
ax.plot([0, 120], [0, 120], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Predicted angle (°)')
ax.set_ylabel('Measured angle (°)')
ax.set_title('(a) Geometric model: bending angle')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(0, 120)
ax.set_ylim(0, 95)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.text(100, 10, f'RMSE = {18.41:.1f}°', fontsize=10, ha='right')

# --- Top Right: Stiffness-scaled α ---
ax = axes[0, 1]
ax.scatter([stiff_a[i] for i in single_idx], [meas_a[i] for i in single_idx],
           color=c_single, s=50, label='Single-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([stiff_a[i] for i in equal_idx], [meas_a[i] for i in equal_idx],
           color=c_equal, s=50, label='Equal dual-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([stiff_a[i] for i in unequal_idx], [meas_a[i] for i in unequal_idx],
           color=c_unequal, s=50, label='Unequal dual-tendon', edgecolors='black', linewidth=0.5)
ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Predicted angle (°)')
ax.set_ylabel('Measured angle (°)')
ax.set_title('(b) Stiffness-scaled model: bending angle')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(0, 100)
ax.set_ylim(0, 95)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.text(85, 10, f'RMSE = {6.44:.1f}°', fontsize=10, ha='right')

# --- Bottom Left: Geometric ρ ---
ax = axes[1, 0]
ax.scatter([geo_r[i] for i in single_idx], [meas_r[i] for i in single_idx],
           color=c_single, s=50, label='Single-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([geo_r[i] for i in equal_idx], [meas_r[i] for i in equal_idx],
           color=c_equal, s=50, label='Equal dual-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([geo_r[i] for i in unequal_idx], [meas_r[i] for i in unequal_idx],
           color=c_unequal, s=50, label='Unequal dual-tendon', edgecolors='black', linewidth=0.5)
ax.plot([20, 150], [20, 150], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Predicted direction (°)')
ax.set_ylabel('Measured direction (°)')
ax.set_title('(c) Geometric model: bending direction')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(20, 150)
ax.set_ylim(35, 145)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.text(140, 42, f'RMSE = {11.10:.1f}°', fontsize=10, ha='right')

# --- Bottom Right: Asymmetry-corrected ρ ---
ax = axes[1, 1]
ax.scatter([asym_r[i] for i in single_idx], [meas_r[i] for i in single_idx],
           color=c_single, s=50, label='Single-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([asym_r[i] for i in equal_idx], [meas_r[i] for i in equal_idx],
           color=c_equal, s=50, label='Equal dual-tendon', edgecolors='black', linewidth=0.5)
ax.scatter([asym_r[i] for i in unequal_idx], [meas_r[i] for i in unequal_idx],
           color=c_unequal, s=50, label='Unequal dual-tendon', edgecolors='black', linewidth=0.5)
ax.plot([20, 160], [20, 160], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Predicted direction (°)')
ax.set_ylabel('Measured direction (°)')
ax.set_title('(d) Asymmetry-corrected model: bending direction')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_xlim(20, 160)
ax.set_ylim(35, 145)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.text(150, 42, f'RMSE = {7.75:.1f}°', fontsize=10, ha='right')



# ============================================================================
# Calculate R² values
# ============================================================================
def r_squared(predicted, measured):
    pred = np.array(predicted)
    meas = np.array(measured)
    ss_res = np.sum((meas - pred)**2)
    ss_tot = np.sum((meas - np.mean(meas))**2)
    return 1 - (ss_res / ss_tot)

print("\n" + "="*60)
print("R² VALUES FOR MODEL COMPARISONS")
print("="*60)
print(f"Bending Angle (α):")
print(f"  Geometric model:       R² = {r_squared(geo_a, meas_a):.3f}")
print(f"  Stiffness-scaled:      R² = {r_squared(stiff_a, meas_a):.3f}")
print(f"\nBending Direction (ρ):")
print(f"  Geometric model:       R² = {r_squared(geo_r, meas_r):.3f}")
print(f"  Asymmetry-corrected:   R² = {r_squared(asym_r, meas_r):.3f}")

# Unequal only for direction
geo_r_uneq = [geo_r[i] for i in unequal_idx]
asym_r_uneq = [asym_r[i] for i in unequal_idx]
meas_r_uneq = [meas_r[i] for i in unequal_idx]
print(f"\nBending Direction (ρ) - Unequal cases only:")
print(f"  Geometric model:       R² = {r_squared(geo_r_uneq, meas_r_uneq):.3f}")
print(f"  Asymmetry-corrected:   R² = {r_squared(asym_r_uneq, meas_r_uneq):.3f}")
