
"""
Two-Panel Figure - Side View (α) and Top View (ρ)
=================================================
All objects centered/starting at origin (0,0)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle

# ============================================================================
# PARAMETERS - ADJUST THESE
# ============================================================================
L = 4.0              # Spring length
alpha_deg = 35       # Bending angle (degrees)
rho_deg = 40         # Bending direction (degrees)

# Base dimensions
base_width = 1.6
base_height = 0.5

# Arc sizes
arc_radius_alpha = 1.3
arc_radius_rho = 1.0

# Origin points (SEPARATE for each view)
origin_x_side = -1
origin_y_side = 1

origin_x_top = 0      # Moved to center
origin_y_top = 0      # Moved to bottom

# ============================================================================
# CALCULATIONS
# ============================================================================
alpha_rad = np.radians(alpha_deg)
rho_rad = np.radians(rho_deg)
R_curve = L / alpha_rad

n_points = 100
theta_spring = np.linspace(0, alpha_rad, n_points)

# Side view coordinates
x_side = origin_x_side + R_curve * np.sin(theta_spring)
z_side = origin_y_side + R_curve * (1 - np.cos(theta_spring))

x_tip_side = x_side[-1]
z_tip_side = z_side[-1]

# Top view coordinates (using separate origin)
x_top = origin_x_top + R_curve * np.sin(theta_spring) * np.cos(rho_rad)
y_top = origin_y_top + R_curve * np.sin(theta_spring) * np.sin(rho_rad)

x_tip_top = x_top[-1]
y_tip_top = y_top[-1]

# ============================================================================
# FIGURE: Two panels
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Set a single main title for the entire figure
fig.suptitle('Bending of Continuum Spring Actuator: Side and Front Views', fontsize=18, fontweight='bold')

# ============================================================================
# LEFT PANEL: SIDE VIEW (shows α)
# ============================================================================
ax1.set_aspect('equal')
ax1.axis('off')

# Base rectangle - CENTERED at origin
ax1.add_patch(patches.Rectangle(
    (origin_x_side - base_width/2, origin_y_side - base_height/2),
    0.8, base_height,
    facecolor='#404040', edgecolor='black', linewidth=2.5
))

# Spring - starts from origin
ax1.plot(x_side, z_side, color='#A0A0A0', linewidth=28, solid_capstyle='round', alpha=0.7)
ax1.plot(x_side, z_side, color='#606060', linewidth=3, solid_capstyle='round')

# Red line from origin to tip
ax1.plot([origin_x_side, x_tip_side], [origin_y_side, z_tip_side],
         color='#E74C3C', linewidth=2.5)

# α arc - centered at origin
angle_to_tip = np.degrees(np.arctan2(z_tip_side - origin_y_side, x_tip_side - origin_x_side))
arc_alpha = Arc((origin_x_side, origin_y_side), 2*arc_radius_alpha, 2*arc_radius_alpha,
                angle=0, theta1=0, theta2=angle_to_tip,
                color='#E74C3C', linewidth=2.5)
ax1.add_patch(arc_alpha)

# α label
alpha_label_angle = np.radians(angle_to_tip / 2)
alpha_label_r = arc_radius_alpha + 0.4
ax1.text(origin_x_side + alpha_label_r * np.cos(alpha_label_angle),
         origin_y_side + alpha_label_r * np.sin(alpha_label_angle),
         'α', fontsize=26, ha='center', va='center',
         color='#E74C3C', fontweight='bold')

# Axes
arrow_color = '#2C3E50'
axis_length = 1.5

ax1.annotate('', xy=(origin_x_side + 4.0 + 0.3, origin_y_side),
             xytext=(origin_x_side - 0.3, origin_y_side),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax1.text(origin_x_side + 4.0 + 0.5, origin_y_side, 'x',
         fontsize=16, ha='left', va='center', fontweight='bold')

ax1.annotate('', xy=(origin_x_side, origin_y_side + 3.5),
             xytext=(origin_x_side, origin_y_side - 0.3),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax1.text(origin_x_side, origin_y_side + 3.7, 'z',
         fontsize=16, ha='center', va='bottom', fontweight='bold')

# Modified limits for equal axis length
ax1.set_xlim(-2, 5)
ax1.set_ylim(-2, 5)
ax1.set_title('Side View', fontsize=16, fontweight='bold', pad=15) # Keep subplot title for clarity

# ============================================================================
# RIGHT PANEL: TOP VIEW (shows ρ)
# ============================================================================
ax2.set_aspect('equal')
ax2.axis('off')

# Base circle - CENTERED at origin
base_radius = 0.3
ax2.add_patch(patches.Circle((origin_x_top, origin_y_top), base_radius,
                              facecolor='#404040', edgecolor='black', linewidth=2.5))

# Spring - starts from origin
ax2.plot(x_top, y_top, color='#A0A0A0', linewidth=28, solid_capstyle='round', alpha=0.7)
ax2.plot(x_top, y_top, color='black', linewidth=2.5, solid_capstyle='round', linestyle='-')

# Reference line along x-axis
ax2.plot([origin_x_top, x_tip_top + 0.5], [origin_y_top, origin_y_top],
         color='gray', linestyle='--', linewidth=2, alpha=0.6)

# Blue dashed line from origin to tip
ax2.plot([origin_x_top, x_tip_top], [origin_y_top, y_tip_top],
         color='#3498DB', linewidth=2.5, linestyle='--')

# ρ arc
arc_rho = Arc((origin_x_top, origin_y_top), 2*arc_radius_rho, 2*arc_radius_rho,
              angle=0, theta1=0, theta2=rho_deg,
              color='#3498DB', linewidth=2.5)
ax2.add_patch(arc_rho)

# ρ label
rho_label_angle = np.radians(rho_deg / 2)
rho_label_r = arc_radius_rho + 0.4
ax2.text(origin_x_top + rho_label_r * np.cos(rho_label_angle),
         origin_y_top + rho_label_r * np.sin(rho_label_angle),
         'ρ', fontsize=26, ha='center', va='center',
         color='#3498DB', fontweight='bold')

# Axes
ax2.annotate('', xy=(origin_x_top + 4.0, origin_y_top),
             xytext=(origin_x_top - 0.3, origin_y_top),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax2.text(origin_x_top + 4.2, origin_y_top, 'x',
         fontsize=16, ha='left', va='center', fontweight='bold')

ax2.annotate('', xy=(origin_x_top, origin_y_top + 3.5),
             xytext=(origin_x_top, origin_y_top - 0.3),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
ax2.text(origin_x_top, origin_y_top + 3.7, 'y',
         fontsize=16, ha='center', va='bottom', fontweight='bold')

# Modified limits for equal axis length
ax2.set_xlim(-1, 5)
ax2.set_ylim(-1.5, 4.5)
ax2.set_title('Front View', fontsize=16, fontweight='bold', pad=15) # Keep subplot title for clarity

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle
plt.savefig('figure_views_combined.png', dpi=300, bbox_inches='tight', facecolor='white')


plt.show()