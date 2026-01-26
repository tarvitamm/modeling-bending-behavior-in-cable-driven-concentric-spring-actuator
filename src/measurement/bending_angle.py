"""
Bending Angle Analysis Tool


Methodology:
1. Interactive curve tracing through mouse input
2. Arc-length parameterized resampling for uniform point distribution
3. Tangent vector computation using central difference approximation
4. Angle measurement relative to horizontal reference axis

"""

import cv2
import numpy as np
import math
import csv
from pathlib import Path

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

# Video dataset identifier - modify this to process different experimental conditions
VIDEO_NAME = "side_view"  

# Directory structure for experimental data
INPUT_DIR = Path("selected_frames") / VIDEO_NAME      # Source frames for analysis
ANNOTATED_DIR = Path("annotated_frames") / VIDEO_NAME # Output directory for annotated results
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Results file for quantitative measurements
OUTPUT_CSV = f"angle_results_{VIDEO_NAME}.csv"

# Initialize CSV file with headers for measurement data
if not Path(OUTPUT_CSV).exists():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "1st point", "2nd point", "3rd point", "4th point", "final angle"])

# =============================================================================
# MATHEMATICAL ANALYSIS FUNCTIONS
# =============================================================================

def angle_between(v1, v2):
    """
    Compute signed angle between two vectors using cross product.
    
    This function calculates the oriented angle between two 2D vectors,
    preserving directional information essential for deformation analysis.
    
    Args:
        v1, v2 (numpy.ndarray): Input vectors in 2D space
        
    Returns:
        float: Signed angle in degrees (-180° to +180°)
        
    Mathematical basis:
        θ = arccos(v1·v2 / |v1||v2|) with sign from v1 × v2
    """
    # Normalize vectors to unit length
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    
    # Compute dot product with numerical stability
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    
    # Determine sign using cross product (2D: z-component only)
    cross = v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0]
    return angle_deg if cross >= 0 else -angle_deg

def compute_tangent(points, i, window=2):
    """
    Compute tangent vector at point i using central difference approximation.
    
    This method estimates the local tangent direction by using neighboring points
    within a specified window, providing robust derivatives for discrete curve data.
    
    Args:
        points (list): Sequence of (x,y) coordinates defining the curve
        i (int): Index of point where tangent is computed
        window (int): Half-width of neighborhood for derivative estimation
        
    Returns:
        numpy.ndarray: Tangent vector as 2D displacement
        
    Mathematical basis:
        T(i) ≈ (P(i+w) - P(i-w)) / (2w) where w is the window size
    """
    # Define neighborhood bounds with boundary protection
    i1 = max(i - window, 0)
    i2 = min(i + window, len(points) - 1)
    
    # Compute tangent as finite difference
    pt1 = np.array(points[i1])
    pt2 = np.array(points[i2])
    return pt2 - pt1

def resample_points(points, n=5):
    """
    Resample curve to uniform arc-length parameterization.
    
    This function redistributes points along the curve based on arc-length
    rather than pixel coordinates, ensuring consistent sampling density
    for reliable tangent computation and angle measurements.
    
    Args:
        points (list): Original sequence of (x,y) coordinates
        n (int): Number of points in resampled curve
        
    Returns:
        list: Uniformly distributed points along the curve
        
    Mathematical approach:
        1. Compute cumulative arc-length function s(i)
        2. Define uniform parameter values t_k = k*L/(n-1)
        3. Interpolate spatial coordinates at parameter values
    """
    # Compute cumulative arc-length distances
    distances = [0]
    for i in range(1, len(points)):
        dist = np.linalg.norm(np.array(points[i]) - np.array(points[i - 1]))
        distances.append(distances[-1] + dist)

    # Total curve length and uniform interval spacing
    total_len = distances[-1]
    interval = total_len / (n - 1)

    # Initialize with start point
    resampled = [points[0]]
    j = 1
    
    # Interpolate intermediate points at uniform arc-length intervals
    for i in range(1, n - 1):
        target_len = i * interval
        
        # Find bracketing points in original parameterization
        while j < len(distances) and distances[j] < target_len:
            j += 1
        if j >= len(points):
            break
            
        # Linear interpolation between bracketing points
        ratio = (target_len - distances[j - 1]) / (distances[j] - distances[j - 1])
        pt1 = np.array(points[j - 1])
        pt2 = np.array(points[j])
        interp = pt1 + ratio * (pt2 - pt1)
        resampled.append(tuple(interp))
    
    # Add endpoint
    resampled.append(points[-1])
    return resampled

# =============================================================================
# INTERACTIVE MEASUREMENT PROTOCOL
# =============================================================================

# Load experimental frame sequence
image_files = sorted(INPUT_DIR.glob("*.jpg"))
print(f"Processing {len(image_files)} experimental frames from {VIDEO_NAME}")

for frame_index, image_path in enumerate(image_files, 1):
    frame_name = image_path.name
    
    # Load and validate image data
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load frame {frame_name}")
        continue
    
    # Create working copy for restoration capability
    clone = img.copy()
    
    # Initialize measurement state variables
    drawing = False                 # Mouse interaction state
    curve_points = []              # Raw traced coordinates
    result_ready = False           # Measurement completion flag
    annotated = None               # Processed visualization
    angles = []                    # Computed angle measurements

    def mouse_callback(event, x, y, flags, param):
        """
        Handle mouse events for interactive curve tracing.
        
        Implements real-time curve drawing with immediate visual feedback
        and automatic measurement computation upon curve completion.
        """
        global drawing, curve_points, img, annotated, angles, result_ready
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Initialize new curve tracing session
            drawing = True
            curve_points = [(x, y)]
            img = clone.copy()
            annotated = None
            result_ready = False

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Real-time curve visualization during tracing
            curve_points.append((x, y))
            if len(curve_points) >= 2:
                cv2.line(img, curve_points[-2], curve_points[-1], (0, 0, 255), 2)
            cv2.imshow("Draw Curve", img)

        elif event == cv2.EVENT_LBUTTONUP:
            # Complete curve and perform measurements
            drawing = False
            
            # Validate minimum curve length for reliable analysis
            if len(curve_points) < 5:
                print("Curve too short - minimum 5 points required for analysis")
                return

            # Apply arc-length resampling for uniform point distribution
            resampled = resample_points(curve_points, n=5)
            
            # Establish reference orientation from initial tangent
            base_vec = compute_tangent(resampled, 0)
            base_angle = math.degrees(math.atan2(base_vec[1], base_vec[0]))

            # Perform angle measurements at each resampled point
            angles = []
            annotated = clone.copy()
            
            for i, pt in enumerate(resampled):
                # Compute local tangent vector
                tan_vec = compute_tangent(resampled, i)
                
                # Calculate angle relative to horizontal reference (0°)
                # Note: +180° offset normalizes range to [0°, 360°)
                delta = math.degrees(math.atan2(tan_vec[1], tan_vec[0])) + 180

                # Visualize tangent vector for verification
                v_unit = tan_vec / (np.linalg.norm(tan_vec) + 1e-6)
                tangent_length = 50  # Pixels for visualization
                pt1 = (int(pt[0] - v_unit[0]*tangent_length), int(pt[1] - v_unit[1]*tangent_length))
                pt2 = (int(pt[0] + v_unit[0]*tangent_length), int(pt[1] + v_unit[1]*tangent_length))
                cv2.line(annotated, pt1, pt2, (255, 0, 0), 2)

                # Annotate measured angle value
                cv2.putText(annotated, f"{delta:.1f}°", (int(pt[0] + 10), int(pt[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                angles.append(delta)

            cv2.imshow("Draw Curve", annotated)
            result_ready = True

    # Configure interactive measurement interface
    cv2.namedWindow("Draw Curve", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Draw Curve", mouse_callback)
    cv2.imshow("Draw Curve", img)
    
    print(f"\n🖼️  Frame {frame_index}/{len(image_files)}: {frame_name}")
    print("   Protocol: Trace specimen curve, then press 's' to save or 'r' to retry")

    # Interactive measurement session
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord("s") and result_ready:
            # Export quantitative measurements to CSV database
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_name] + [round(angle, 2) for angle in angles])

            # Save annotated image for qualitative verification
            save_path = ANNOTATED_DIR / frame_name
            cv2.imwrite(str(save_path), annotated)
            print(f"Measurements saved: {frame_name}")
            break
            
        elif key == ord("r"):
            # Reset measurement session for current frame
            img = clone.copy()
            curve_points = []
            result_ready = False
            cv2.imshow("Draw Curve", img)
            print("Measurement session reset - trace curve again")
            
        elif key == 27:  # ESC key
            print(f"Frame skipped: {frame_name}")
            break

    # Clean up GUI resources
    cv2.destroyAllWindows()