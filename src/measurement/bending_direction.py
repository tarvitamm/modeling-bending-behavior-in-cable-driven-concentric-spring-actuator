"""
Bending Direction Measurement Tool

Interactive measurement tool for extracting bending direction angles
from front-view experimental images of continuum actuators.

Methodology:
1. Manual selection of three points defining bending geometry
2. Vector-based angle computation between reference and deflection lines
3. Quantitative angle measurement with visual verification
"""

import cv2
import numpy as np
from pathlib import Path
import math
import csv

# Experimental configuration
VIDEO_NAME = "front_view"
FRAME_FOLDER = Path("selected_frames") / VIDEO_NAME
OUTPUT_CSV = Path("bending_direction.csv")

# Collect experimental frame sequence
frame_paths = sorted(FRAME_FOLDER.glob("*.jpg"))

def angle_between(v1, v2):
    """Compute angle in degrees between two vectors"""
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

# Initialize results storage for angle measurements
results = []

for frame_path in frame_paths:
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"Warning: Could not load {frame_path.name}")
        continue

    clone = img.copy()
    points = []

    # Draw reference lines for angle measurement
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    ref_start = (center_x - 400, center_y + 115)
    ref_end = (center_x - 15, center_y + 115)
    cv2.line(img, ref_start, ref_end, (255, 0, 0), 3)

    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    ref_start = (center_x - 15, center_y - 330)
    ref_end = (center_x - 15, center_y + 115)
    cv2.line(img, ref_start, ref_end, (0, 0, 255), 3)

    # Interactive point selection handler
    def click_event(event, x, y, flags, param):
        global points, img
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Measure Angle", img)

            if len(points) == 3:
                A, B, C = points  # Three-point angle measurement

                # Visualize measurement lines
                cv2.line(img, A, B, (255, 0, 0), 2)
                cv2.line(img, C, B, (0, 0, 255), 2)

                # Compute angle between vectors
                v1 = np.array(A) - np.array(B)
                v2 = np.array(C) - np.array(B)
                angle = angle_between(v1, v2)

                # Display angle measurement
                offset_vec = v1 / (np.linalg.norm(v1) + 1e-6)
                text_offset = 300
                text_pos = (int(B[0] + offset_vec[0] * text_offset),
                            int(B[1] + offset_vec[1] * text_offset - 30))
                cv2.putText(img, f"{angle:.2f}", text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                print(f"Measured angle for {frame_path.name}: {angle:.2f}°")
                results.append([frame_path.name, angle])
                cv2.imshow("Measure Angle", img)
                cv2.waitKey(0)
    print(f"Processing frame: {frame_path.name}")
    cv2.imshow("Measure Angle", img)
    cv2.setMouseCallback("Measure Angle", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Export measurements to CSV file
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Angle (degrees)"])
    writer.writerows(results)

print(f"\nExported {len(results)} angle measurements to {OUTPUT_CSV}")