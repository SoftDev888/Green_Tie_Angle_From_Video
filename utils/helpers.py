"""
Helper functions for green tie detection processing.
Contains the complete computer vision pipeline for detecting green ties.
"""

import cv2
import numpy as np
import math
import os
from typing import List, Dict, Any, Tuple, Optional

# ----------------- Image Processing Helpers -----------------
def resize_keep_ar(img, target_w):
    """Resize image while maintaining aspect ratio"""
    h, w = img.shape[:2]
    if w <= target_w:
        return img, 1.0
    s = target_w / float(w)
    return cv2.resize(img, (int(w*s), int(h*s)), cv2.INTER_AREA), s

def white_balance_grayworld(img):
    """Apply gray world white balancing"""
    f = img.astype(np.float32)
    mB, mG, mR = f.reshape(-1, 3).mean(axis=0)
    mg = (mB + mG + mR) / 3.0 + 1e-6
    gain = np.array([mg/(mB+1e-6), mg/(mG+1e-6), mg/(mR+1e-6)], np.float32)
    return np.clip(f * gain, 0, 255).astype(np.uint8)

# ----------------- Green Detection -----------------
def green_mask(bgr):
    """Detect green points using HSV + green-dominance"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(bgr)
    ssum = r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32) + 1e-6
    rr = r.astype(np.float32) / ssum
    gg = g.astype(np.float32) / ssum

    # Hue window for neon/lime greens
    mask_h = cv2.inRange(hsv, (18, int(0.20*255), int(0.12*255)), (100, 255, 255))
    mask_g = (gg > rr + 0.04) & (gg > 0.30)
    m = ((mask_h > 0) & mask_g).astype(np.uint8) * 255

    # Clean up the mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
    return m

# ----------------- Outline Detection -----------------
def get_outlines_from_mask(mask):
    """Get clean outlines using differential operations"""
    # Method 1: Morphological gradient (dilation - erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    eroded = cv2.erode(mask, kernel, iterations=2)
    outline = cv2.subtract(dilated, eroded)
    
    # Method 2: Canny edges for additional edge detection
    edges = cv2.Canny(mask, 50, 150)
    
    # Combine both methods
    combined = cv2.bitwise_or(outline, edges)
    
    return combined

# ----------------- Line Linking Functions -----------------
def can_lines_be_linked(line1, line2, distance_tolerance):
    """Check if two line segments can be linked together"""
    # Calculate distances between endpoints
    points1 = [np.array(line1['points'][0]), np.array(line1['points'][1])]
    points2 = [np.array(line2['points'][0]), np.array(line2['points'][1])]
    
    min_distance = float('inf')
    for p1 in points1:
        for p2 in points2:
            distance = np.linalg.norm(p1 - p2)
            min_distance = min(min_distance, distance)
    
    # Also check if lines are collinear
    line1_vec = line1['vector']
    line2_vec = line2['vector']
    dot_product = abs(np.dot(line1_vec, line2_vec) / 
                     (np.linalg.norm(line1_vec) * np.linalg.norm(line2_vec) + 1e-9))
    
    return min_distance <= distance_tolerance and dot_product > 0.9

def merge_line_group(line_group):
    """Merge a group of line segments into one continuous line"""
    if not line_group:
        return None
    
    # Collect all endpoints
    all_points = []
    for line in line_group:
        all_points.extend(line['points'])
    
    # Find the two farthest points to form the merged line
    if len(all_points) < 2:
        return line_group[0]
    
    # Convert to numpy array for easier calculation
    points_array = np.array(all_points)
    
    # Find the two most distant points
    max_distance = 0
    best_points = None
    
    for i in range(len(points_array)):
        for j in range(i + 1, len(points_array)):
            distance = np.linalg.norm(points_array[i] - points_array[j])
            if distance > max_distance:
                max_distance = distance
                best_points = (points_array[i], points_array[j])
    
    if best_points is None:
        return line_group[0]
    
    p1, p2 = best_points
    
    # Calculate merged line properties
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.linalg.norm([dx, dy])
    angle = math.degrees(math.atan2(dy, dx)) % 180
    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    direction = np.array([dx, dy]) / (length + 1e-9)
    
    return {
        'points': [tuple(p1), tuple(p2)],
        'length': length,
        'angle': angle,
        'center': center,
        'direction': direction,
        'vector': np.array([dx, dy]),
        'merged_count': len(line_group)  # Track how many segments were merged
    }

def link_neighbor_lines(lines, angle_tolerance=15, distance_tolerance=50, min_group_size=2):
    """Link fragmented line segments into complete lines"""
    if len(lines) < 2:
        return lines
    
    # Group lines by similar angle
    groups = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
            
        # Start a new group
        group = [line1]
        used.add(i)
        
        # Find lines with similar angle and close endpoints
        for j, line2 in enumerate(lines):
            if j in used or i == j:
                continue
                
            # Check angle similarity
            angle_diff = min(
                abs(line1['angle'] - line2['angle']),
                abs(line1['angle'] - line2['angle'] - 180),
                abs(line1['angle'] - line2['angle'] + 180)
            )
            
            if angle_diff > angle_tolerance:
                continue
                
            # Check if lines are close to each other (can be linked)
            if can_lines_be_linked(line1, line2, distance_tolerance):
                group.append(line2)
                used.add(j)
        
        groups.append(group)
    
    # Merge lines in each group
    merged_lines = []
    for group in groups:
        if len(group) >= min_group_size:
            merged_line = merge_line_group(group)
            merged_lines.append(merged_line)
        else:
            # Keep original lines if group is too small
            merged_lines.extend(group)
    
    return merged_lines

# ----------------- Line Detection -----------------
def find_straight_lines_from_outlines(outline_mask, min_line_length=50):
    """Find straight lines from outlines using Hough transform with line linking"""
    # Use HoughLinesP to detect line segments
    lines = cv2.HoughLinesP(outline_mask, 1, np.pi/180, threshold=25,
                           minLineLength=min_line_length, maxLineGap=10)
    
    if lines is None:
        return []
    
    # Convert to a more usable format
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate direction vector
        dx, dy = x2 - x1, y2 - y1
        direction = np.array([dx, dy])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        
        line_data.append({
            'points': [(x1, y1), (x2, y2)],
            'length': length,
            'angle': angle,
            'center': (center_x, center_y),
            'direction': direction,
            'vector': np.array([dx, dy])
        })
    
    # Link neighbor lines before returning
    linked_lines = link_neighbor_lines(
        line_data, 
        angle_tolerance=1.5,  # Default values
        distance_tolerance=5,
        min_group_size=2
    )
    
    return linked_lines

def find_two_main_directions(lines, green_mask):
    """Find the two main line directions from all detected lines"""
    if len(lines) < 2:
        return None, None
    
    # Group lines by angle (with some tolerance)
    angle_groups = {}
    angle_tolerance = 15  # degrees
    
    for line in lines:
        angle = line['angle']
        
        # Find existing group with similar angle
        matched = False
        for group_angle in angle_groups:
            angle_diff = min(
                abs(angle - group_angle),
                abs(angle - group_angle - 180),
                abs(angle - group_angle + 180)
            )
            if angle_diff <= angle_tolerance:
                angle_groups[group_angle].append(line)
                matched = True
                break
        
        if not matched:
            angle_groups[angle] = [line]
    
    # If we have at least two angle groups, take the strongest ones
    if len(angle_groups) >= 2:
        # Sort groups by total line length
        groups_sorted = sorted(angle_groups.items(), 
                             key=lambda x: sum(line['length'] for line in x[1]), 
                             reverse=True)
        
        group1_angle, group1_lines = groups_sorted[0]
        group2_angle, group2_lines = groups_sorted[1]
        
        # Take the longest line from each group
        line1 = max(group1_lines, key=lambda x: x['length'])
        line2 = max(group2_lines, key=lambda x: x['length'])
        
        return line1, line2
    
    # If only one angle group, try to find lines that are spatially separated
    elif len(angle_groups) == 1:
        lines_sorted = sorted(lines, key=lambda x: x['length'], reverse=True)
        
        if len(lines_sorted) >= 2:
            # Take the two longest lines that are spatially separated
            line1 = lines_sorted[0]
            line2 = None
            
            for candidate in lines_sorted[1:]:
                center_dist = math.hypot(
                    candidate['center'][0] - line1['center'][0],
                    candidate['center'][1] - line1['center'][1]
                )
                
                if center_dist > green_mask.shape[0] * 0.2:  # Reasonable separation
                    line2 = candidate
                    break
            
            if line2 is None and len(lines_sorted) >= 2:
                line2 = lines_sorted[1]  # Fallback to second longest
            
            return line1, line2
    
    return None, None

# ----------------- Geometry Functions -----------------
def angle_between(u, v):
    """Calculate angle between two vectors in degrees"""
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    dot = max(-1.0, min(1.0, abs(float(np.dot(u, v)))))
    return math.degrees(math.acos(dot))

def overlay_line(img, mu, d, length_px, color):
    """Overlay a line on the image for visualization"""
    p1 = (mu - d * length_px / 2).astype(int)
    p2 = (mu + d * length_px / 2).astype(int)
    cv2.line(img, tuple(p1), tuple(p2), color, 4, cv2.LINE_AA)
    cv2.circle(img, tuple(mu.astype(int)), 4, (255, 255, 255), -1)

# ----------------- Debugging Functions -----------------
def save_debug_step(img, step_name, tag, extra_data=None):
    """Save debug image for each processing step"""
    debug_img = img.copy()
    
    if step_name == "1_original":
        cv2.putText(debug_img, "Step 1: Original Image", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    elif step_name == "2_green_mask":
        cv2.putText(debug_img, "Step 2: Green Mask", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        green_pixels = cv2.countNonZero(img)
        cv2.putText(debug_img, f"Green pixels: {green_pixels}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    elif step_name == "3_outlines":
        cv2.putText(debug_img, "Step 3: Outlines", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        outline_pixels = cv2.countNonZero(img)
        cv2.putText(debug_img, f"Outline pixels: {outline_pixels}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    elif step_name == "4_lines_detected":
        cv2.putText(debug_img, "Step 4: Lines Detected", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if extra_data:
            cv2.putText(debug_img, f"Lines found: {extra_data}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    elif step_name == "5_final_result":
        cv2.putText(debug_img, "Step 5: Final Result", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if extra_data:
            angle, fqs = extra_data
            cv2.putText(debug_img, f"Angle: {angle:.1f}°, FQS: {fqs:.3f}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return debug_img

# ----------------- API Helper Functions -----------------
def validate_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize processing parameters"""
    validated = parameters.copy()
    
    # Ensure positive values within reasonable ranges
    if 'target_width' in validated:
        validated['target_width'] = max(100, min(2000, validated['target_width']))
    
    if 'line_linking_distance_tolerance' in validated:
        validated['line_linking_distance_tolerance'] = max(1, min(50, validated['line_linking_distance_tolerance']))
    
    if 'line_linking_angle_tolerance' in validated:
        validated['line_linking_angle_tolerance'] = max(0.1, min(10.0, validated['line_linking_angle_tolerance']))
    
    if 'min_line_group_size' in validated:
        validated['min_line_group_size'] = max(1, min(10, validated['min_line_group_size']))
    
    if 'max_samples' in validated and validated['max_samples'] is not None:
        validated['max_samples'] = max(1, validated['max_samples'])
    
    return validated

def calculate_fqs(line1: Dict, line2: Dict, mask_shape: Tuple[int, int]) -> float:
    """Calculate Frame Quality Score"""
    H, W = mask_shape
    avg_length = (line1['length'] + line2['length']) / 2
    length_score = min(1.0, avg_length / (0.6 * H))
    
    # Calculate angle between lines
    theta = angle_between(line1['direction'], line2['direction'])
    angle_score = 1.0 if 20 < theta < 160 else 0.5  # Prefer reasonable angles
    
    return 0.6 * length_score + 0.4 * angle_score

def create_result_dict(frame_idx: int, angle_deg: float, fqs: float, lines_detected: int, 
                      green_pixels: int, line1: Dict, line2: Dict) -> Dict[str, Any]:
    """Create standardized result dictionary"""
    return {
        "frame_idx": frame_idx,
        "angle_deg": float(angle_deg),
        "fqs": float(fqs),
        "lines_detected": lines_detected,
        "green_pixels": green_pixels,
        "line1_length": float(line1['length']),
        "line2_length": float(line2['length']),
        "line1_angle": float(line1['angle']),
        "line2_angle": float(line2['angle']),
        "line1_merged_count": line1.get('merged_count', 1),
        "line2_merged_count": line2.get('merged_count', 1)
    }

def process_frame_pipeline(frame: np.ndarray, parameters: Dict[str, Any], 
                          frame_idx: int = 0, debug_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Complete frame processing pipeline
    Returns None if no green ties detected, otherwise returns result dict
    """
    try:
        # Step 1: Preprocessing
        target_width = parameters.get('target_width', 900)
        img, scale = resize_keep_ar(frame, target_width)
        img = white_balance_grayworld(img)
        
        if debug_dir:
            debug_img = save_debug_step(img, "1_original", f"frame_{frame_idx:06d}")
            cv2.imwrite(os.path.join(debug_dir, f"1_original_frame_{frame_idx:06d}.png"), debug_img)
        
        # Step 2: Green mask detection
        mask = green_mask(img)
        green_pixels = cv2.countNonZero(mask)
        
        if debug_dir:
            debug_img = save_debug_step(mask, "2_green_mask", f"frame_{frame_idx:06d}")
            cv2.imwrite(os.path.join(debug_dir, f"2_green_mask_frame_{frame_idx:06d}.png"), debug_img)
        
        if green_pixels < 200:
            return None
        
        # Step 3: Outline detection
        outline_mask = get_outlines_from_mask(mask)
        outline_pixels = cv2.countNonZero(outline_mask)
        
        if debug_dir:
            debug_img = save_debug_step(outline_mask, "3_outlines", f"frame_{frame_idx:06d}")
            cv2.imwrite(os.path.join(debug_dir, f"3_outlines_frame_{frame_idx:06d}.png"), debug_img)
        
        if outline_pixels < 50:
            return None
        
        # Step 4: Line detection with linking
        line_params = {
            'angle_tolerance': parameters.get('line_linking_angle_tolerance', 1.5),
            'distance_tolerance': parameters.get('line_linking_distance_tolerance', 5),
            'min_group_size': parameters.get('min_line_group_size', 2)
        }
        
        lines = find_straight_lines_from_outlines(outline_mask, min_line_length=30)
        
        # Apply custom linking parameters
        if len(lines) > 1:
            lines = link_neighbor_lines(
                lines,
                angle_tolerance=line_params['angle_tolerance'],
                distance_tolerance=line_params['distance_tolerance'],
                min_group_size=line_params['min_group_size']
            )
        
        if debug_dir:
            lines_img = img.copy()
            for i, line in enumerate(lines):
                x1, y1 = line['points'][0]
                x2, y2 = line['points'][1]
                color = (0, 255, 0) if line.get('merged_count', 1) > 1 else (0, 255, 255)
                thickness = 3 if line.get('merged_count', 1) > 1 else 2
                cv2.line(lines_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            debug_img = save_debug_step(lines_img, "4_lines_detected", f"frame_{frame_idx:06d}", len(lines))
            cv2.imwrite(os.path.join(debug_dir, f"4_lines_detected_frame_{frame_idx:06d}.png"), debug_img)
        
        if len(lines) < 2:
            return None
        
        # Step 5: Find main directions and calculate angle
        line1, line2 = find_two_main_directions(lines, mask)
        if line1 is None or line2 is None:
            return None
        
        theta = angle_between(line1['direction'], line2['direction'])
        fqs = calculate_fqs(line1, line2, mask.shape)
        
        # Step 6: Create final visualization (optional)
        if debug_dir:
            final_img = img.copy()
            # Draw the two selected lines
            x1, y1 = line1['points'][0]
            x2, y2 = line1['points'][1]
            cv2.line(final_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            
            x1, y1 = line2['points'][0]
            x2, y2 = line2['points'][1]
            cv2.line(final_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            
            debug_img = save_debug_step(final_img, "5_final_result", f"frame_{frame_idx:06d}", (theta, fqs))
            cv2.imwrite(os.path.join(debug_dir, f"5_final_result_frame_{frame_idx:06d}.png"), debug_img)
        
        # Return standardized results
        return create_result_dict(frame_idx, theta, fqs, len(lines), green_pixels, line1, line2)
        
    except Exception as e:
        print(f"Error in frame processing pipeline (frame {frame_idx}): {e}")
        return None

def validate_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize processing parameters"""
    validated = parameters.copy()
    
    # Ensure positive values within reasonable ranges
    if 'target_width' in validated:
        validated['target_width'] = max(100, min(2000, validated['target_width']))
    
    if 'line_linking_distance_tolerance' in validated:
        validated['line_linking_distance_tolerance'] = max(1, min(50, validated['line_linking_distance_tolerance']))
    
    if 'line_linking_angle_tolerance' in validated:
        validated['line_linking_angle_tolerance'] = max(0.1, min(10.0, validated['line_linking_angle_tolerance']))
    
    if 'min_line_group_size' in validated:
        validated['min_line_group_size'] = max(1, min(10, validated['min_line_group_size']))
    
    if 'max_samples' in validated and validated['max_samples'] is not None:
        validated['max_samples'] = max(1, validated['max_samples'])
    
    return validated