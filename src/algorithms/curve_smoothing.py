"""
Curve Smoothing and Simplification Utilities

Handles the clustering/overlapping issues on curved road segments:
  - Douglas-Peucker simplification with adaptive tolerance
  - Visvalingam-Whyatt area-based simplification
  - Curve-aware point merging
  - Chaikin's corner cutting for smooth curves
"""

import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import LineString


def angle_diff_deg(a: float, b: float) -> float:
    """Compute minimum angle difference in degrees (0-180)."""
    d = abs((a - b) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    return d


def bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute bearing from (x1,y1) to (x2,y2) in degrees [0, 360)."""
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy))  # atan2(dx,dy) for north=0
    return (angle + 360.0) % 360.0


def douglas_peucker_simplify(
    coords: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """
    Apply Douglas-Peucker simplification.
    
    Args:
        coords: Array of (x, y) coordinates
        tolerance: Distance tolerance for simplification
        
    Returns:
        Simplified coordinate array
    """
    if len(coords) < 3:
        return coords.copy()
    
    line = LineString(coords)
    simplified = line.simplify(tolerance, preserve_topology=True)
    
    if simplified.is_empty:
        return coords.copy()
    
    return np.array(simplified.coords)


def adaptive_simplify(
    coords: np.ndarray,
    base_tolerance: float = 2.0,
    curve_multiplier: float = 0.5,
    straight_multiplier: float = 2.0,
) -> np.ndarray:
    """
    Adaptive simplification that uses different tolerances for
    straight vs curved sections.
    
    Curves are simplified less aggressively to preserve shape,
    while straight sections can be simplified more.
    
    Args:
        coords: Array of (x, y) coordinates
        base_tolerance: Base tolerance for simplification
        curve_multiplier: Tolerance multiplier for curves (< 1 = less simplification)
        straight_multiplier: Tolerance multiplier for straight segments
        
    Returns:
        Simplified coordinate array
    """
    if len(coords) < 4:
        return coords.copy()
    
    # Detect curve and straight sections
    sections = detect_curve_sections(coords)
    
    result_coords = []
    processed_up_to = 0
    
    for start, end, is_curve in sections:
        if start > processed_up_to:
            # Add any gap points
            result_coords.extend(coords[processed_up_to:start].tolist())
        
        section = coords[start:end + 1]
        if len(section) < 3:
            result_coords.extend(section.tolist())
        else:
            tol = base_tolerance * (curve_multiplier if is_curve else straight_multiplier)
            simplified = douglas_peucker_simplify(section, tol)
            # Avoid duplicating endpoints between sections
            if result_coords and len(simplified) > 0:
                if np.allclose(result_coords[-1], simplified[0], atol=0.01):
                    simplified = simplified[1:]
            result_coords.extend(simplified.tolist())
        
        processed_up_to = end + 1
    
    # Add remaining points
    if processed_up_to < len(coords):
        result_coords.extend(coords[processed_up_to:].tolist())
    
    return np.array(result_coords) if result_coords else coords.copy()


def detect_curve_sections(
    coords: np.ndarray,
    heading_threshold_deg: float = 20.0,
    min_section_length: int = 3,
) -> List[Tuple[int, int, bool]]:
    """
    Detect curve vs straight sections in a polyline.
    
    Args:
        coords: Array of (x, y) coordinates
        heading_threshold_deg: Angle change per segment indicating curve
        min_section_length: Minimum points per section
        
    Returns:
        List of (start_idx, end_idx, is_curve) tuples
    """
    if len(coords) < 3:
        return [(0, len(coords) - 1, False)]
    
    # Compute segment headings
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    headings = np.degrees(np.arctan2(dy, dx))
    
    # Compute heading changes
    heading_changes = np.abs(np.diff(headings))
    heading_changes = np.minimum(heading_changes, 360 - heading_changes)
    
    sections = []
    i = 0
    n = len(heading_changes)
    
    while i < n:
        # Determine section type
        is_curve = heading_changes[i] >= heading_threshold_deg
        start = i
        
        # Extend section while type matches
        while i < n and (heading_changes[i] >= heading_threshold_deg) == is_curve:
            i += 1
        
        end = min(i + 1, len(coords) - 1)  # +1 because heading_changes is shorter
        sections.append((start, end, is_curve))
    
    return sections if sections else [(0, len(coords) - 1, False)]


def merge_nearby_points(
    coords: np.ndarray,
    merge_distance: float = 2.0,
    heading_tolerance_deg: float = 30.0,
) -> np.ndarray:
    """
    Merge points that are too close together.
    
    This solves the cluster proliferation issue on curves where
    the Kharita algorithm creates many nearby nodes.
    
    Args:
        coords: Array of (x, y) coordinates
        merge_distance: Distance threshold for merging
        heading_tolerance_deg: Heading tolerance for merging curve points
        
    Returns:
        Merged coordinate array
    """
    if len(coords) < 3:
        return coords.copy()
    
    result = [coords[0].tolist()]
    
    i = 1
    while i < len(coords) - 1:
        # Check distance to last kept point
        dist = np.hypot(
            coords[i, 0] - result[-1][0],
            coords[i, 1] - result[-1][1]
        )
        
        if dist < merge_distance:
            # Check if this is within a curve (look at heading progression)
            if i < len(coords) - 1:
                h1 = bearing_from_xy(result[-1][0], result[-1][1],
                                     coords[i, 0], coords[i, 1])
                h2 = bearing_from_xy(coords[i, 0], coords[i, 1],
                                     coords[i + 1, 0], coords[i + 1, 1])
                
                if angle_diff_deg(h1, h2) > heading_tolerance_deg:
                    # This is a turn point - keep it even if close
                    result.append(coords[i].tolist())
            # Otherwise skip this point (merge with previous)
        else:
            result.append(coords[i].tolist())
        
        i += 1
    
    # Always keep last point
    result.append(coords[-1].tolist())
    
    return np.array(result)


def chaikin_smooth(
    coords: np.ndarray,
    iterations: int = 2,
    keep_endpoints: bool = True,
) -> np.ndarray:
    """
    Apply Chaikin's corner-cutting algorithm for smooth curves.
    
    This produces a smoother curve by recursively cutting corners.
    
    Args:
        coords: Array of (x, y) coordinates
        iterations: Number of smoothing iterations
        keep_endpoints: Whether to preserve exact endpoints
        
    Returns:
        Smoothed coordinate array
    """
    if len(coords) < 3 or iterations <= 0:
        return coords.copy()
    
    result = coords.copy()
    
    for _ in range(iterations):
        n = len(result)
        if n < 3:
            break
        
        new_pts = []
        
        # Handle first point
        if keep_endpoints:
            new_pts.append(result[0])
        
        # Cut corners
        for i in range(n - 1):
            p0 = result[i]
            p1 = result[i + 1]
            
            # Q = 3/4 * P0 + 1/4 * P1
            q = 0.75 * p0 + 0.25 * p1
            # R = 1/4 * P0 + 3/4 * P1
            r = 0.25 * p0 + 0.75 * p1
            
            if not keep_endpoints or i > 0:
                new_pts.append(q)
            new_pts.append(r)
        
        # Handle last point
        if keep_endpoints:
            new_pts.append(result[-1])
        
        result = np.array(new_pts)
    
    return result


def smooth_curve_preserving_shape(
    coords: np.ndarray,
    smooth_weight: float = 0.25,
    iterations: int = 2,
    turn_threshold_deg: float = 35.0,
) -> np.ndarray:
    """
    Smooth a curve while preserving its overall shape.
    
    Uses weighted averaging with neighboring points, but skips
    vertices at sharp turns to maintain road geometry.
    
    Args:
        coords: Array of (x, y) coordinates
        smooth_weight: Weight for neighbors (0-0.5)
        iterations: Number of smoothing passes
        turn_threshold_deg: Angle threshold for turn detection
        
    Returns:
        Smoothed coordinate array
    """
    if len(coords) <= 2 or iterations <= 0:
        return coords.copy()
    
    result = coords.copy()
    
    for _ in range(iterations):
        new_coords = result.copy()
        
        for i in range(1, len(result) - 1):
            # Check if this is a turn point
            h1 = bearing_from_xy(result[i-1, 0], result[i-1, 1],
                                 result[i, 0], result[i, 1])
            h2 = bearing_from_xy(result[i, 0], result[i, 1],
                                 result[i+1, 0], result[i+1, 1])
            
            if angle_diff_deg(h1, h2) >= turn_threshold_deg:
                # Preserve turn vertex
                continue
            
            # Weighted average with neighbors
            center_weight = 1.0 - 2 * smooth_weight
            new_coords[i] = (
                smooth_weight * result[i - 1] +
                center_weight * result[i] +
                smooth_weight * result[i + 1]
            )
        
        result = new_coords
    
    return result


def simplify_and_smooth_centerline(
    coords: np.ndarray,
    simplify_tolerance: float = 2.0,
    merge_distance: float = 3.0,
    smooth_iterations: int = 2,
    smooth_weight: float = 0.2,
) -> np.ndarray:
    """
    Complete centerline cleanup: simplify, merge, and smooth.
    
    This is the main function to fix the curve clustering issue.
    
    Args:
        coords: Array of (x, y) coordinates
        simplify_tolerance: Douglas-Peucker tolerance
        merge_distance: Distance for merging nearby points
        smooth_iterations: Smoothing iterations
        smooth_weight: Weight for smoothing
        
    Returns:
        Cleaned coordinate array
    """
    if len(coords) < 3:
        return coords.copy()
    
    # Step 1: Merge nearby points first (fixes cluster proliferation)
    merged = merge_nearby_points(coords, merge_distance)
    
    if len(merged) < 3:
        return merged
    
    # Step 2: Adaptive simplification (removes zigzags on curves)
    simplified = adaptive_simplify(merged, simplify_tolerance)
    
    if len(simplified) < 3:
        return simplified
    
    # Step 3: Shape-preserving smoothing
    smoothed = smooth_curve_preserving_shape(
        simplified,
        smooth_weight=smooth_weight,
        iterations=smooth_iterations,
    )
    
    return smoothed


def fix_overlapping_segments(
    segments: List[np.ndarray],
    overlap_threshold: float = 5.0,
    angle_threshold_deg: float = 30.0,
) -> List[np.ndarray]:
    """
    Fix overlapping segments that should be a single centerline.
    
    Identifies segments that are nearly parallel and overlapping,
    then merges them into a single line.
    
    Args:
        segments: List of coordinate arrays
        overlap_threshold: Distance threshold for overlap detection
        angle_threshold_deg: Heading threshold for parallel detection
        
    Returns:
        List of cleaned segments
    """
    if len(segments) <= 1:
        return segments
    
    # Build LineStrings for spatial operations
    lines = []
    for coords in segments:
        if len(coords) >= 2:
            lines.append(LineString(coords))
    
    if len(lines) <= 1:
        return segments
    
    # Find overlapping pairs
    merged_indices = set()
    result_lines = []
    
    for i, line_i in enumerate(lines):
        if i in merged_indices:
            continue
        
        # Find parallel overlapping lines
        to_merge = [line_i]
        
        for j, line_j in enumerate(lines):
            if j <= i or j in merged_indices:
                continue
            
            # Check if parallel
            h_i = bearing_from_line(line_i)
            h_j = bearing_from_line(line_j)
            
            if angle_diff_deg(h_i, h_j) > angle_threshold_deg:
                if angle_diff_deg(h_i, (h_j + 180) % 360) > angle_threshold_deg:
                    continue
            
            # Check if overlapping
            try:
                dist = line_i.hausdorff_distance(line_j)
                if dist < overlap_threshold:
                    to_merge.append(line_j)
                    merged_indices.add(j)
            except Exception:
                continue
        
        merged_indices.add(i)
        
        if len(to_merge) == 1:
            result_lines.append(np.array(to_merge[0].coords))
        else:
            # Merge overlapping lines
            merged = merge_parallel_lines(to_merge)
            result_lines.append(merged)
    
    return result_lines


def bearing_from_line(line: LineString) -> float:
    """Get overall bearing of a LineString."""
    coords = np.array(line.coords)
    if len(coords) < 2:
        return 0.0
    
    return bearing_from_xy(
        coords[0, 0], coords[0, 1],
        coords[-1, 0], coords[-1, 1]
    )


def merge_parallel_lines(lines: List[LineString]) -> np.ndarray:
    """Merge parallel overlapping lines into one."""
    if not lines:
        return np.array([])
    
    if len(lines) == 1:
        return np.array(lines[0].coords)
    
    # Collect all coordinates
    all_coords = []
    for line in lines:
        all_coords.extend(list(line.coords))
    
    points = np.array(all_coords)
    
    if len(points) < 2:
        return points
    
    # Find overall direction
    centroid = np.mean(points, axis=0)
    
    # Project all points onto the main axis
    # Use PCA to find main direction
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Project points onto main axis
    projections = np.dot(centered, main_axis)
    order = np.argsort(projections)
    
    # Average nearby points perpendicular to main axis
    result = []
    i = 0
    while i < len(order):
        # Collect nearby points
        current_proj = projections[order[i]]
        group = [order[i]]
        
        j = i + 1
        while j < len(order) and projections[order[j]] - current_proj < 3.0:
            group.append(order[j])
            j += 1
        
        # Average the group
        avg_point = np.mean(points[group], axis=0)
        result.append(avg_point)
        
        i = j
    
    return np.array(result)
