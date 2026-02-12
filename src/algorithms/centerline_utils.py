"""
Centerline Generation Utilities

Common utility functions for centerline generation algorithms:
  - Geometry operations (sampling, interpolation, smoothing)
  - Heading/bearing calculations
  - Graph operations (shortest path, stitching)
  - Coordinate transformations
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import LineString


def angle_diff_deg(a: float, b: float) -> float:
    """
    Compute minimum angle difference in degrees (0-180).
    
    Handles wraparound correctly for heading comparisons.
    """
    d = abs((a - b) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    return d


def bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Compute bearing from (x1,y1) to (x2,y2) in degrees.
    
    Returns heading in [0, 360) where 0=north, 90=east.
    """
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy))  # atan2(dx,dy) for north=0
    return (angle + 360.0) % 360.0


def sample_line_projected(
    line_xy: LineString,
    sample_spacing_m: float = 8.0,
    max_points: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a projected LineString at regular intervals.
    
    Args:
        line_xy: LineString in projected coordinates (meters)
        sample_spacing_m: Distance between sample points
        max_points: Maximum number of sample points
        
    Returns:
        Tuple of:
          - distances: Array of distances along the line
          - points: Array of (x, y) coordinates
          - headings: Array of heading values in degrees
    """
    if line_xy.is_empty or line_xy.length <= 1.0:
        return np.array([]), np.array([]).reshape(0, 2), np.array([])
    
    coords = np.array(line_xy.coords)
    if len(coords) < 2:
        return np.array([]), np.array([]).reshape(0, 2), np.array([])
    
    # Compute cumulative distances
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_dist[-1]
    
    if total_length <= 0:
        return np.array([]), np.array([]).reshape(0, 2), np.array([])
    
    # Generate sample distances
    n_samples = min(max_points, max(2, int(total_length / sample_spacing_m) + 1))
    sample_dists = np.linspace(0, total_length, n_samples)
    
    # Interpolate coordinates at sample distances
    sample_x = np.interp(sample_dists, cum_dist, coords[:, 0])
    sample_y = np.interp(sample_dists, cum_dist, coords[:, 1])
    sample_xy = np.column_stack([sample_x, sample_y])
    
    # Compute headings at each sample point
    headings = np.zeros(len(sample_xy))
    for i in range(len(sample_xy)):
        if i < len(sample_xy) - 1:
            headings[i] = bearing_from_xy(
                sample_xy[i, 0], sample_xy[i, 1],
                sample_xy[i + 1, 0], sample_xy[i + 1, 1]
            )
        else:
            headings[i] = headings[i - 1] if i > 0 else 0.0
    
    return sample_dists, sample_xy, headings


def interpolate_altitudes(
    altitudes: List[float],
    line_xy: LineString,
    sample_dists: np.ndarray,
) -> np.ndarray:
    """
    Interpolate altitudes at sample distances along a line.
    
    Args:
        altitudes: List of altitude values at original line vertices
        line_xy: Original LineString
        sample_dists: Distances at which to interpolate
        
    Returns:
        Array of interpolated altitude values
    """
    if not altitudes or len(sample_dists) == 0:
        return np.full(len(sample_dists), np.nan)
    
    coords = np.array(line_xy.coords)
    if len(coords) != len(altitudes):
        return np.full(len(sample_dists), np.nan)
    
    # Compute cumulative distances for original vertices
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    
    # Convert altitudes to array, handling None values
    alt_array = np.array([float(a) if a is not None else np.nan for a in altitudes])
    
    # Interpolate, handling NaN values
    valid_mask = ~np.isnan(alt_array)
    if not valid_mask.any():
        return np.full(len(sample_dists), np.nan)
    
    # Use only valid points for interpolation
    valid_dists = cum_dist[valid_mask]
    valid_alts = alt_array[valid_mask]
    
    if len(valid_dists) < 2:
        return np.full(len(sample_dists), valid_alts[0] if len(valid_alts) > 0 else np.nan)
    
    return np.interp(sample_dists, valid_dists, valid_alts)


def smooth_polyline_preserve_turns(
    coords: np.ndarray,
    passes: int = 2,
    turn_deg: float = 30.0,
    neighbor_weight: float = 0.25,
) -> np.ndarray:
    """
    Smooth a polyline while preserving sharp turns.
    
    Uses weighted averaging of neighbors, but skips vertices
    at sharp turns to maintain road geometry.
    
    Args:
        coords: Array of (x, y) coordinates
        passes: Number of smoothing passes
        turn_deg: Angle threshold for detecting turns
        neighbor_weight: Weight for neighbor contribution (0-0.5)
        
    Returns:
        Smoothed coordinate array
    """
    if len(coords) <= 2 or passes <= 0:
        return coords.copy()
    
    result = coords.copy()
    
    for _ in range(passes):
        new_coords = result.copy()
        
        for i in range(1, len(result) - 1):
            # Check if this is a turn point
            h1 = bearing_from_xy(result[i-1, 0], result[i-1, 1], result[i, 0], result[i, 1])
            h2 = bearing_from_xy(result[i, 0], result[i, 1], result[i+1, 0], result[i+1, 1])
            
            if angle_diff_deg(h1, h2) >= turn_deg:
                # Preserve turn vertex exactly
                continue
            
            # Weighted average with neighbors
            center_weight = 1.0 - 2 * neighbor_weight
            new_coords[i] = (
                neighbor_weight * result[i - 1]
                + center_weight * result[i]
                + neighbor_weight * result[i + 1]
            )
        
        result = new_coords
    
    return result


def shortest_alternative_with_hop_limit(
    graph: Dict[int, List[Tuple[int, float]]],
    source: int,
    target: int,
    skip_edge: Tuple[int, int],
    max_hops: int = 4,
    max_dist: float = float("inf"),
) -> float:
    """
    Find shortest alternative path avoiding a specific edge.
    
    Used for transitive pruning: if there's a short alternative path,
    the direct edge may be redundant.
    
    Args:
        graph: Adjacency list with (neighbor, distance) tuples
        source: Start node
        target: End node
        skip_edge: Edge to avoid (u, v)
        max_hops: Maximum path length in edges
        max_dist: Maximum total distance
        
    Returns:
        Shortest alternative path length, or inf if none found
    """
    if source == target:
        return 0.0
    
    # Dijkstra with hop limit
    # State: (distance, hops, node)
    pq = [(0.0, 0, source)]
    visited: Set[Tuple[int, int]] = set()  # (node, hops)
    
    while pq:
        dist, hops, node = heapq.heappop(pq)
        
        if node == target:
            return dist
        
        if dist > max_dist:
            continue
        
        state = (node, hops)
        if state in visited:
            continue
        visited.add(state)
        
        if hops >= max_hops:
            continue
        
        for neighbor, edge_dist in graph.get(node, []):
            # Skip the forbidden edge
            if (node, neighbor) == skip_edge or (neighbor, node) == skip_edge:
                continue
            
            new_dist = dist + edge_dist
            if new_dist <= max_dist:
                heapq.heappush(pq, (new_dist, hops + 1, neighbor))
    
    return float("inf")


def stitch_centerline_paths(
    edge_support: Dict[Tuple[int, int], dict],
) -> List[List[int]]:
    """
    Stitch edges into continuous centerline paths.
    
    Uses greedy path extension to build long centerlines from
    the edge graph, prioritizing high-support edges.
    
    Args:
        edge_support: Dictionary mapping (u, v) edges to their stats
        
    Returns:
        List of node paths (each path is a list of node IDs)
    """
    if not edge_support:
        return []
    
    # Build adjacency lists
    outgoing: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    incoming: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    
    for (u, v), stats in edge_support.items():
        support = float(stats.get("weighted_support", stats.get("support", 1.0)))
        outgoing[u].append((v, support))
        incoming[v].append((u, support))
    
    # Track used edges
    used_edges: Set[Tuple[int, int]] = set()
    paths: List[List[int]] = []
    
    # Process edges in descending support order
    sorted_edges = sorted(
        edge_support.keys(),
        key=lambda e: edge_support[e].get("weighted_support", edge_support[e].get("support", 0)),
        reverse=True,
    )
    
    for start_u, start_v in sorted_edges:
        if (start_u, start_v) in used_edges:
            continue
        
        # Build path by extending in both directions
        path = [start_u, start_v]
        used_edges.add((start_u, start_v))
        
        # Extend forward from end of path
        while True:
            tail = path[-1]
            best_next = None
            best_support = 0.0
            
            for next_node, support in outgoing.get(tail, []):
                if (tail, next_node) in used_edges:
                    continue
                if next_node in path:  # Avoid cycles
                    continue
                if support > best_support:
                    best_support = support
                    best_next = next_node
            
            if best_next is None:
                break
            
            path.append(best_next)
            used_edges.add((tail, best_next))
        
        # Extend backward from start of path
        while True:
            head = path[0]
            best_prev = None
            best_support = 0.0
            
            for prev_node, support in incoming.get(head, []):
                if (prev_node, head) in used_edges:
                    continue
                if prev_node in path:
                    continue
                if support > best_support:
                    best_support = support
                    best_prev = prev_node
            
            if best_prev is None:
                break
            
            path.insert(0, best_prev)
            used_edges.add((best_prev, head))
        
        if len(path) >= 2:
            paths.append(path)
    
    return paths


def resample_polyline(coords: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample polyline to exactly n_points equidistant points.
    
    Args:
        coords: Array of (x, y) coordinates
        n_points: Target number of points
        
    Returns:
        Resampled coordinate array
    """
    if len(coords) <= 1:
        return coords.copy()
    if n_points <= 2:
        return np.asarray([coords[0], coords[-1]], dtype=np.float64)
    
    # Compute cumulative distances
    d = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    cd = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cd[-1])
    
    if total <= 0.0:
        return np.repeat(coords[:1], n_points, axis=0)
    
    # Interpolate at target distances
    target = np.linspace(0.0, total, n_points, dtype=np.float64)
    x = np.interp(target, cd, coords[:, 0])
    y = np.interp(target, cd, coords[:, 1])
    
    return np.column_stack([x, y])


def discrete_frechet_distance(a: np.ndarray, b: np.ndarray, threshold: float = float("inf")) -> float:
    """
    Compute discrete Fréchet distance between two 2D polylines.
    
    Uses iterative DP with optional early termination when the distance
    exceeds a threshold, avoiding expensive full computation.
    
    Args:
        a: First polyline as (N, 2) array
        b: Second polyline as (M, 2) array
        threshold: Early exit if distance exceeds this value
        
    Returns:
        Fréchet distance value (or value > threshold if exceeded)
    """
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    
    na, nb = len(a), len(b)
    
    # Precompute all pairwise distances at once (vectorized)
    dx = a[:, 0:1] - b[:, 0]  # (na, nb)
    dy = a[:, 1:2] - b[:, 1]  # (na, nb)
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Iterative DP (much faster than recursive Python calls)
    ca = np.empty((na, nb), dtype=np.float64)
    
    # Base case
    ca[0, 0] = dist[0, 0]
    
    # First column
    for i in range(1, na):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    
    # First row
    for j in range(1, nb):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    
    # Fill rest of DP table with early termination
    for i in range(1, na):
        row_min = float("inf")
        for j in range(1, nb):
            v = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), dist[i, j])
            ca[i, j] = v
            if v < row_min:
                row_min = v
        # If the minimum possible value in this row already exceeds threshold,
        # the final answer will too — early exit
        if row_min > threshold:
            return row_min
    
    return float(ca[na - 1, nb - 1])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted median of values.
    
    Args:
        values: Array of values
        weights: Array of weights (same length as values)
        
    Returns:
        Weighted median value
    """
    if len(values) == 0:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    
    order = np.argsort(values)
    vals = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * float(np.sum(w))
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = max(0, min(idx, len(vals) - 1))
    
    return float(vals[idx])


# ═══════════════════════════════════════════════════════════════════════════════
#  CURVE-AWARE CENTERLINE GENERATION - Fixes for Cloverleaf Interchanges
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_edge_with_traces(
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    trace_points: List[Tuple[float, float]],
    edge_length_m: float,
    min_intermediate_points: int = 2,
    max_deviation_m: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Interpolate intermediate points along an edge using nearby trace points.
    
    This fixes the "cutting chords" problem on curves by using actual trace
    point positions instead of just connecting cluster centroids.
    
    Args:
        start_xy: Start point (x, y)
        end_xy: End point (x, y)
        trace_points: List of (x, y) trace points that contributed to this edge
        edge_length_m: Length of the edge in meters
        min_intermediate_points: Minimum number of intermediate points to add
        max_deviation_m: Max distance from straight line to consider curve
        
    Returns:
        List of (x, y) coordinates forming the interpolated edge
    """
    if len(trace_points) < 3:
        return [start_xy, end_xy]
    
    # Sort trace points along the edge direction
    edge_vec = np.array([end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]])
    edge_len = np.linalg.norm(edge_vec)
    
    if edge_len < 1.0:
        return [start_xy, end_xy]
    
    edge_unit = edge_vec / edge_len
    edge_perp = np.array([-edge_unit[1], edge_unit[0]])
    
    # Project trace points onto edge and compute perpendicular distance
    projected = []
    for px, py in trace_points:
        pt = np.array([px - start_xy[0], py - start_xy[1]])
        along = float(np.dot(pt, edge_unit))
        perp_dist = float(np.dot(pt, edge_perp))
        
        # Only consider points within the edge span and not too far from line
        if 0.05 * edge_len <= along <= 0.95 * edge_len:
            projected.append((along, perp_dist, px, py))
    
    if len(projected) < 2:
        return [start_xy, end_xy]
    
    # Sort by position along edge
    projected.sort(key=lambda x: x[0])
    
    # Check if there's significant curvature (deviation from straight line)
    perp_dists = [p[1] for p in projected]
    max_dev = max(abs(min(perp_dists)), abs(max(perp_dists)))
    
    if max_dev < max_deviation_m:
        # Straight edge, no interpolation needed
        return [start_xy, end_xy]
    
    # Build interpolated path using binned trace positions
    n_bins = max(min_intermediate_points, int(edge_length_m / 15.0))  # ~15m per bin
    n_bins = min(n_bins, 10)  # Cap at 10 intermediate points
    
    bin_edges = np.linspace(0, edge_len, n_bins + 2)
    bin_points = [[] for _ in range(n_bins)]
    
    for along, perp, px, py in projected:
        bin_idx = min(int(along / edge_len * n_bins), n_bins - 1)
        bin_points[bin_idx].append((px, py))
    
    # Build interpolated path
    result = [start_xy]
    
    for bin_idx in range(n_bins):
        if bin_points[bin_idx]:
            # Use median position of points in this bin
            xs = [p[0] for p in bin_points[bin_idx]]
            ys = [p[1] for p in bin_points[bin_idx]]
            median_x = float(np.median(xs))
            median_y = float(np.median(ys))
            result.append((median_x, median_y))
        else:
            # No points in bin, interpolate linearly
            t = (bin_idx + 0.5) / n_bins
            interp_x = start_xy[0] + t * edge_vec[0]
            interp_y = start_xy[1] + t * edge_vec[1]
            result.append((interp_x, interp_y))
    
    result.append(end_xy)
    return result


def compute_curvature_at_point(
    prev_xy: Tuple[float, float],
    curr_xy: Tuple[float, float],
    next_xy: Tuple[float, float],
) -> float:
    """
    Compute curvature at a point using Menger curvature formula.
    
    Returns curvature in 1/meters (inverse of radius).
    """
    ax, ay = prev_xy
    bx, by = curr_xy
    cx, cy = next_xy
    
    # Side lengths
    a = math.sqrt((bx - cx)**2 + (by - cy)**2)  # BC
    b = math.sqrt((ax - cx)**2 + (ay - cy)**2)  # AC
    c = math.sqrt((ax - bx)**2 + (ay - by)**2)  # AB
    
    if a < 0.1 or b < 0.1 or c < 0.1:
        return 0.0
    
    # Semi-perimeter and area
    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)
    
    if area_sq <= 0:
        return 0.0
    
    area = math.sqrt(area_sq)
    
    # Menger curvature: k = 4 * Area / (a * b * c)
    curvature = 4.0 * area / (a * b * c)
    return curvature


def detect_high_curvature_zones(
    coords: np.ndarray,
    curvature_threshold: float = 0.02,  # 1/50m = cloverleaf ramp radius
    min_zone_length: int = 3,
) -> List[Tuple[int, int]]:
    """
    Detect zones of high curvature in a polyline.
    
    Used to identify sections that need careful handling (like cloverleaf ramps).
    
    Args:
        coords: Array of (x, y) coordinates
        curvature_threshold: Minimum curvature to consider "high" (1/radius)
        min_zone_length: Minimum consecutive high-curvature points
        
    Returns:
        List of (start_idx, end_idx) tuples for high-curvature zones
    """
    if len(coords) < 3:
        return []
    
    # Compute curvature at each interior point
    curvatures = np.zeros(len(coords))
    for i in range(1, len(coords) - 1):
        curvatures[i] = compute_curvature_at_point(
            tuple(coords[i-1]),
            tuple(coords[i]),
            tuple(coords[i+1])
        )
    
    # Find zones where curvature exceeds threshold
    high_curv = curvatures > curvature_threshold
    zones = []
    in_zone = False
    zone_start = 0
    
    for i, is_high in enumerate(high_curv):
        if is_high and not in_zone:
            zone_start = i
            in_zone = True
        elif not is_high and in_zone:
            if i - zone_start >= min_zone_length:
                zones.append((zone_start, i))
            in_zone = False
    
    if in_zone and len(coords) - zone_start >= min_zone_length:
        zones.append((zone_start, len(coords)))
    
    return zones


def separate_z_levels(
    points_xyz: List[Tuple[float, float, float]],
    z_separation_threshold_m: float = 3.0,
) -> List[List[int]]:
    """
    Separate points into groups by Z-level (altitude).
    
    Used to prevent conflating bridge traffic with underpass traffic
    at interchange locations.
    
    Args:
        points_xyz: List of (x, y, z) coordinates where z is altitude
        z_separation_threshold_m: Minimum Z difference to consider separate levels
        
    Returns:
        List of index groups, where each group contains points at same Z-level
    """
    if not points_xyz:
        return []
    
    # Extract Z values, handling None/nan
    z_vals = []
    valid_indices = []
    for i, (x, y, z) in enumerate(points_xyz):
        if z is not None and not math.isnan(z):
            z_vals.append(z)
            valid_indices.append(i)
    
    if len(z_vals) < 2:
        return [list(range(len(points_xyz)))]  # All in one group
    
    # Cluster by Z-level using simple hierarchical approach
    z_arr = np.array(z_vals)
    sorted_idx = np.argsort(z_arr)
    sorted_z = z_arr[sorted_idx]
    
    # Find breaks in Z values
    groups = []
    current_group = [valid_indices[sorted_idx[0]]]
    prev_z = sorted_z[0]
    
    for i in range(1, len(sorted_z)):
        if sorted_z[i] - prev_z > z_separation_threshold_m:
            # Start new group
            groups.append(current_group)
            current_group = []
        current_group.append(valid_indices[sorted_idx[i]])
        prev_z = sorted_z[i]
    
    if current_group:
        groups.append(current_group)
    
    # Add points without Z values to the largest group
    points_with_z = set(valid_indices)
    points_without_z = [i for i in range(len(points_xyz)) if i not in points_with_z]
    
    if points_without_z and groups:
        largest_group = max(groups, key=len)
        largest_group.extend(points_without_z)
    elif points_without_z:
        groups.append(points_without_z)
    
    return groups
