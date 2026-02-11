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


def discrete_frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute discrete Fréchet distance between two 2D polylines.
    
    Fréchet distance measures the maximum deviation between two curves,
    considering the order of points ("walking the dog" metaphor).
    
    Args:
        a: First polyline as (N, 2) array
        b: Second polyline as (M, 2) array
        
    Returns:
        Fréchet distance value
    """
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    
    # Dynamic programming table
    ca = np.full((len(a), len(b)), -1.0, dtype=np.float64)

    def rec(i: int, j: int) -> float:
        if ca[i, j] >= 0.0:
            return float(ca[i, j])
        
        dij = float(np.hypot(a[i, 0] - b[j, 0], a[i, 1] - b[j, 1]))
        
        if i == 0 and j == 0:
            v = dij
        elif i > 0 and j == 0:
            v = max(rec(i - 1, 0), dij)
        elif i == 0 and j > 0:
            v = max(rec(0, j - 1), dij)
        else:
            v = max(min(rec(i - 1, j), rec(i - 1, j - 1), rec(i, j - 1)), dij)
        
        ca[i, j] = v
        return v

    return rec(len(a) - 1, len(b) - 1)


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
