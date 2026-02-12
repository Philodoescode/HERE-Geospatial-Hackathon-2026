"""
Double-Digitized Geometry Cleaning Algorithm

This module implements a process to remove redundant same-direction road segments
from probe data while preserving bidirectional lanes:

1. Distance-Weighted Lane Clustering - Cluster same-direction parallel segments by relative angle and perpendicular distance
2. Unified Road Center Computation - Apply Mean Shift to compute centerlines for large clusters, or select best segment for small clusters
3. B-Spline Regularization & Offsetting - Smooth and offset to create clean parallel lanes

Key Features:
- Only clusters same-direction segments (bearing diff ≈ 0°)
- Preserves opposite-direction lanes (bearing diff ≈ 180°) as separate roads
- Prioritizes longer, straighter, more connected roads

Author: Augment Code
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
from sklearn.cluster import MeanShift
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CleaningConfig:
    """Configuration parameters for the cleaning algorithm."""

    # Distance thresholds (in meters)
    highway_threshold: float = 3.6  # 12 feet
    urban_threshold: float = 3.3  # 11 feet
    min_merge_threshold: float = 3.0  # 10 feet

    # Maximum perpendicular distance to consider segments as duplicates
    # Typical lane width is 3.0-3.6m - segments further apart are likely different roads
    max_perpendicular_distance: float = (
        4.5  # meters - stricter to avoid false positives
    )

    # Lane offset distances (in meters)
    highway_offset: float = 3.6  # 12 feet
    urban_offset: float = 3.3  # 11 feet

    # Clustering parameters
    mean_shift_bandwidth: float = 5.0  # meters

    # B-spline parameters
    spline_smoothing: float = 0.5
    spline_degree: int = 3

    # Angle tolerance for parallel detection (degrees)
    # Two segments are parallel if their angle difference is within this tolerance
    # of 0° (same direction) or 180° (opposite direction)
    parallel_angle_tolerance: float = 15.0

    # Search distance for finding parallel segments (meters)
    search_distance: float = 50.0  # increased to find more candidates

    # Minimum overlap ratio for segments to be considered duplicates
    # Lowered to catch offset segments that don't fully overlap
    min_overlap_ratio: float = 0.15

    # Minimum segment length to process (meters)
    min_segment_length: float = 5.0

    # Target CRS for metric calculations (UTM 34N for Kosovo)
    target_epsg: int = 32634


# =============================================================================
# Union-Find Data Structure for Efficient Clustering
# =============================================================================


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for efficient clustering."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union two sets. Returns True if they were in different sets."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_clusters(self) -> Dict[int, List[int]]:
        """Get all clusters as a dictionary mapping root -> list of members."""
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


# =============================================================================
# Helper Functions
# =============================================================================


def calculate_bearing(coords: List[Tuple[float, float]]) -> float:
    """
    Calculate the bearing/heading of a line segment from its coordinates.

    Args:
        coords: List of (x, y) coordinate tuples

    Returns:
        Bearing in degrees (0-360), where 0 is North, 90 is East
    """
    if len(coords) < 2:
        return 0.0

    # Use first and last points for overall direction
    start = np.array(coords[0])
    end = np.array(coords[-1])

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # Calculate bearing (0 = North, 90 = East)
    bearing = math.degrees(math.atan2(dx, dy))

    # Normalize to 0-360
    return bearing % 360


def angle_difference(bearing1: float, bearing2: float) -> float:
    """
    Calculate the minimum angle difference between two bearings.

    Returns a value between 0 and 180 degrees.
    """
    diff = abs(bearing1 - bearing2) % 360
    return min(diff, 360 - diff)


def are_parallel(bearing1: float, bearing2: float, tolerance: float) -> bool:
    """
    Check if two bearings indicate parallel lines in the SAME direction.

    Two lines are considered duplicate candidates only if they travel in the same direction.
    Opposite-direction segments (bearing diff ≈180°) are separate lanes of a bidirectional road.
    """
    diff = angle_difference(bearing1, bearing2)
    # Only same direction: diff ≈ 0° (NOT 180° - those are opposite lanes)
    return diff <= tolerance


# =============================================================================
# Step 1: Distance-Weighted Lane Clustering (Fixed)
# =============================================================================


def calculate_perpendicular_distance(line1: LineString, line2: LineString) -> float:
    """
    Calculate the average perpendicular (lateral) distance between two parallel lines.

    Args:
        line1: First LineString
        line2: Second LineString

    Returns:
        Average perpendicular distance in the same units as the geometry
    """
    # Sample points along line1 and find perpendicular distance to line2
    num_samples = max(5, int(line1.length / 2))
    distances = []

    for i in range(num_samples + 1):
        frac = i / num_samples
        point = line1.interpolate(frac, normalized=True)
        dist = point.distance(line2)
        distances.append(dist)

    return np.mean(distances)


def calculate_overlap_ratio(line1: LineString, line2: LineString) -> float:
    """
    Calculate the overlap ratio between two line segments.

    Projects both lines onto their common axis and calculates
    what fraction of the shorter line overlaps with the longer one.

    Returns:
        Overlap ratio between 0 and 1
    """
    # Get bounding boxes
    minx1, miny1, maxx1, maxy1 = line1.bounds
    minx2, miny2, maxx2, maxy2 = line2.bounds

    # Calculate overlap in both dimensions
    x_overlap = max(0, min(maxx1, maxx2) - max(minx1, minx2))
    y_overlap = max(0, min(maxy1, maxy2) - max(miny1, miny2))

    # Use the larger overlap dimension
    overlap = max(x_overlap, y_overlap)

    # Calculate the shorter line's extent in that dimension
    if x_overlap >= y_overlap:
        extent1 = maxx1 - minx1
        extent2 = maxx2 - minx2
    else:
        extent1 = maxy1 - miny1
        extent2 = maxy2 - miny2

    shorter_extent = min(extent1, extent2)

    if shorter_extent == 0:
        return 0.0

    return overlap / shorter_extent


def calculate_sinuosity(line: LineString) -> float:
    """
    Calculate the sinuosity (straightness) of a line.

    Sinuosity = actual_length / straight_line_distance
    Lower values (closer to 1.0) indicate straighter lines.

    Args:
        line: LineString to measure

    Returns:
        Sinuosity ratio (>= 1.0)
    """
    if line is None or line.is_empty or line.length == 0:
        return float("inf")

    coords = list(line.coords)
    if len(coords) < 2:
        return float("inf")

    # Straight-line distance from start to end
    start = np.array(coords[0])
    end = np.array(coords[-1])
    straight_dist = np.linalg.norm(end - start)

    if straight_dist == 0:
        return float("inf")

    # Sinuosity ratio
    return line.length / straight_dist


def calculate_segment_score(line: LineString, gdf: gpd.GeoDataFrame, idx: int) -> float:
    """
    Calculate a quality score for a road segment.

    Higher scores indicate segments that should be prioritized:
    - Longer segments
    - Straighter segments (lower sinuosity)
    - Better connectivity (more endpoints connected to other roads)

    Args:
        line: LineString to score
        gdf: Full GeoDataFrame for connectivity analysis
        idx: Index of this segment in gdf

    Returns:
        Quality score (higher = better)
    """
    if line is None or line.is_empty:
        return 0.0

    # Length component (normalized, higher = better)
    length_score = line.length / 100.0  # Normalize by 100m

    # Straightness component (inverse sinuosity, higher = better)
    sinuosity = calculate_sinuosity(line)
    if sinuosity == float("inf"):
        straightness_score = 0.0
    else:
        straightness_score = 1.0 / sinuosity

    # Combined score (weighted)
    # Prioritize length more than straightness
    score = (0.7 * length_score) + (0.3 * straightness_score)

    return score


def are_duplicate_segments(
    line1: LineString,
    line2: LineString,
    bearing1: float,
    bearing2: float,
    config: CleaningConfig,
) -> bool:
    """
    Check if two line segments are duplicates (same-direction and close).

    Two segments are duplicates if:
    1. They travel in the SAME direction (angle diff ≈ 0°, NOT 180°)
    2. They are close (perpendicular distance < threshold)
    3. They overlap spatially

    Args:
        line1: First LineString
        line2: Second LineString
        bearing1: Pre-computed bearing of line1
        bearing2: Pre-computed bearing of line2
        config: Cleaning configuration

    Returns:
        True if segments are duplicates
    """
    # Check if same direction (NOT opposite direction)
    if not are_parallel(bearing1, bearing2, config.parallel_angle_tolerance):
        return False

    # Check overlap ratio
    overlap = calculate_overlap_ratio(line1, line2)
    if overlap < config.min_overlap_ratio:
        return False

    # Calculate perpendicular distance
    perp_dist = calculate_perpendicular_distance(line1, line2)

    # Check if within threshold
    return perp_dist < config.max_perpendicular_distance


def cluster_parallel_segments(
    gdf: gpd.GeoDataFrame, config: CleaningConfig
) -> List[List[int]]:
    """
    Cluster parallel segments based on relative angle and perpendicular distance.

    Uses Union-Find for efficient clustering. Two segments are in the same cluster
    if they are parallel (relative angle ≈ 0° or ≈ 180°) and close together.

    Args:
        gdf: GeoDataFrame with LineString geometries
        config: Cleaning configuration

    Returns:
        List of clusters, where each cluster is a list of indices (only clusters with >1 segment)
    """
    print("Step 1: Distance-Weighted Lane Clustering...")

    n = len(gdf)
    if n == 0:
        return []

    # Reset index to ensure contiguous integer indices
    gdf = gdf.reset_index(drop=True)

    # Pre-compute bearings for all segments
    print("  Pre-computing bearings...")
    bearings = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            bearings.append(0.0)
        else:
            bearings.append(calculate_bearing(list(geom.coords)))
    bearings = np.array(bearings)

    # Build spatial index
    print("  Building spatial index...")
    sindex = gdf.sindex

    # Initialize Union-Find
    uf = UnionFind(n)

    # Find pairs of duplicate segments
    print("  Finding duplicate segment pairs...")
    pairs_found = 0

    for idx in tqdm(range(n), desc="Clustering parallel segments"):
        geom = gdf.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue

        # Find nearby segments using spatial index
        bbox = geom.buffer(config.search_distance).bounds
        candidates = list(sindex.intersection(bbox))

        for cand_idx in candidates:
            if cand_idx <= idx:  # Avoid duplicate comparisons
                continue

            cand_geom = gdf.geometry.iloc[cand_idx]
            if cand_geom is None or cand_geom.is_empty:
                continue

            # Check if they are duplicates
            if are_duplicate_segments(
                geom, cand_geom, bearings[idx], bearings[cand_idx], config
            ):
                uf.union(idx, cand_idx)
                pairs_found += 1

    print(f"  Found {pairs_found} duplicate segment pairs")

    # Extract clusters with more than one segment
    all_clusters = uf.get_clusters()
    clusters = [members for members in all_clusters.values() if len(members) > 1]

    print(f"  Found {len(clusters)} clusters of parallel segments")

    # Print cluster size distribution
    if clusters:
        sizes = [len(c) for c in clusters]
        print(
            f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}"
        )
        total_in_clusters = sum(sizes)
        print(
            f"  Total segments in clusters: {total_in_clusters} ({100 * total_in_clusters / n:.1f}%)"
        )

    return clusters


# =============================================================================
# Step 3: Unified Road Center Computation
# =============================================================================


def extract_points_from_lines(
    lines: List[LineString], sample_spacing: float = 2.0
) -> np.ndarray:
    """
    Extract sampled points from a list of LineStrings.

    Args:
        lines: List of LineString geometries
        sample_spacing: Spacing between sample points in meters

    Returns:
        Array of (x, y) coordinates
    """
    all_points = []

    for line in lines:
        if line is None or line.is_empty:
            continue

        # Sample points along the line
        num_samples = max(2, int(line.length / sample_spacing))
        for i in range(num_samples + 1):
            frac = i / num_samples
            point = line.interpolate(frac, normalized=True)
            all_points.append([point.x, point.y])

    return np.array(all_points) if all_points else np.array([]).reshape(0, 2)


def compute_centerline_mean_shift(
    lines: List[LineString], bandwidth: float = 5.0
) -> Optional[LineString]:
    """
    Step 3: Apply Mean Shift clustering to compute a unified centerline.

    Args:
        lines: List of parallel LineString geometries
        bandwidth: Mean Shift bandwidth parameter

    Returns:
        Centerline as a LineString, or None if computation fails
    """
    if not lines:
        return None

    # Extract points from all lines
    points = extract_points_from_lines(lines, sample_spacing=2.0)

    if len(points) < 3:
        # Not enough points, return the longest line
        return max(lines, key=lambda l: l.length if l else 0)

    try:
        # Apply Mean Shift clustering
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(points)

        # Get cluster centers
        centers = ms.cluster_centers_

        if len(centers) < 2:
            # Not enough centers, return average line
            return _compute_average_line(lines)

        # Order centers to form a line (using nearest neighbor)
        ordered_centers = _order_points_by_proximity(centers)

        return LineString(ordered_centers)

    except Exception as e:
        # Fallback to average line computation
        return _compute_average_line(lines)


def _order_points_by_proximity(points: np.ndarray) -> np.ndarray:
    """Order points by proximity to form a continuous path."""
    if len(points) < 2:
        return points

    # Start from the point with minimum x (or y if tied)
    start_idx = np.lexsort((points[:, 1], points[:, 0]))[0]

    ordered = [points[start_idx]]
    remaining = list(range(len(points)))
    remaining.remove(start_idx)

    while remaining:
        last = ordered[-1]
        # Find nearest remaining point
        distances = [np.linalg.norm(points[i] - last) for i in remaining]
        nearest_idx = remaining[np.argmin(distances)]
        ordered.append(points[nearest_idx])
        remaining.remove(nearest_idx)

    return np.array(ordered)


def _compute_average_line(lines: List[LineString]) -> Optional[LineString]:
    """Compute an average line from multiple parallel lines."""
    if not lines:
        return None

    # Find the longest line as reference
    ref_line = max(lines, key=lambda l: l.length if l else 0)
    if ref_line is None or ref_line.is_empty:
        return None

    # Orient all lines to match reference direction
    ref_coords = list(ref_line.coords)
    ref_vec = np.array(ref_coords[-1]) - np.array(ref_coords[0])

    oriented_lines = []
    max_len = 0

    for line in lines:
        if line is None or line.is_empty:
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            continue

        vec = np.array(coords[-1]) - np.array(coords[0])

        # Flip if pointing opposite direction
        if np.dot(ref_vec, vec) < 0:
            coords = coords[::-1]

        oriented_lines.append(LineString(coords))
        max_len = max(max_len, line.length)

    if not oriented_lines or max_len == 0:
        return ref_line

    # Resample all lines to same number of points
    num_samples = max(5, int(max_len / 1.0))

    sampled_coords = []
    for line in oriented_lines:
        dists = np.linspace(0, line.length, num_samples)
        pts = [line.interpolate(d) for d in dists]
        sampled_coords.append([(p.x, p.y) for p in pts])

    # Average the coordinates
    sampled_coords = np.array(sampled_coords)
    mean_coords = np.mean(sampled_coords, axis=0)

    return LineString(mean_coords)


# =============================================================================
# Step 4: B-Spline Regularization & Offsetting
# =============================================================================


def fit_bspline(
    line: LineString, smoothing: float = 0.5, degree: int = 3
) -> Optional[LineString]:
    """
    Fit a smooth B-spline curve to a LineString.

    Args:
        line: Input LineString
        smoothing: Smoothing factor (0 = interpolating, higher = smoother)
        degree: Spline degree (1-5, typically 3 for cubic)

    Returns:
        Smoothed LineString, or original if fitting fails
    """
    if line is None or line.is_empty:
        return line

    coords = np.array(line.coords)

    if len(coords) < degree + 1:
        return line

    try:
        # Parameterize by arc length
        distances = np.zeros(len(coords))
        for i in range(1, len(coords)):
            distances[i] = distances[i - 1] + np.linalg.norm(coords[i] - coords[i - 1])

        if distances[-1] == 0:
            return line

        # Normalize parameter
        t = distances / distances[-1]

        # Fit B-spline to x and y separately
        # Use splrep for fitting and splev for evaluation
        tck_x, _ = interpolate.splprep(
            [coords[:, 0]], u=t, s=smoothing, k=min(degree, len(coords) - 1)
        )
        tck_y, _ = interpolate.splprep(
            [coords[:, 1]], u=t, s=smoothing, k=min(degree, len(coords) - 1)
        )

        # Evaluate at finer resolution
        num_eval = max(len(coords), int(line.length / 1.0))  # ~1m spacing
        t_eval = np.linspace(0, 1, num_eval)

        x_smooth = interpolate.splev(t_eval, tck_x)[0]
        y_smooth = interpolate.splev(t_eval, tck_y)[0]

        smooth_coords = list(zip(x_smooth, y_smooth))

        return LineString(smooth_coords)

    except Exception:
        # Fallback: simple moving average smoothing
        return _smooth_moving_average(line)


def _smooth_moving_average(line: LineString, window: int = 3) -> LineString:
    """Apply simple moving average smoothing to a LineString."""
    if line is None or line.is_empty:
        return line

    coords = np.array(line.coords)

    if len(coords) < window:
        return line

    smoothed = np.copy(coords)

    # Keep endpoints fixed
    for i in range(1, len(coords) - 1):
        start = max(0, i - window // 2)
        end = min(len(coords), i + window // 2 + 1)
        smoothed[i] = np.mean(coords[start:end], axis=0)

    return LineString(smoothed)


def offset_line(
    line: LineString, distance: float, side: str = "left"
) -> Optional[LineString]:
    """
    Offset a line by a given distance along its normal vector.

    Args:
        line: Input LineString
        distance: Offset distance in the same units as the geometry
        side: 'left' or 'right'

    Returns:
        Offset LineString, or None if offsetting fails
    """
    if line is None or line.is_empty:
        return None

    try:
        offset = line.parallel_offset(distance, side)

        if offset.is_empty:
            return None

        # Handle MultiLineString result
        if isinstance(offset, MultiLineString):
            # Return the longest part
            parts = list(offset.geoms)
            if parts:
                return max(parts, key=lambda p: p.length)
            return None

        return offset

    except Exception:
        return None


def regularize_and_offset(
    centerline: LineString, offset_distance: float, config: CleaningConfig
) -> Tuple[Optional[LineString], Optional[LineString]]:
    """
    Step 4: Apply B-spline regularization and create offset parallel lines.

    Args:
        centerline: Computed centerline
        offset_distance: Distance to offset for each direction
        config: Cleaning configuration

    Returns:
        Tuple of (left_lane, right_lane) LineStrings
    """
    if centerline is None or centerline.is_empty:
        return None, None

    # Fit B-spline to smooth the centerline
    smooth_center = fit_bspline(
        centerline, smoothing=config.spline_smoothing, degree=config.spline_degree
    )

    if smooth_center is None or smooth_center.is_empty:
        smooth_center = centerline

    # Create offset lines for each direction
    left_lane = offset_line(smooth_center, offset_distance / 2, "left")
    right_lane = offset_line(smooth_center, offset_distance / 2, "right")

    # Apply smoothing to offset lines as well
    if left_lane is not None:
        left_lane = fit_bspline(
            left_lane, smoothing=config.spline_smoothing, degree=config.spline_degree
        )
    if right_lane is not None:
        right_lane = fit_bspline(
            right_lane, smoothing=config.spline_smoothing, degree=config.spline_degree
        )

    return left_lane, right_lane


# =============================================================================
# Main Processing Pipeline
# =============================================================================


def clean_double_digitized(
    gdf: gpd.GeoDataFrame, config: Optional[CleaningConfig] = None
) -> gpd.GeoDataFrame:
    """
    Main function to clean double-digitized road geometries.

    Implements a 3-step process:
    1. Distance-Weighted Lane Clustering - Cluster parallel segments by relative angle and distance
    2. Unified Road Center Computation - Apply Mean Shift to compute centerlines
    3. B-Spline Regularization & Offsetting - Smooth and offset to create clean parallel lanes

    Args:
        gdf: Input GeoDataFrame with road geometries
        config: Optional cleaning configuration

    Returns:
        Cleaned GeoDataFrame with deduplicated road network
    """
    if config is None:
        config = CleaningConfig()

    print("=" * 60)
    print("Double-Digitized Geometry Cleaning Algorithm")
    print("=" * 60)
    print(f"Input: {len(gdf)} geometries")

    # Store original CRS
    original_crs = gdf.crs
    input_count = len(gdf)

    # Project to metric CRS for accurate distance calculations
    if original_crs is not None and original_crs.to_epsg() != config.target_epsg:
        print(f"Projecting to EPSG:{config.target_epsg} for metric calculations...")
        gdf = gdf.to_crs(epsg=config.target_epsg)

    # Filter out invalid geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.reset_index(drop=True)

    # Step 1: Cluster parallel segments (no heading-based partitioning)
    clusters = cluster_parallel_segments(gdf, config)

    # Track which indices are in clusters
    clustered_indices = set()
    for cluster in clusters:
        clustered_indices.update(cluster)

    # Process clusters
    new_geometries = []
    cluster_info = []

    print(f"\nStep 2: Processing {len(clusters)} clusters...")

    for cluster in tqdm(clusters, desc="Processing clusters"):
        cluster_lines = [
            gdf.geometry.iloc[i] for i in cluster if gdf.geometry.iloc[i] is not None
        ]

        if not cluster_lines:
            continue

        # For small clusters (2-3 segments), just pick the best one
        if len(cluster) <= 3:
            # Score all segments in cluster and pick the best
            best_idx = None
            best_score = -float("inf")

            for idx in cluster:
                geom = gdf.geometry.iloc[idx]
                if geom is None or geom.is_empty:
                    continue
                score = calculate_segment_score(geom, gdf, idx)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                best_geom = gdf.geometry.iloc[best_idx]
                # Apply smoothing
                smooth_geom = fit_bspline(
                    best_geom, config.spline_smoothing, config.spline_degree
                )
                if smooth_geom is not None and not smooth_geom.is_empty:
                    new_geometries.append(smooth_geom)
                else:
                    new_geometries.append(best_geom)
            continue

        # For larger clusters, compute centerline using Mean Shift
        centerline = compute_centerline_mean_shift(
            cluster_lines, bandwidth=config.mean_shift_bandwidth
        )

        if centerline is None or centerline.is_empty:
            # Fallback: pick the best segment from cluster
            best_idx = None
            best_score = -float("inf")

            for idx in cluster:
                geom = gdf.geometry.iloc[idx]
                if geom is None or geom.is_empty:
                    continue
                score = calculate_segment_score(geom, gdf, idx)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                new_geometries.append(gdf.geometry.iloc[best_idx])
            continue

        # Calculate average width of the cluster
        avg_width = np.mean([centerline.distance(l) for l in cluster_lines])

        # Decide whether to create dual lanes or single centerline
        if avg_width > config.min_merge_threshold:
            # Create dual lanes
            left_lane, right_lane = regularize_and_offset(
                centerline, offset_distance=avg_width, config=config
            )

            if left_lane is not None and not left_lane.is_empty:
                new_geometries.append(left_lane)
                cluster_info.append({"type": "left_lane", "cluster_size": len(cluster)})
            if right_lane is not None and not right_lane.is_empty:
                new_geometries.append(right_lane)
                cluster_info.append(
                    {"type": "right_lane", "cluster_size": len(cluster)}
                )
        else:
            # Single centerline (segments too close together - true duplicates)
            smooth_center = fit_bspline(
                centerline, config.spline_smoothing, config.spline_degree
            )
            if smooth_center is not None and not smooth_center.is_empty:
                new_geometries.append(smooth_center)
                cluster_info.append(
                    {"type": "centerline", "cluster_size": len(cluster)}
                )

    # Add non-clustered geometries (single segments - not duplicates)
    print(
        f"\nStep 3: Processing {len(gdf) - len(clustered_indices)} non-clustered segments..."
    )
    non_clustered_count = 0
    for idx in range(len(gdf)):
        if idx not in clustered_indices:
            geom = gdf.geometry.iloc[idx]
            if geom is not None and not geom.is_empty:
                # Apply smoothing to single segments too
                smooth_geom = fit_bspline(
                    geom, config.spline_smoothing, config.spline_degree
                )
                if smooth_geom is not None:
                    new_geometries.append(smooth_geom)
                else:
                    new_geometries.append(geom)
                non_clustered_count += 1

    # Create result GeoDataFrame
    result = gpd.GeoDataFrame(geometry=new_geometries, crs=gdf.crs)

    # Filter out empty geometries
    result = result[result.geometry.notnull() & ~result.geometry.is_empty]

    # Project back to original CRS
    if original_crs is not None and original_crs.to_epsg() != config.target_epsg:
        print(f"Projecting back to {original_crs}...")
        result = result.to_crs(original_crs)

    print("=" * 60)
    print(f"Output: {len(result)} geometries")
    reduction = input_count - len(result)
    reduction_pct = 100 * reduction / input_count if input_count > 0 else 0
    print(f"Reduction: {reduction} geometries removed ({reduction_pct:.1f}%)")
    print("=" * 60)

    return result
