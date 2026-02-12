"""
Intersection-Aware Same-Direction Duplicate Cleaning

Enhanced version of cleaner.py that preserves detected intersection anchor points.

Key changes:
- Accepts pre-detected intersection points as input
- Checks if segment endpoints are near intersections before clustering
- Preserves segments that connect to intersections
- Snaps cleaned centerlines to intersection nodes

Author: Augment Code
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import MeanShift
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import all helper functions and classes from the original module
from .cleaner import (
    CleaningConfig,
    UnionFind,
    calculate_bearing,
    angle_difference,
    are_parallel,
    calculate_perpendicular_distance,
    calculate_overlap_ratio,
    calculate_sinuosity,
    calculate_segment_score,
    are_duplicate_segments,
    extract_points_from_lines,
    compute_centerline_mean_shift,
    _order_points_by_proximity,
    _compute_average_line,
    fit_bspline,
    _smooth_moving_average,
    offset_line,
    regularize_and_offset,
)


class IntersectionAwareCleaningConfig(CleaningConfig):
    """Extended configuration with intersection preservation settings"""

    # Intersection preservation
    intersection_proximity_threshold: float = 15.0  # Meters
    preserve_intersection_connections: bool = True
    snap_to_intersection_distance: float = 5.0  # Meters
    min_intersection_confidence: float = 0.3  # Only use confident intersections


def is_endpoint_near_intersection(
    point: Point, intersections_gdf: gpd.GeoDataFrame, threshold: float = 15.0
) -> bool:
    """
    Check if a point (segment endpoint) is near any intersection.

    Args:
        point: Point to check
        intersections_gdf: GeoDataFrame with intersection points
        threshold: Distance threshold in meters

    Returns:
        True if point is near an intersection
    """
    if intersections_gdf is None or len(intersections_gdf) == 0:
        return False

    # Use spatial index for efficiency
    sindex = intersections_gdf.sindex
    buffer = point.buffer(threshold)
    candidates = list(sindex.intersection(buffer.bounds))

    for idx in candidates:
        intersection_point = intersections_gdf.iloc[idx].geometry
        if point.distance(intersection_point) <= threshold:
            return True

    return False


def get_nearest_intersection(
    point: Point, intersections_gdf: gpd.GeoDataFrame, max_distance: float = 15.0
) -> Optional[Point]:
    """
    Find the nearest intersection to a point within max_distance.

    Args:
        point: Point to check
        intersections_gdf: GeoDataFrame with intersection points
        max_distance: Maximum search distance in meters

    Returns:
        Nearest intersection Point or None
    """
    if intersections_gdf is None or len(intersections_gdf) == 0:
        return None

    # Use spatial index
    sindex = intersections_gdf.sindex
    buffer = point.buffer(max_distance)
    candidates = list(sindex.intersection(buffer.bounds))

    if not candidates:
        return None

    # Find closest
    min_dist = float("inf")
    nearest = None

    for idx in candidates:
        intersection_point = intersections_gdf.iloc[idx].geometry
        dist = point.distance(intersection_point)
        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            nearest = intersection_point

    return nearest


def snap_endpoints_to_intersections(
    line: LineString, intersections_gdf: gpd.GeoDataFrame, snap_distance: float = 5.0
) -> LineString:
    """
    Snap line endpoints to nearby intersections.

    Args:
        line: LineString to snap
        intersections_gdf: GeoDataFrame with intersection points
        snap_distance: Distance threshold for snapping

    Returns:
        LineString with snapped endpoints
    """
    if intersections_gdf is None or len(intersections_gdf) == 0:
        return line

    coords = list(line.coords)

    # Check start point
    start_point = Point(coords[0])
    nearest_start = get_nearest_intersection(
        start_point, intersections_gdf, snap_distance
    )
    if nearest_start:
        coords[0] = nearest_start.coords[0]

    # Check end point
    end_point = Point(coords[-1])
    nearest_end = get_nearest_intersection(end_point, intersections_gdf, snap_distance)
    if nearest_end:
        coords[-1] = nearest_end.coords[0]

    return LineString(coords)


def should_preserve_segment(
    geom: LineString,
    intersections_gdf: gpd.GeoDataFrame,
    config: IntersectionAwareCleaningConfig,
) -> bool:
    """
    Determine if a segment should be preserved due to intersection connectivity.

    Args:
        geom: LineString to check
        intersections_gdf: GeoDataFrame with intersection points
        config: Configuration

    Returns:
        True if segment should be preserved
    """
    if not config.preserve_intersection_connections:
        return False

    if intersections_gdf is None or len(intersections_gdf) == 0:
        return False

    # Check both endpoints
    start_point = Point(geom.coords[0])
    end_point = Point(geom.coords[-1])

    start_near = is_endpoint_near_intersection(
        start_point, intersections_gdf, config.intersection_proximity_threshold
    )
    end_near = is_endpoint_near_intersection(
        end_point, intersections_gdf, config.intersection_proximity_threshold
    )

    return start_near or end_near


def cluster_parallel_segments_intersection_aware(
    gdf: gpd.GeoDataFrame,
    config: IntersectionAwareCleaningConfig,
    intersections_gdf: Optional[gpd.GeoDataFrame] = None,
) -> List[Set[int]]:
    """
    Cluster parallel same-direction segments while respecting intersections.

    Modified version of cluster_parallel_segments() that:
    - Identifies intersection-protected segments
    - Prevents them from being clustered/merged

    Args:
        gdf: GeoDataFrame with road segments
        config: Configuration
        intersections_gdf: Optional GeoDataFrame with intersection points

    Returns:
        List of clusters (sets of indices)
    """
    print("\n" + "=" * 60)
    print("Step 1: Clustering Parallel Same-Direction Segments")
    print("=" * 60)

    # Identify protected segments
    intersection_protected = set()
    if config.preserve_intersection_connections and intersections_gdf is not None:
        print("Identifying intersection-connected segments...")
        for idx in tqdm(range(len(gdf)), desc="Checking intersection connectivity"):
            geom = gdf.geometry.iloc[idx]
            if should_preserve_segment(geom, intersections_gdf, config):
                intersection_protected.add(idx)

        print(
            f"Protected segments (intersection-connected): {len(intersection_protected)}"
        )

    # Use Union-Find for efficient clustering
    uf = UnionFind(len(gdf))

    # Build spatial index
    sindex = gdf.sindex

    # Track duplicate pairs
    duplicate_pairs = []

    print(f"Analyzing {len(gdf)} segments for same-direction duplicates...")

    for idx in tqdm(range(len(gdf)), desc="Finding duplicate pairs"):
        # Skip protected segments
        if idx in intersection_protected:
            continue

        geom = gdf.geometry.iloc[idx]

        if geom.is_empty or geom is None:
            continue

        # Filter by minimum length
        if geom.length < config.min_segment_length:
            continue

        # Get bearing
        coords = list(geom.coords)
        bearing1 = calculate_bearing(coords)

        # Find nearby candidates using spatial index
        bbox = geom.buffer(config.search_distance).bounds
        candidates = list(sindex.intersection(bbox))

        for cand_idx in candidates:
            # Skip self, already processed, and protected segments
            if cand_idx <= idx or cand_idx in intersection_protected:
                continue

            cand_geom = gdf.geometry.iloc[cand_idx]

            if cand_geom.is_empty or cand_geom is None:
                continue

            # Filter by minimum length
            if cand_geom.length < config.min_segment_length:
                continue

            # Check if they are duplicate segments (same direction, parallel, nearby)
            is_duplicate, perp_dist, overlap = are_duplicate_segments(
                geom, cand_geom, bearing1, config
            )

            if is_duplicate:
                # Union them in the clustering structure
                uf.union(idx, cand_idx)
                duplicate_pairs.append((idx, cand_idx, perp_dist, overlap))

    print(f"Found {len(duplicate_pairs)} duplicate pairs")

    # Extract clusters from union-find
    cluster_map: Dict[int, Set[int]] = {}
    for idx in range(len(gdf)):
        root = uf.find(idx)
        if root not in cluster_map:
            cluster_map[root] = set()
        cluster_map[root].add(idx)

    # Filter clusters (only keep clusters with 2+ members)
    clusters = [cluster for cluster in cluster_map.values() if len(cluster) >= 2]

    print(f"Clusters formed: {len(clusters)}")

    # Statistics
    cluster_sizes = [len(c) for c in clusters]
    if cluster_sizes:
        print(
            f"Cluster size stats: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes) / len(cluster_sizes):.1f}"
        )

    return clusters


def clean_double_digitized_intersection_aware(
    gdf: gpd.GeoDataFrame,
    intersections_gdf: Optional[gpd.GeoDataFrame] = None,
    config: Optional[IntersectionAwareCleaningConfig] = None,
) -> gpd.GeoDataFrame:
    """
    Main function to clean double-digitized road geometries while preserving intersections.

    Args:
        gdf: Input GeoDataFrame with road geometries
        intersections_gdf: Optional GeoDataFrame with pre-detected intersection points
        config: Optional cleaning configuration

    Returns:
        Cleaned GeoDataFrame with deduplicated road network and preserved intersections
    """
    if config is None:
        config = IntersectionAwareCleaningConfig()

    print("=" * 60)
    print("Intersection-Aware Same-Direction Duplicate Cleaning")
    print("=" * 60)
    print(f"Input: {len(gdf)} geometries")

    if intersections_gdf is not None:
        print(f"Intersections: {len(intersections_gdf)} detected")

        # Filter by confidence if available
        if "confidence" in intersections_gdf.columns:
            before = len(intersections_gdf)
            intersections_gdf = intersections_gdf[
                intersections_gdf["confidence"] >= config.min_intersection_confidence
            ].copy()
            print(
                f"High-confidence intersections: {len(intersections_gdf)} (filtered from {before})"
            )
    else:
        print("No intersections provided - running without intersection preservation")

    # Store original CRS
    original_crs = gdf.crs
    input_count = len(gdf)

    # Ensure both datasets are in same CRS
    if original_crs is not None and original_crs.to_epsg() != config.target_epsg:
        print(f"Projecting to EPSG:{config.target_epsg} for metric calculations...")
        gdf = gdf.to_crs(epsg=config.target_epsg)
        if intersections_gdf is not None:
            intersections_gdf = intersections_gdf.to_crs(epsg=config.target_epsg)

    # Filter out invalid geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.reset_index(drop=True)

    # Step 1: Cluster parallel segments (intersection-aware)
    clusters = cluster_parallel_segments_intersection_aware(
        gdf, config, intersections_gdf
    )

    # Track which indices are in clusters
    clustered_indices = set()
    for cluster in clusters:
        clustered_indices.update(cluster)

    # Process clusters
    new_geometries = []
    cluster_info = []

    print("\n" + "=" * 60)
    print("Step 2: Computing Centerlines for Clusters")
    print("=" * 60)

    for i, cluster in enumerate(tqdm(clusters, desc="Processing clusters")):
        cluster_list = list(cluster)
        cluster_geoms = [gdf.geometry.iloc[idx] for idx in cluster_list]

        # For small clusters (≤3 segments), just pick the best one
        if len(cluster_list) <= 3:
            # Calculate scores for each segment
            scores = []
            for idx in cluster_list:
                score = calculate_segment_score(gdf.geometry.iloc[idx], gdf, idx)
                scores.append(score)

            # Select the segment with the highest score
            best_idx = cluster_list[np.argmax(scores)]
            centerline = gdf.geometry.iloc[best_idx]
        else:
            # For larger clusters, use Mean Shift
            centerline = compute_centerline_mean_shift(cluster_geoms, config)

        if centerline is not None and not centerline.is_empty:
            new_geometries.append(centerline)
            cluster_info.append(
                {
                    "cluster_id": i,
                    "cluster_size": len(cluster_list),
                    "length": centerline.length,
                }
            )

    print(f"Computed {len(new_geometries)} centerlines")

    # Step 3: Keep unclustered segments (including intersection-protected ones)
    unclustered_indices = [i for i in range(len(gdf)) if i not in clustered_indices]
    unclustered_geoms = [gdf.geometry.iloc[i] for i in unclustered_indices]

    print(f"Unclustered segments preserved: {len(unclustered_geoms)}")

    # Combine all geometries
    all_geometries = new_geometries + unclustered_geoms

    # Build output GeoDataFrame
    output_gdf = gpd.GeoDataFrame({"geometry": all_geometries}, crs=gdf.crs)

    # Step 4: Snap endpoints to intersections
    if config.preserve_intersection_connections and intersections_gdf is not None:
        print("\n" + "=" * 60)
        print("Step 3: Snapping Endpoints to Intersections")
        print("=" * 60)

        snapped_geoms = []
        for geom in tqdm(output_gdf.geometry, desc="Snapping to intersections"):
            snapped = snap_endpoints_to_intersections(
                geom, intersections_gdf, config.snap_to_intersection_distance
            )
            snapped_geoms.append(snapped)

        output_gdf.geometry = snapped_geoms

    # Reset index
    output_gdf = output_gdf.reset_index(drop=True)

    # Convert back to original CRS
    if original_crs:
        output_gdf = output_gdf.to_crs(original_crs)

    print("\n" + "=" * 60)
    print(f"Input segments: {input_count}")
    print(f"Output segments: {len(output_gdf)}")
    print(
        f"Reduction: {input_count - len(output_gdf)} ({100 * (1 - len(output_gdf) / input_count):.1f}%)"
    )
    print("=" * 60)

    return output_gdf
