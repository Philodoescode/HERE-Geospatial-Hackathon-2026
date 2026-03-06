"""
Intersection-Aware Bidirectional Lane Deduplication

This module is an enhanced version of opposite_direction_dedup.py that preserves
detected intersection anchor points during cleaning operations.

Key changes:
- Accepts pre-detected intersection points as input
- Checks if segment endpoints are near intersections before merging
- Preserves segments that connect to intersections
- Ensures merged/offset roads properly connect to intersection nodes

Author: Augment Code
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
from scipy import interpolate
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from sklearn.cluster import MeanShift, DBSCAN
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import all helper functions from the original module
from .opposite_direction_dedup import (
    OppositeDirectionConfig,
    calculate_azimuth,
    get_direction_vector,
    calculate_cosine_similarity,
    calculate_perpendicular_distance,
    check_direction_consistency,
    calculate_overlap_ratio,
    validate_opposite_pair,
    normalize_azimuth_to_0_180,
    group_by_direction_bins,
    determine_road_direction_by_voting,
    get_majority_direction_indices,
    compute_centerline,
    _order_points_by_proximity,
    create_offset_lines,
)


class IntersectionAwareConfig(OppositeDirectionConfig):
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
    config: IntersectionAwareConfig,
) -> bool:
    """
    Determine if a segment should be preserved due to intersection connectivity.

    A segment is preserved if:
    - Either endpoint is near an intersection
    - And preserve_intersection_connections is True

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


def deduplicate_opposite_directions_intersection_aware(
    gdf: gpd.GeoDataFrame,
    intersections_gdf: Optional[gpd.GeoDataFrame] = None,
    config: Optional[IntersectionAwareConfig] = None,
) -> gpd.GeoDataFrame:
    """
    Main function to deduplicate opposite-direction roads while preserving intersections.

    Args:
        gdf: Input GeoDataFrame with road segments
        intersections_gdf: GeoDataFrame with pre-detected intersection points
        config: Configuration

    Returns:
        Cleaned GeoDataFrame with preserved intersections
    """
    if config is None:
        config = IntersectionAwareConfig()

    print("=" * 60)
    print("Intersection-Aware Opposite-Direction Deduplication")
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
        print(f"Projecting to EPSG:{config.target_epsg}...")
        gdf = gdf.to_crs(epsg=config.target_epsg)
        if intersections_gdf is not None:
            intersections_gdf = intersections_gdf.to_crs(epsg=config.target_epsg)

    # Filter valid geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.reset_index(drop=True)

    # Filter by minimum length
    gdf = gdf[gdf.geometry.length >= config.min_segment_length].copy()
    print(f"After length filter (>{config.min_segment_length}m): {len(gdf)} segments")

    # STEP 1: Extract azimuths and direction vectors
    print("\nStep 1: Extracting azimuths and direction vectors...")
    azimuths = []
    direction_vectors = []

    for geom in tqdm(gdf.geometry, desc="Computing azimuths"):
        azimuth = calculate_azimuth(geom)
        azimuths.append(azimuth)
        direction_vectors.append(get_direction_vector(azimuth))

    azimuths = np.array(azimuths)

    # STEP 2 & 3: Find opposite-direction pairs
    print("\nStep 2 & 3: Finding opposite-direction pairs...")

    # Build spatial index
    sindex = gdf.sindex

    opposite_pairs = []
    processed = set()
    intersection_protected_segments = (
        set()
    )  # Segments we cannot merge due to intersections

    # First pass: identify segments that must be preserved for intersection connectivity
    if config.preserve_intersection_connections and intersections_gdf is not None:
        print("\nIdentifying intersection-connected segments...")
        for idx in tqdm(range(len(gdf)), desc="Checking intersection connectivity"):
            geom = gdf.geometry.iloc[idx]
            if should_preserve_segment(geom, intersections_gdf, config):
                intersection_protected_segments.add(idx)

        print(
            f"Protected segments (intersection-connected): {len(intersection_protected_segments)}"
        )

    # Second pass: find opposite pairs
    for idx in tqdm(range(len(gdf)), desc="Finding opposite pairs"):
        if idx in processed:
            continue

        geom = gdf.geometry.iloc[idx]
        if geom is None or geom.is_empty:
            continue

        # Find nearby segments
        bbox = geom.buffer(config.max_proximity).bounds
        candidates = list(sindex.intersection(bbox))

        for cand_idx in candidates:
            if cand_idx <= idx or cand_idx in processed:
                continue

            cand_geom = gdf.geometry.iloc[cand_idx]
            if cand_geom is None or cand_geom.is_empty:
                continue

            # Calculate perpendicular distance
            perp_dist = calculate_perpendicular_distance(geom, cand_geom)

            if perp_dist >= config.max_proximity:
                continue

            # Calculate cosine similarity
            cosine_sim = calculate_cosine_similarity(
                direction_vectors[idx], direction_vectors[cand_idx]
            )

            # Check if opposite direction
            if cosine_sim < config.cosine_threshold:
                # Calculate angle difference
                angle_diff = math.degrees(math.acos(max(-1, min(1, cosine_sim))))

                # Check direction consistency
                is_realistic = check_direction_consistency(
                    geom, cand_geom, azimuths[idx], azimuths[cand_idx], config
                )

                if not is_realistic:
                    continue

                # Validate as opposite pair
                is_valid = validate_opposite_pair(geom, cand_geom, config)

                if is_valid:
                    # Calculate overlap ratio for statistics
                    overlap_ratio = calculate_overlap_ratio(
                        geom, cand_geom, buffer_dist=2.0
                    )

                    opposite_pairs.append(
                        {
                            "idx1": idx,
                            "idx2": cand_idx,
                            "distance": perp_dist,
                            "angle_diff": angle_diff,
                            "overlap_ratio": overlap_ratio,
                        }
                    )

    print(f"Found {len(opposite_pairs)} opposite-direction pairs")

    if len(opposite_pairs) == 0:
        print("No opposite pairs found. Returning original data.")
        return gdf.to_crs(original_crs) if original_crs else gdf

    # STEP 4: Spatial Clustering + Majority Voting
    print("\nStep 4: Spatial clustering and majority voting...")

    # Build clusters using DBSCAN on segment centroids
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])

    clustering = DBSCAN(
        eps=config.spatial_cluster_radius, min_samples=2, metric="euclidean"
    ).fit(centroids)

    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Spatial clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    # Process each cluster with majority voting
    merged_segments = {}
    bidirectional_segments = {}
    segments_to_remove = set()

    for cluster_id in tqdm(range(n_clusters), desc="Processing clusters"):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # Check if any segments in this cluster are intersection-protected
        has_protected = any(
            idx in intersection_protected_segments for idx in cluster_indices
        )

        if has_protected:
            # Don't merge this cluster - preserve all segments
            continue

        # Get geometries and azimuths for this cluster
        cluster_geoms = [gdf.geometry.iloc[i] for i in cluster_indices]

        # Determine direction by voting
        direction_decision = determine_road_direction_by_voting(
            cluster_indices, azimuths, gdf, config
        )

        if direction_decision == "bidirectional":
            # Keep as bidirectional - separate by direction bins
            bin0_indices, bin1_indices = group_by_direction_bins(
                cluster_indices, azimuths
            )

            if len(bin0_indices) > 0:
                centerline0 = compute_centerline(
                    [gdf.geometry.iloc[i] for i in bin0_indices], config
                )
                bidirectional_segments[f"cluster_{cluster_id}_dir0"] = centerline0
                segments_to_remove.update(bin0_indices)

            if len(bin1_indices) > 0:
                centerline1 = compute_centerline(
                    [gdf.geometry.iloc[i] for i in bin1_indices], config
                )
                bidirectional_segments[f"cluster_{cluster_id}_dir1"] = centerline1
                segments_to_remove.update(bin1_indices)

        elif direction_decision == "unidirectional":
            # Merge to single centerline
            centerline = compute_centerline(cluster_geoms, config)
            merged_segments[f"cluster_{cluster_id}"] = centerline
            segments_to_remove.update(cluster_indices)

    print(f"Merged segments: {len(merged_segments)}")
    print(f"Bidirectional segments: {len(bidirectional_segments)}")
    print(f"Segments to remove: {len(segments_to_remove)}")
    print(f"Protected segments preserved: {len(intersection_protected_segments)}")

    # STEP 5: Build output GeoDataFrame
    print("\nStep 5: Building output...")

    # Keep all original segments NOT in removal set
    kept_indices = [i for i in range(len(gdf)) if i not in segments_to_remove]
    output_gdf = gdf.iloc[kept_indices].copy()

    # Add merged centerlines
    if merged_segments:
        merged_gdf = gpd.GeoDataFrame(
            {"geometry": list(merged_segments.values())}, crs=gdf.crs
        )
        output_gdf = gpd.GeoDataFrame(
            pd.concat([output_gdf, merged_gdf], ignore_index=True), crs=gdf.crs
        )

    # Add bidirectional centerlines
    if bidirectional_segments:
        bidir_gdf = gpd.GeoDataFrame(
            {"geometry": list(bidirectional_segments.values())}, crs=gdf.crs
        )
        output_gdf = gpd.GeoDataFrame(
            pd.concat([output_gdf, bidir_gdf], ignore_index=True), crs=gdf.crs
        )

    # STEP 6: Snap endpoints to intersections
    if config.preserve_intersection_connections and intersections_gdf is not None:
        print("\nStep 6: Snapping endpoints to intersections...")
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


# Add missing import
import pandas as pd
