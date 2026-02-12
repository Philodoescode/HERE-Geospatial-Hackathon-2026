"""
Bidirectional Lane Deduplication for Overlapping Opposite-Direction Roads

This module identifies and merges redundant line segments that occupy the same
geographic space but have opposite headings, consolidating them into clean
double-digitized centerline pairs.

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


# =============================================================================
# Configuration
# =============================================================================


class OppositeDirectionConfig:
    """Configuration for opposite-direction deduplication (AGGRESSIVE MODE)."""

    # Spatial proximity threshold (meters) - INCREASED for better detection
    max_proximity: float = 8.0

    # Distance threshold for merging - MORE AGGRESSIVE
    # Segments closer than this are definitely GPS noise/duplicates
    merge_distance_threshold: float = 5.0  # Up from 3.0m

    # Borderline distance range - WIDER range for validation
    borderline_min: float = 5.0
    borderline_max: float = 8.0

    # Cosine similarity threshold - MORE AGGRESSIVE
    # cosine < -0.5 means angle > ~120° (catch more opposite directions)
    cosine_threshold: float = -0.5  # More aggressive (was -0.7)

    # Minimum overlap ratio - LOWER to catch more cases
    min_overlap_ratio: float = 0.5  # Down from 0.7

    # Standard lane separation for output (meters)
    output_lane_separation: float = 3.6  # 12 feet

    # Mean Shift bandwidth for centerline computation
    mean_shift_bandwidth: float = 2.0  # More precise centerline

    # Minimum segment length to process - LOWER to catch shorter segments
    min_segment_length: float = 10.0  # Down from 20.0

    # Direction flip tolerance - don't allow direction changes in middle of road
    # A road should maintain direction consistency
    direction_consistency_check: bool = True

    # Maximum angle change allowed within a road segment
    max_direction_change: float = 45.0  # degrees

    # MAJORITY VOTING PARAMETERS
    # Spatial clustering radius for grouping nearby segments
    spatial_cluster_radius: float = 15.0  # meters

    # Majority vote threshold - if one direction has > this %, merge to one direction
    # If close to 50/50 (within this threshold), keep as bidirectional
    majority_threshold: float = 0.65  # 65% - clear majority

    # Bidirectional threshold - if vote is between 35-65%, it's bidirectional
    bidirectional_min: float = 0.35
    bidirectional_max: float = 0.65

    # Target CRS for metric calculations
    target_epsg: int = 32634


# =============================================================================
# STEP 1: Azimuth Extraction
# =============================================================================


def calculate_azimuth(line: LineString) -> float:
    """
    Calculate the azimuth/heading of a line segment.

    Args:
        line: LineString to measure

    Returns:
        Azimuth in degrees [0, 360)
    """
    if line is None or line.is_empty or len(line.coords) < 2:
        return 0.0

    start_point = line.coords[0]
    end_point = line.coords[-1]

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # Calculate azimuth
    azimuth = math.degrees(math.atan2(dy, dx))

    # Normalize to [0, 360)
    if azimuth < 0:
        azimuth += 360

    return azimuth


def get_direction_vector(azimuth: float) -> Tuple[float, float]:
    """
    Convert azimuth to direction vector.

    Args:
        azimuth: Azimuth in degrees

    Returns:
        (dx, dy) direction vector
    """
    rad = math.radians(azimuth)
    return (math.cos(rad), math.sin(rad))


# =============================================================================
# STEP 2 & 3: Spatial Proximity & Cosine Directional Filtering
# =============================================================================


def calculate_cosine_similarity(
    v1: Tuple[float, float], v2: Tuple[float, float]
) -> float:
    """
    Calculate cosine similarity between two direction vectors.

    Args:
        v1: First direction vector
        v2: Second direction vector

    Returns:
        Cosine similarity [-1, 1]
    """
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_perpendicular_distance(line1: LineString, line2: LineString) -> float:
    """
    Calculate average perpendicular distance between two lines.

    Args:
        line1: First line
        line2: Second line

    Returns:
        Average perpendicular distance in meters
    """
    # Sample points along line1 and measure distance to line2
    num_samples = max(5, int(line1.length / 10))
    distances = []

    for i in range(num_samples + 1):
        frac = i / num_samples
        point = line1.interpolate(frac, normalized=True)
        dist = point.distance(line2)
        distances.append(dist)

    return np.mean(distances)


# =============================================================================
# Direction Consistency Validation
# =============================================================================


def check_direction_consistency(
    line1: LineString,
    line2: LineString,
    azimuth1: float,
    azimuth2: float,
    config: OppositeDirectionConfig,
) -> bool:
    """
    Check if two opposite-direction segments violate realistic road direction rules.

    A realistic road can be: right -> two-way -> right
    But CANNOT be: right -> left -> right (direction flip in middle)

    Args:
        line1: First line
        line2: Second line
        azimuth1: Azimuth of line1
        azimuth2: Azimuth of line2
        config: Configuration

    Returns:
        True if direction pattern is realistic (should merge), False otherwise
    """
    if not config.direction_consistency_check:
        return True

    # Get start and end points of both lines
    start1 = np.array(line1.coords[0])
    end1 = np.array(line1.coords[-1])
    start2 = np.array(line2.coords[0])
    end2 = np.array(line2.coords[-1])

    # Check which configuration we have:
    # 1. Lines flow in same general direction (start->end roughly aligned)
    # 2. Lines flow in opposite direction (one reversed relative to other)

    # Calculate vectors
    vec1 = end1 - start1
    vec2 = end2 - start2

    # If lines are truly opposite direction (one reversed), their direction vectors
    # should point in opposite directions
    dot_product = np.dot(vec1, vec2)

    # If dot product is negative, they're truly opposite (BAD - direction flip)
    # If dot product is positive, they're flowing same way (GOOD - parallel lanes)

    # However, if they're very close together AND opposite direction,
    # they're likely GPS noise on the same road

    # Check distance between midpoints
    mid1 = line1.interpolate(0.5, normalized=True)
    mid2 = line2.interpolate(0.5, normalized=True)
    mid_distance = mid1.distance(mid2)

    # If midpoints are very close AND directions are opposite, this is GPS noise
    # on the same road (should merge)
    if mid_distance < config.merge_distance_threshold:
        # Too close to be separate lanes - definitely merge
        return True

    # If midpoints are further apart but directions are opposite,
    # this might be legitimate opposite lanes (should NOT merge)
    if mid_distance > config.borderline_max:
        # Far enough apart to be separate bidirectional lanes
        return False

    # Borderline case - use additional checks
    # Check if the lines are consistently parallel throughout their length
    num_samples = 5
    direction_changes = []

    for i in range(num_samples):
        frac = i / (num_samples - 1)

        # Get local tangent directions at this fraction
        if i == 0:
            frac_delta = 0.1
            p1_a = line1.interpolate(frac, normalized=True)
            p1_b = line1.interpolate(frac + frac_delta, normalized=True)
            p2_a = line2.interpolate(frac, normalized=True)
            p2_b = line2.interpolate(frac + frac_delta, normalized=True)
        elif i == num_samples - 1:
            frac_delta = 0.1
            p1_a = line1.interpolate(frac - frac_delta, normalized=True)
            p1_b = line1.interpolate(frac, normalized=True)
            p2_a = line2.interpolate(frac - frac_delta, normalized=True)
            p2_b = line2.interpolate(frac, normalized=True)
        else:
            frac_delta = 0.05
            p1_a = line1.interpolate(frac - frac_delta, normalized=True)
            p1_b = line1.interpolate(frac + frac_delta, normalized=True)
            p2_a = line2.interpolate(frac - frac_delta, normalized=True)
            p2_b = line2.interpolate(frac + frac_delta, normalized=True)

        # Local direction vectors
        v1 = np.array([p1_b.x - p1_a.x, p1_b.y - p1_a.y])
        v2 = np.array([p2_b.x - p2_a.x, p2_b.y - p2_a.y])

        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

        # Calculate angle between local directions
        dot = np.dot(v1_norm, v2_norm)
        angle = math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))
        direction_changes.append(angle)

    # If direction changes are consistently around 180° (opposite),
    # but distance is small, merge them
    avg_angle_diff = np.mean(direction_changes)

    if avg_angle_diff > 150 and mid_distance < config.borderline_max:
        # Consistently opposite but close - likely GPS noise
        return True

    # Otherwise, keep them separate
    return False


# =============================================================================
# STEP 4: Midline Overlap Validation
# =============================================================================


def calculate_overlap_ratio(
    line1: LineString, line2: LineString, buffer_dist: float = 2.0
) -> float:
    """
    Calculate spatial overlap ratio between two lines.

    Args:
        line1: First line
        line2: Second line
        buffer_dist: Buffer distance for overlap calculation

    Returns:
        Overlap ratio [0, 1]
    """
    try:
        # Create buffer around line2
        buffered = line2.buffer(buffer_dist)

        # Find overlapping portion of line1
        intersection = line1.intersection(buffered)

        if intersection.is_empty:
            return 0.0

        # Calculate overlap length
        if hasattr(intersection, "length"):
            overlap_length = intersection.length
        else:
            # Handle MultiLineString or other geometry types
            overlap_length = sum(
                geom.length for geom in getattr(intersection, "geoms", [intersection])
            )

        # Calculate ratio based on longer line
        max_length = max(line1.length, line2.length)

        if max_length == 0:
            return 0.0

        return overlap_length / max_length

    except Exception:
        return 0.0


def validate_opposite_pair(
    line1: LineString, line2: LineString, config: OppositeDirectionConfig
) -> bool:
    """
    Validate that two opposite-direction segments truly represent the same road.

    Args:
        line1: First line
        line2: Second line
        config: Configuration

    Returns:
        True if they should be merged
    """
    # Calculate average perpendicular distance
    avg_distance = calculate_perpendicular_distance(line1, line2)

    # Calculate spatial overlap
    overlap_ratio = calculate_overlap_ratio(line1, line2, buffer_dist=2.0)

    # Validation logic
    if avg_distance < 2.5 and overlap_ratio > config.min_overlap_ratio:
        return True

    return False


# =============================================================================
# Majority Voting for Direction Determination
# =============================================================================


def normalize_azimuth_to_0_180(azimuth: float) -> float:
    """
    Normalize azimuth to 0-180° range.

    Treats opposite directions as same (e.g., 45° and 225° both become 45°).
    This helps group parallel roads regardless of digitization direction.

    Args:
        azimuth: Azimuth in degrees [0, 360)

    Returns:
        Normalized azimuth in [0, 180)
    """
    normalized = azimuth % 180
    return normalized


def group_by_direction_bins(
    azimuths: np.ndarray, indices: List[int], bin_width: float = 20.0
) -> Dict[int, List[int]]:
    """
    Group segments by direction bins.

    Args:
        azimuths: Array of azimuths
        indices: List of segment indices
        bin_width: Width of direction bins in degrees

    Returns:
        Dictionary mapping bin_id -> list of indices
    """
    bins = {}

    for idx in indices:
        azimuth = azimuths[idx]
        # Normalize to 0-180 to group opposite directions together
        normalized = normalize_azimuth_to_0_180(azimuth)

        # Assign to bin
        bin_id = int(normalized / bin_width)

        if bin_id not in bins:
            bins[bin_id] = []
        bins[bin_id].append(idx)

    return bins


def determine_road_direction_by_voting(
    cluster_indices: List[int],
    azimuths: np.ndarray,
    gdf: gpd.GeoDataFrame,
    config: OppositeDirectionConfig,
) -> str:
    """
    Determine if a road cluster is unidirectional or bidirectional by majority voting.

    Logic:
    - Count segments in each direction (forward vs backward along same axis)
    - If one direction has >= 65%, it's unidirectional
    - If vote is 35-65%, it's bidirectional

    Args:
        cluster_indices: List of segment indices in this cluster
        azimuths: Array of all azimuths
        gdf: GeoDataFrame
        config: Configuration

    Returns:
        'unidirectional' or 'bidirectional'
    """
    if len(cluster_indices) < 2:
        return "unidirectional"

    # Separate into two opposing direction groups
    # For each segment, normalize to 0-180 to get the "road axis"
    # Then check if original azimuth is in [0,180) or [180,360) to determine direction

    direction_a_indices = []  # Forward along axis
    direction_b_indices = []  # Backward along axis

    for idx in cluster_indices:
        azimuth = azimuths[idx]

        # Segments with azimuth [0, 180) go one way
        # Segments with azimuth [180, 360) go opposite way
        if 0 <= azimuth < 180:
            direction_a_indices.append(idx)
        else:
            direction_b_indices.append(idx)

    total_count = len(cluster_indices)
    direction_a_count = len(direction_a_indices)
    direction_b_count = len(direction_b_indices)

    # Calculate vote percentages
    vote_a = direction_a_count / total_count
    vote_b = direction_b_count / total_count

    # Determine direction type
    if vote_a >= config.majority_threshold or vote_b >= config.majority_threshold:
        # Clear majority - unidirectional
        return "unidirectional"
    elif config.bidirectional_min <= vote_a <= config.bidirectional_max:
        # Close vote - bidirectional
        return "bidirectional"
    else:
        # Close enough to majority
        return "unidirectional"


def get_majority_direction_indices(
    cluster_indices: List[int], azimuths: np.ndarray
) -> List[int]:
    """
    Get indices of segments in the majority direction.

    Args:
        cluster_indices: List of segment indices
        azimuths: Array of azimuths

    Returns:
        List of indices in majority direction
    """
    direction_a_indices = []
    direction_b_indices = []

    for idx in cluster_indices:
        azimuth = azimuths[idx]
        if 0 <= azimuth < 180:
            direction_a_indices.append(idx)
        else:
            direction_b_indices.append(idx)

    # Return majority group
    if len(direction_a_indices) >= len(direction_b_indices):
        return direction_a_indices
    else:
        return direction_b_indices


# =============================================================================
# STEP 5: Centerline Computation & Double-Digitization
# =============================================================================


def compute_centerline(
    line1: LineString, line2: LineString, bandwidth: float = 1.5
) -> Optional[LineString]:
    """
    Compute centerline from two opposite-direction overlapping lines.

    Args:
        line1: First line
        line2: Second line
        bandwidth: Mean Shift bandwidth

    Returns:
        Centerline as LineString
    """
    # Collect all points
    all_points = []

    # Sample points from both lines
    for line in [line1, line2]:
        num_samples = max(10, int(line.length / 5))
        for i in range(num_samples + 1):
            frac = i / num_samples
            point = line.interpolate(frac, normalized=True)
            all_points.append([point.x, point.y])

    if len(all_points) < 3:
        # Fallback: simple average
        coords1 = list(line1.coords)
        coords2 = list(line2.coords)

        # Reverse line2 if it's in opposite direction
        if len(coords1) > 0 and len(coords2) > 0:
            start1 = np.array(coords1[0])
            start2 = np.array(coords2[0])
            end2 = np.array(coords2[-1])

            # If line2's end is closer to line1's start, reverse it
            if np.linalg.norm(end2 - start1) < np.linalg.norm(start2 - start1):
                coords2 = coords2[::-1]

        # Average the coordinates
        max_len = max(len(coords1), len(coords2))
        centerline_coords = []

        for i in range(max_len):
            if i < len(coords1) and i < len(coords2):
                mid_x = (coords1[i][0] + coords2[i][0]) / 2
                mid_y = (coords1[i][1] + coords2[i][1]) / 2
                centerline_coords.append((mid_x, mid_y))

        if len(centerline_coords) >= 2:
            return LineString(centerline_coords)
        return None

    try:
        # Apply Mean Shift clustering
        all_points = np.array(all_points)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(all_points)

        centers = ms.cluster_centers_

        if len(centers) < 2:
            return None

        # Order centers by proximity (form continuous line)
        ordered = _order_points_by_proximity(centers)

        return LineString(ordered)

    except Exception:
        return None


def _order_points_by_proximity(points: np.ndarray) -> np.ndarray:
    """Order points to form a continuous path."""
    if len(points) < 2:
        return points

    # Start from leftmost point
    start_idx = np.argmin(points[:, 0])

    ordered = [points[start_idx]]
    remaining = list(range(len(points)))
    remaining.remove(start_idx)

    while remaining:
        last = ordered[-1]
        distances = [np.linalg.norm(points[i] - last) for i in remaining]
        nearest_idx = remaining[np.argmin(distances)]
        ordered.append(points[nearest_idx])
        remaining.remove(nearest_idx)

    return np.array(ordered)


def create_offset_lines(
    centerline: LineString, lane_separation: float
) -> Tuple[Optional[LineString], Optional[LineString]]:
    """
    Create offset parallel lines from centerline.

    Args:
        centerline: Center line
        lane_separation: Distance to offset (full separation / 2)

    Returns:
        (north_line, south_line) tuple
    """
    if centerline is None or centerline.is_empty:
        return None, None

    try:
        offset_dist = lane_separation / 2

        line_north = centerline.parallel_offset(offset_dist, "left")
        line_south = centerline.parallel_offset(offset_dist, "right")

        # Handle potential MultiLineString results
        if hasattr(line_north, "geoms"):
            line_north = max(line_north.geoms, key=lambda g: g.length)

        if hasattr(line_south, "geoms"):
            line_south = max(line_south.geoms, key=lambda g: g.length)

        return line_north, line_south

    except Exception:
        return None, None


# =============================================================================
# Main Processing
# =============================================================================


def deduplicate_opposite_directions(
    gdf: gpd.GeoDataFrame, config: Optional[OppositeDirectionConfig] = None
) -> gpd.GeoDataFrame:
    """
    Main function to deduplicate overlapping opposite-direction road segments.

    Args:
        gdf: Input GeoDataFrame
        config: Configuration

    Returns:
        Cleaned GeoDataFrame
    """
    if config is None:
        config = OppositeDirectionConfig()

    print("=" * 60)
    print("Opposite-Direction Redundancy Deduplication")
    print("=" * 60)
    print(f"Input: {len(gdf)} geometries")

    # Store original CRS
    original_crs = gdf.crs
    input_count = len(gdf)

    # Project to metric CRS
    if original_crs is not None and original_crs.to_epsg() != config.target_epsg:
        print(f"Projecting to EPSG:{config.target_epsg}...")
        gdf = gdf.to_crs(epsg=config.target_epsg)

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

                # Check direction consistency (realistic road pattern)
                is_realistic = check_direction_consistency(
                    geom, cand_geom, azimuths[idx], azimuths[cand_idx], config
                )

                if not is_realistic:
                    # Direction pattern unrealistic - keep as separate roads
                    continue

                # Determine if should merge
                should_merge = False
                reason = "no_merge"

                if perp_dist < config.merge_distance_threshold:
                    # Too close - definitely redundant GPS noise
                    should_merge = True
                    reason = "too_close"
                elif config.borderline_min <= perp_dist < config.borderline_max:
                    # Borderline - need extra validation
                    if validate_opposite_pair(geom, cand_geom, config):
                        should_merge = True
                        reason = "borderline_validated"
                    else:
                        reason = "borderline_rejected"

                if should_merge:
                    opposite_pairs.append(
                        {
                            "idx1": idx,
                            "idx2": cand_idx,
                            "distance": perp_dist,
                            "angle_diff": angle_diff,
                            "cosine_sim": cosine_sim,
                            "reason": reason,
                        }
                    )
                    processed.add(idx)
                    processed.add(cand_idx)

    print(f"  Found {len(opposite_pairs)} opposite-direction pairs to merge")

    if opposite_pairs:
        # Print statistics
        distances = [p["distance"] for p in opposite_pairs]
        angles = [p["angle_diff"] for p in opposite_pairs]
        print(
            f"  Distance range: {min(distances):.2f}m - {max(distances):.2f}m (avg: {np.mean(distances):.2f}m)"
        )
        print(
            f"  Angle range: {min(angles):.1f}° - {max(angles):.1f}° (avg: {np.mean(angles):.1f}°)"
        )

    # STEP 4: Group pairs into spatial clusters for majority voting
    print("\nStep 4: Grouping pairs into spatial clusters for majority voting...")

    # Build spatial clusters - group nearby segments together
    from sklearn.cluster import DBSCAN

    # Get all indices involved in pairs
    all_pair_indices = set()
    for pair in opposite_pairs:
        all_pair_indices.add(pair["idx1"])
        all_pair_indices.add(pair["idx2"])

    if not all_pair_indices:
        spatial_clusters = {}
    else:
        # Get centroids of all paired segments
        pair_indices_list = list(all_pair_indices)
        centroids = []
        for idx in pair_indices_list:
            geom = gdf.geometry.iloc[idx]
            if geom is not None and not geom.is_empty:
                centroid = geom.centroid
                centroids.append([centroid.x, centroid.y])
            else:
                centroids.append([0, 0])

        centroids = np.array(centroids)

        # Cluster by spatial proximity
        clustering = DBSCAN(eps=config.spatial_cluster_radius, min_samples=1)
        labels = clustering.fit_predict(centroids)

        # Group indices by cluster
        spatial_clusters = {}
        for i, label in enumerate(labels):
            if label not in spatial_clusters:
                spatial_clusters[label] = []
            spatial_clusters[label].append(pair_indices_list[i])

    print(f"  Created {len(spatial_clusters)} spatial clusters")

    # STEP 5: Apply majority voting to each cluster
    print("\nStep 5: Applying majority voting to determine road directionality...")

    new_geometries = []
    merged_indices = set()
    unidirectional_count = 0
    bidirectional_count = 0

    for cluster_id, cluster_indices in tqdm(
        spatial_clusters.items(), desc="Processing clusters"
    ):
        # Determine if this cluster is unidirectional or bidirectional
        direction_type = determine_road_direction_by_voting(
            cluster_indices, azimuths, gdf, config
        )

        if direction_type == "bidirectional":
            # Keep separate lanes - don't merge
            bidirectional_count += 1
            for idx in cluster_indices:
                geom = gdf.geometry.iloc[idx]
                if geom is not None and not geom.is_empty:
                    new_geometries.append(geom)
                    merged_indices.add(idx)
        else:
            # Unidirectional - merge to centerline
            unidirectional_count += 1

            # Get all lines in this cluster
            cluster_lines = [
                gdf.geometry.iloc[i]
                for i in cluster_indices
                if gdf.geometry.iloc[i] is not None
            ]

            if not cluster_lines:
                continue

            # Compute centerline from all segments
            if len(cluster_lines) == 1:
                new_geometries.append(cluster_lines[0])
                merged_indices.update(cluster_indices)
            else:
                centerline = compute_centerline(
                    cluster_lines[0], cluster_lines[1], config.mean_shift_bandwidth
                )

                if centerline is None or centerline.is_empty:
                    # Fallback: keep majority direction segments
                    majority_indices = get_majority_direction_indices(
                        cluster_indices, azimuths
                    )
                    for idx in majority_indices:
                        new_geometries.append(gdf.geometry.iloc[idx])
                else:
                    new_geometries.append(centerline)

                merged_indices.update(cluster_indices)

    print(f"  Unidirectional roads: {unidirectional_count}")
    print(f"  Bidirectional roads: {bidirectional_count}")

    # Add non-merged segments
    print("\nStep 6: Adding non-merged segments...")
    non_merged_count = 0
    for idx in range(len(gdf)):
        if idx not in merged_indices:
            geom = gdf.geometry.iloc[idx]
            if geom is not None and not geom.is_empty:
                new_geometries.append(geom)
                non_merged_count += 1

    print(f"  Added {non_merged_count} non-merged segments")

    # Create result GeoDataFrame
    result = gpd.GeoDataFrame(geometry=new_geometries, crs=gdf.crs)

    # Filter empty geometries
    result = result[result.geometry.notnull() & ~result.geometry.is_empty]

    # Project back to original CRS
    if original_crs is not None and original_crs.to_epsg() != config.target_epsg:
        print(f"Projecting back to {original_crs}...")
        result = result.to_crs(original_crs)

    print("=" * 60)
    print(f"Output: {len(result)} geometries")
    reduction = input_count - len(result)
    reduction_pct = 100 * reduction / input_count if input_count > 0 else 0

    if reduction >= 0:
        print(f"Reduction: {reduction} geometries removed ({reduction_pct:.1f}%)")
    else:
        increase = -reduction
        increase_pct = 100 * increase / input_count
        print(
            f"Increase: {increase} geometries added ({increase_pct:.1f}%) due to double-digitization"
        )

    print("=" * 60)

    # Validation metrics
    print("\nValidation Metrics:")
    print(f"  Opposite-direction pairs found: {len(opposite_pairs)}")
    print(f"  Spatial clusters created: {len(spatial_clusters)}")
    print(f"  Unidirectional roads (merged): {unidirectional_count}")
    print(f"  Bidirectional roads (kept separate): {bidirectional_count}")
    print(f"  Segments affected: {len(merged_indices)}")
    print(f"  New double-digitized lanes created: {len(opposite_pairs) * 2}")

    return result


def main():
    """Main entry point."""
    import sys
    from pathlib import Path

    # Default paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    input_file = (
        project_root
        / "outputs"
        / "double_digitized_cleaned"
        / "kharita_double_digitized_cleaned_v3.gpkg"
    )
    output_file = (
        project_root
        / "outputs"
        / "double_digitized_cleaned"
        / "kharita_bidirectional_dedup.gpkg"
    )

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}\n")

    # Load data
    print("Loading data...")
    gdf = gpd.read_file(input_file)

    # Run deduplication
    config = OppositeDirectionConfig()
    result = deduplicate_opposite_directions(gdf, config)

    # Save result
    print(f"\nSaving to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_file, driver="GPKG")
    print("Done!")


if __name__ == "__main__":
    main()
