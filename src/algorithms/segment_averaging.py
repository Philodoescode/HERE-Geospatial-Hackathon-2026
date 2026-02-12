"""
Segment Averaging Module

Implements Fréchet-based averaging of overlapping road segment candidates.
Instead of picking a "winner" among parallel candidates, this computes a
weighted geometric median that leverages all trajectory data.

Uses existing building blocks:
- Discrete Fréchet distance for trajectory comparison
- Weighted median per-point averaging
- Turn-preserving smoothing
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from shapely.geometry import LineString


@dataclass
class SegmentAveragingConfig:
    """Configuration for segment averaging."""
    
    # Resampling
    resample_spacing_m: float = 5.0  # Target spacing for resampling
    min_resample_points: int = 6
    max_resample_points: int = 50
    
    # Grouping thresholds
    corridor_buffer_m: float = 12.0  # Lateral buffer for corridor overlap
    heading_tolerance_deg: float = 30.0  # Max heading difference to be same road
    min_corridor_overlap: float = 0.60  # Min fraction inside corridor
    
    # Fréchet weighting
    use_frechet_weighting: bool = True
    frechet_eccentricity_power: float = 1.0  # Higher = more penalty on outliers
    
    # Z-level handling
    z_separation_threshold_m: float = 3.0  # Min altitude difference for separate levels
    
    # Smoothing
    smooth_passes: int = 2
    turn_angle_deg: float = 30.0  # Angle threshold for turn detection


def resample_polyline(coords: np.ndarray, n_points: int) -> np.ndarray:
    """Resample polyline to exactly n_points equidistant points."""
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


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Compute discrete Fréchet distance between two 2D polylines."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    
    na, nb = len(a), len(b)
    
    # Precompute all pairwise distances
    dx = a[:, 0:1] - b[:, 0]
    dy = a[:, 1:2] - b[:, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Iterative DP
    ca = np.empty((na, nb), dtype=np.float64)
    ca[0, 0] = dist[0, 0]
    
    for i in range(1, na):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, nb):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    
    for i in range(1, na):
        for j in range(1, nb):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), dist[i, j])
    
    return float(ca[na - 1, nb - 1])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of values."""
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


def bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute bearing from (x1,y1) to (x2,y2) in degrees."""
    dx = x2 - x1
    dy = y2 - y1
    return np.degrees(np.arctan2(dy, dx)) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Compute minimal angle difference between two headings."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def get_line_heading(geom: LineString) -> float:
    """Get heading of line from start to end."""
    if geom is None or geom.is_empty or len(geom.coords) < 2:
        return 0.0
    coords = list(geom.coords)
    return bearing_from_xy(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])


def smooth_polyline(coords: np.ndarray, keep_idx: set, passes: int) -> np.ndarray:
    """Apply smoothing while preserving points at turn indices."""
    out = coords.copy()
    
    if len(out) <= 2 or passes <= 0:
        return out
    
    for _ in range(passes):
        nxt = out.copy()
        for i in range(1, len(out) - 1):
            if i in keep_idx:
                continue
            nxt[i] = 0.2 * out[i - 1] + 0.6 * out[i] + 0.2 * out[i + 1]
        out = nxt
    
    return out


class SegmentGrouper:
    """
    Groups overlapping segment candidates that represent the same road section.
    
    Uses corridor-based overlap and heading compatibility to identify
    segments that should be averaged together.
    """
    
    def __init__(self, config: Optional[SegmentAveragingConfig] = None):
        self.config = config or SegmentAveragingConfig()
    
    def group_segments(
        self,
        geometries: List[LineString],
        segment_ids: List[int],
        supports: List[float],
        altitudes: Optional[List[float]] = None,
        node_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict]:
        """
        Group overlapping segments that represent the same road.
        
        Args:
            geometries: LineString geometries (in projected CRS)
            segment_ids: Segment identifiers
            supports: Support/weight values per segment
            altitudes: Optional mean altitude per segment
            node_pairs: Optional (start_node, end_node) per segment
        
        Returns:
            List of groups, each containing indices of overlapping segments
        """
        cfg = self.config
        n = len(geometries)
        
        if n == 0:
            return []
        
        if altitudes is None:
            altitudes = [np.nan] * n
        
        # Compute headings and lengths
        headings = np.array([get_line_heading(g) % 180.0 for g in geometries])
        lengths = np.array([g.length if g and not g.is_empty else 0.0 for g in geometries])
        
        # Build spatial index
        from shapely import STRtree
        valid_mask = [g is not None and not g.is_empty and g.length > 0 for g in geometries]
        valid_geoms = [g for g, v in zip(geometries, valid_mask) if v]
        valid_idx = [i for i, v in enumerate(valid_mask) if v]
        
        if len(valid_geoms) < 1:
            return []
        
        tree = STRtree(valid_geoms)
        
        # Build groups using Union-Find
        from src.algorithms.intersection_detection import UnionFind
        uf = UnionFind(n)
        
        # Pre-compute buffers
        buffers = {}
        for i in valid_idx:
            buffers[i] = geometries[i].buffer(cfg.corridor_buffer_m)
        
        # Find overlapping pairs
        for local_i, i in enumerate(valid_idx):
            geom_i = valid_geoms[local_i]
            len_i = lengths[i]
            h_i = headings[i]
            alt_i = altitudes[i]
            
            # Query nearby candidates
            candidates = tree.query(geom_i.buffer(cfg.corridor_buffer_m))
            
            for local_j in candidates:
                j = valid_idx[local_j]
                if i >= j:
                    continue
                
                # Heading compatibility
                h_diff = abs(h_i - headings[j])
                h_diff = min(h_diff, 180.0 - h_diff)
                if h_diff > cfg.heading_tolerance_deg:
                    continue
                
                # Z-level compatibility
                alt_j = altitudes[j]
                if np.isfinite(alt_i) and np.isfinite(alt_j):
                    if abs(alt_i - alt_j) > cfg.z_separation_threshold_m:
                        continue
                
                # Node pair compatibility (if available)
                # Segments must share BOTH endpoints to be considered parallel
                # (same road section), not just one endpoint (different roads meeting)
                has_complete_nodes_i = False
                has_complete_nodes_j = False
                nodes_match = False
                
                if node_pairs is not None:
                    pair_i = node_pairs[i]
                    pair_j = node_pairs[j]
                    
                    has_complete_nodes_i = (pair_i[0] is not None and pair_i[1] is not None)
                    has_complete_nodes_j = (pair_j[0] is not None and pair_j[1] is not None)
                    
                    if has_complete_nodes_i and has_complete_nodes_j:
                        # Both have complete node pairs - require both endpoints to match
                        same_direction = (pair_i[0] == pair_j[0] and pair_i[1] == pair_j[1])
                        reversed_direction = (pair_i[0] == pair_j[1] and pair_i[1] == pair_j[0])
                        if not (same_direction or reversed_direction):
                            continue
                        nodes_match = True
                
                # Corridor overlap check
                geom_j = geometries[j]
                len_j = lengths[j]
                
                shorter_idx = i if len_i <= len_j else j
                longer_idx = j if len_i <= len_j else i
                shorter_geom = geometries[shorter_idx]
                longer_buf = buffers[longer_idx]
                
                try:
                    inside = shorter_geom.intersection(longer_buf)
                    overlap_ratio = inside.length / lengths[shorter_idx]
                except Exception:
                    continue
                
                # Require stricter overlap when node pairs are incomplete
                # This prevents incorrectly grouping different roads that just happen to be nearby
                if nodes_match:
                    # Node pairs match - use standard overlap threshold
                    required_overlap = cfg.min_corridor_overlap
                else:
                    # Node pairs incomplete - require MUCH higher overlap (essentially same line)
                    required_overlap = 0.85
                
                if overlap_ratio >= required_overlap:
                    uf.union(i, j)
        
        # Extract groups
        groups_dict = defaultdict(list)
        for i in range(n):
            if valid_mask[i]:
                root = uf.find(i)
                groups_dict[root].append(i)
        
        # Build group records
        groups = []
        for root, members in groups_dict.items():
            groups.append({
                "group_id": root,
                "member_indices": members,
                "member_count": len(members),
                "total_support": sum(supports[i] for i in members),
            })
        
        return groups


class SegmentAverager:
    """
    Computes Fréchet-weighted average of overlapping segment candidates.
    
    Instead of picking one "winner", this produces a centerline that is
    the weighted geometric median of all candidates, leveraging all data.
    """
    
    def __init__(self, config: Optional[SegmentAveragingConfig] = None):
        self.config = config or SegmentAveragingConfig()
    
    def average_group(
        self,
        geometries: List[LineString],
        supports: List[float],
        altitudes: Optional[List[float]] = None,
        source_types: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        Compute weighted average centerline from a group of candidates.
        
        Args:
            geometries: LineString geometries of group members
            supports: Weight/support values
            altitudes: Optional altitude values for Z-level handling
            source_types: Optional source indicators (VPD, HPD, etc.)
        
        Returns:
            Dict with averaged geometry and metadata, or None if failed
        """
        cfg = self.config
        
        if not geometries:
            return None
        
        # Filter valid geometries
        valid = []
        for i, geom in enumerate(geometries):
            if geom is not None and not geom.is_empty and geom.length > 0:
                valid.append({
                    "geom": geom,
                    "support": supports[i] if i < len(supports) else 1.0,
                    "altitude": altitudes[i] if altitudes and i < len(altitudes) else np.nan,
                    "source": source_types[i] if source_types and i < len(source_types) else "unknown",
                })
        
        if len(valid) == 0:
            return None
        
        if len(valid) == 1:
            # Single member - just return it
            m = valid[0]
            return {
                "geometry": m["geom"],
                "weighted_support": m["support"],
                "member_count": 1,
                "altitude_mean": m["altitude"],
                "source": m["source"],
            }
        
        # Separate by Z-level if needed
        z_groups = self._separate_by_z_level(valid)
        
        results = []
        for z_group in z_groups:
            result = self._average_single_z_group(z_group)
            if result is not None:
                results.append(result)
        
        if len(results) == 0:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # Multiple Z-levels - return list or pick primary
        # Return the one with highest support as primary, attach others as metadata
        results.sort(key=lambda r: r["weighted_support"], reverse=True)
        primary = results[0]
        primary["z_level_variants"] = results[1:]
        return primary
    
    def _separate_by_z_level(self, members: List[Dict]) -> List[List[Dict]]:
        """Separate members into Z-level groups."""
        cfg = self.config
        
        # Extract altitudes
        alts = [m["altitude"] for m in members]
        finite_alts = [a for a in alts if np.isfinite(a)]
        
        if len(finite_alts) < 2:
            return [members]
        
        # Check altitude range
        alt_range = max(finite_alts) - min(finite_alts)
        if alt_range < cfg.z_separation_threshold_m:
            return [members]
        
        # Cluster by altitude
        alt_arr = np.array([a if np.isfinite(a) else np.nanmean(finite_alts) for a in alts])
        
        # Simple two-group clustering
        threshold = np.median(finite_alts)
        
        low_group = []
        high_group = []
        
        for m, alt in zip(members, alt_arr):
            if alt < threshold:
                low_group.append(m)
            else:
                high_group.append(m)
        
        groups = []
        if low_group:
            groups.append(low_group)
        if high_group:
            groups.append(high_group)
        
        return groups if groups else [members]
    
    def _average_single_z_group(self, members: List[Dict]) -> Optional[Dict]:
        """Average a single Z-level group using Fréchet-weighted median."""
        cfg = self.config
        
        if not members:
            return None
        
        if len(members) == 1:
            m = members[0]
            return {
                "geometry": m["geom"],
                "weighted_support": m["support"],
                "member_count": 1,
                "altitude_mean": m["altitude"],
                "source": m["source"],
            }
        
        # Determine number of resample points
        lengths = [m["geom"].length for m in members]
        med_length = float(np.median(lengths))
        n_points = int(np.clip(
            med_length / cfg.resample_spacing_m + 1,
            cfg.min_resample_points,
            cfg.max_resample_points
        ))
        
        # Extract coordinates and resample
        resampled = []
        weights = []
        altitudes = []
        
        for m in members:
            coords = np.array(m["geom"].coords)[:, :2]
            # Handle direction: ensure consistent direction
            coords = self._ensure_consistent_direction(coords, members, m)
            resampled.append(resample_polyline(coords, n_points))
            weights.append(m["support"])
            altitudes.append(m["altitude"])
        
        resampled = np.array(resampled)  # (M, N, 2)
        weights = np.array(weights)
        
        # Compute Fréchet-based weights (only for small groups to avoid slowdown)
        # For groups > 10 members, skip Fréchet weighting as it's O(n²)
        if cfg.use_frechet_weighting and 2 < len(members) <= 10:
            frechet_weights = self._compute_frechet_weights(resampled, weights)
        else:
            frechet_weights = weights
        
        # Weighted median per point
        result_coords = np.zeros((n_points, 2), dtype=np.float64)
        
        for i in range(n_points):
            result_coords[i, 0] = weighted_median(resampled[:, i, 0], frechet_weights)
            result_coords[i, 1] = weighted_median(resampled[:, i, 1], frechet_weights)
        
        # Detect turn points
        turn_idx = {0, n_points - 1}
        for i in range(1, n_points - 1):
            h1 = bearing_from_xy(
                result_coords[i-1, 0], result_coords[i-1, 1],
                result_coords[i, 0], result_coords[i, 1]
            )
            h2 = bearing_from_xy(
                result_coords[i, 0], result_coords[i, 1],
                result_coords[i+1, 0], result_coords[i+1, 1]
            )
            if angle_diff_deg(h1, h2) >= cfg.turn_angle_deg:
                turn_idx.add(i)
        
        # Smooth while preserving turns
        result_coords = smooth_polyline(result_coords, turn_idx, cfg.smooth_passes)
        
        # Build output geometry
        result_geom = LineString(result_coords)
        
        # Compute averaged altitude
        finite_alts = [a for a in altitudes if np.isfinite(a)]
        mean_alt = float(np.median(finite_alts)) if finite_alts else np.nan
        
        # Source aggregation
        source_counts = defaultdict(int)
        for m in members:
            source_counts[m["source"]] += 1
        primary_source = max(source_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "geometry": result_geom,
            "weighted_support": float(np.sum(weights)),
            "member_count": len(members),
            "altitude_mean": mean_alt,
            "source": primary_source,
            "turn_indices": sorted(turn_idx),
        }
    
    def _ensure_consistent_direction(
        self, 
        coords: np.ndarray, 
        all_members: List[Dict],
        current_member: Dict
    ) -> np.ndarray:
        """Ensure coordinates are in consistent direction with majority of group."""
        if len(all_members) <= 1:
            return coords
        
        # Use first member as reference direction
        ref_coords = np.array(all_members[0]["geom"].coords)[:, :2]
        ref_start = ref_coords[0]
        
        # Check if current coords should be reversed
        curr_start = coords[0]
        curr_end = coords[-1]
        
        dist_to_start = np.hypot(curr_start[0] - ref_start[0], curr_start[1] - ref_start[1])
        dist_to_end = np.hypot(curr_end[0] - ref_start[0], curr_end[1] - ref_start[1])
        
        if dist_to_end < dist_to_start:
            return coords[::-1]
        return coords
    
    def _compute_frechet_weights(
        self, 
        resampled: np.ndarray, 
        base_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute Fréchet-adjusted weights.
        
        Penalize outlier trajectories that have high Fréchet distance to others.
        """
        cfg = self.config
        n = len(resampled)
        
        if n <= 2:
            return base_weights
        
        # Compute pairwise Fréchet distances
        frechet_dists = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = discrete_frechet(resampled[i], resampled[j])
                frechet_dists[i, j] = d
                frechet_dists[j, i] = d
        
        # Compute eccentricity (mean distance to others)
        eccentricity = np.mean(frechet_dists, axis=1)
        
        # Normalize eccentricity
        if eccentricity.max() > 0:
            eccentricity_norm = eccentricity / eccentricity.max()
        else:
            eccentricity_norm = np.zeros(n)
        
        # Penalize high eccentricity members
        penalty = 1.0 / (1.0 + eccentricity_norm ** cfg.frechet_eccentricity_power)
        
        return base_weights * penalty


def average_segment_groups(
    geometries: List[LineString],
    segment_ids: List[int],
    supports: List[float],
    groups: List[Dict],
    altitudes: Optional[List[float]] = None,
    source_types: Optional[List[str]] = None,
    config: Optional[SegmentAveragingConfig] = None,
) -> List[Dict]:
    """
    Convenience function to average all segment groups.
    
    Args:
        geometries: All segment geometries
        segment_ids: All segment IDs  
        supports: All support values
        groups: Output from SegmentGrouper.group_segments()
        altitudes: Optional altitude values
        source_types: Optional source type labels
        config: Averaging configuration
    
    Returns:
        List of averaged segment records
    """
    averager = SegmentAverager(config)
    results = []
    
    for group in groups:
        indices = group["member_indices"]
        
        group_geoms = [geometries[i] for i in indices]
        group_supports = [supports[i] for i in indices]
        group_alts = [altitudes[i] for i in indices] if altitudes else None
        group_sources = [source_types[i] for i in indices] if source_types else None
        
        result = averager.average_group(
            group_geoms, group_supports, group_alts, group_sources
        )
        
        if result is not None:
            result["original_segment_ids"] = [segment_ids[i] for i in indices]
            result["group_id"] = group["group_id"]
            results.append(result)
    
    return results
