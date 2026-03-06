"""
Phase 3: VPD Geometry Refinement & Topology Cleanup

Consolidated post-processing pipeline (replaces old Phase 3 + Phase 5):
  1. Detect intersections from endpoint clustering
  2. Remove intersection spurs (short dead-end noise)
  3. Merge parallel overlapping segments (Hausdorff-based)
  4. Resolve Z-level crossings (split segments + assign z_level)
  5. Remove sharp-angle vertices (bird-nesting fix)
  6. Smooth geometry (Douglas-Peucker + B-spline with Hausdorff guard)
  7. Remove jitter spikes (starburst GPS noise)
  8. Remove loop/self-intersecting segments
  9. Prune dead-end stubs (iterative, NetworkX-based)
 10. Clean geometry + export

Key principle: Does NOT generate new centerlines.
Removes noise and cleans existing VPD geometry in a single pass.
"""

import gc
import math
import os
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import split, snap, unary_union

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.centerline_utils import (
    angle_diff_deg,
    bearing_from_xy,
)


@dataclass
class RefinementConfig:
    """Configuration for VPD geometry refinement."""
    
    # Intersection detection
    intersection_snap_m: float = 15.0
    min_intersection_degree: int = 3
    intersection_radius_m: float = 30.0
    
    # Noise removal at intersections (VERY CONSERVATIVE for max recovery)
    spur_max_length_m: float = 8.0  # Was 25→15→8 - only remove very short spurs
    spur_min_angle_deg: float = 15.0  # Was 20→15 - stricter angle requirement
    
    # Parallel segment consolidation - VERY CONSERVATIVE to preserve roads
    # The 12m buffer was removing 52% of segments - way too aggressive!
    parallel_hausdorff_m: float = 5.0     # Was 12.0 - only merge truly duplicate lines within 5m
    parallel_heading_tol_deg: float = 20.0  # Was 30.0 - stricter heading match
    parallel_min_length_ratio: float = 0.50  # Was 0.20 - don't remove short segments so aggressively
    parallel_min_overlap: float = 0.90       # Was 0.75 - require 90% overlap to merge
    
    # Z-level conflict resolution
    z_crossing_buffer_m: float = 20.0
    z_crossing_angle_min_deg: float = 30.0  # Min angle to consider a real crossing
    
    # Smoothing (B-spline with Hausdorff guard) - RELAXED for smoother curves
    smoothing_factor: float = 2.0  # Was 1.0 - more smoothing
    points_per_100m: int = 15
    max_hausdorff_dev: float = 4.0  # Was 2.0 - allow more deviation for smooth curves
    dp_tolerance_m: float = 2.0  # Was 3.0 - preserve more detail
    
    # Endpoint snapping (aligns near-miss endpoints)
    endpoint_snap_radius_m: float = 12.0  # Was 8→12 - broader snapping
    
    # Segment stitching (AGGRESSIVE to bridge gaps)
    stitch_snap_radius_m: float = 25.0  # Was 12→15→25 - bridge median gap of 25m
    stitch_max_angle_deg: float = 90.0  # Was 45→70→90 - allow perpendicular connections
    stitch_min_length_m: float = 3.0    # Was 15→5→3 - stitch even short segments
    
    # Stub pruning (MINIMAL for max recovery)
    stub_threshold_m: float = 3.0  # Was 5→4→3 - only very short stubs
    stub_max_iterations: int = 1   # Was 3→2→1 - single pass only
    
    # Jitter spike removal
    jitter_max_deviation_m: float = 15.0
    jitter_min_segment_length_m: float = 50.0
    jitter_sharp_angle_deg: float = 60.0
    
    # Loop removal
    loop_close_distance_m: float = 10.0
    loop_max_length_m: float = 80.0
    loop_shape_factor: float = 0.3
    loop_shape_max_length_m: float = 100.0
    
    # Sharp angle removal
    sharp_angle_max_deg: float = 120.0
    
    # Minimum segment length
    min_segment_length_m: float = 4.0
    
    # Output
    add_refinement_metadata: bool = True


@dataclass
class Intersection:
    """Represents a detected intersection in the VPD network."""
    center_x: float
    center_y: float
    degree: int
    edge_indices: List[int] = field(default_factory=list)
    edge_headings: List[float] = field(default_factory=list)
    endpoint_types: List[str] = field(default_factory=list)


class VPDGeometryRefiner:
    """
    Phase 3: Consolidated VPD geometry cleanup & topology refinement.
    
    Single-pass post-processing that replaces the old Phase 3 + Phase 5.
    Removes noise, resolves Z-level conflicts, merges parallels,
    smooths geometry, prunes stubs — all in one pipeline.
    """
    
    def __init__(
        self,
        skeleton_path: str,
        hpd_path: str,  # Kept for compatibility
        output_dir: str = "data",
        config: RefinementConfig = None,
    ):
        self.skeleton_path = skeleton_path
        self.hpd_path = hpd_path
        self.output_dir = output_dir
        self.config = config or RefinementConfig()
        
        self.local_crs = None
        self.to_proj = None
        self.to_wgs = None
        
        self.skeleton = None
        self.intersections: List[Intersection] = []
        self.refined_skeleton = None
        
        # Counters
        self.removed_spurs = 0
        self.merged_parallels = 0
        self.z_conflicts_resolved = 0
        self.segments_stitched = 0
        self.bridges_created = 0
        self.stubs_pruned = 0

    def load_data(self):
        """Load VPD skeleton."""
        print("  Loading data...")
        
        if not os.path.exists(self.skeleton_path):
            raise FileNotFoundError(f"Skeleton not found: {self.skeleton_path}")
        
        self.skeleton = gpd.read_file(self.skeleton_path)
        
        if self.skeleton.crs is None or self.skeleton.crs.is_geographic:
            bounds = self.skeleton.total_bounds
            cx = (bounds[0] + bounds[2]) / 2.0
            cy = (bounds[1] + bounds[3]) / 2.0
            self.local_crs = CRS.from_proj4(
                f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
                f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            )
        else:
            self.local_crs = self.skeleton.crs
        
        wgs84 = CRS.from_epsg(4326)
        self.to_proj = Transformer.from_crs(wgs84, self.local_crs, always_xy=True)
        self.to_wgs = Transformer.from_crs(self.local_crs, wgs84, always_xy=True)
        
        self.skeleton = self.skeleton.to_crs(self.local_crs)
        
        # Explode MultiLineStrings
        self.skeleton = self.skeleton.explode(index_parts=False).reset_index(drop=True)
        
        # Filter valid
        valid_mask = (
            self.skeleton.geometry.notna()
            & ~self.skeleton.geometry.is_empty
            & (self.skeleton.geom_type == "LineString")
        )
        self.skeleton = self.skeleton[valid_mask].reset_index(drop=True)
        
        # Ensure length column
        self.skeleton["length_m"] = self.skeleton.geometry.length
        
        print(f"    VPD skeleton: {len(self.skeleton)} segments")

    # ------------------------------------------------------------------
    #  INTERSECTION DETECTION
    # ------------------------------------------------------------------

    def detect_intersections(self) -> List[Intersection]:
        """Detect intersections from VPD network topology."""
        print("  Detecting intersections...")
        config = self.config
        
        if len(self.skeleton) < 2:
            return []
        
        # Collect all endpoints
        endpoints = []
        edge_map = defaultdict(list)
        
        for i, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or geom.is_empty:
                continue
            
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            
            start = coords[0]
            end = coords[-1]
            
            # Start endpoint
            ep_idx = len(endpoints)
            endpoints.append((start[0], start[1]))
            heading = bearing_from_xy(start[0], start[1], coords[1][0], coords[1][1]) if len(coords) > 1 else 0.0
            edge_map[ep_idx].append((i, "start", heading))
            
            # End endpoint
            ep_idx = len(endpoints)
            endpoints.append((end[0], end[1]))
            heading = bearing_from_xy(coords[-2][0], coords[-2][1], end[0], end[1]) if len(coords) > 1 else 0.0
            edge_map[ep_idx].append((i, "end", heading))
        
        if not endpoints:
            return []
        
        ep_arr = np.array(endpoints, dtype=np.float64)
        tree = cKDTree(ep_arr)
        
        # Union-find clustering
        parent = list(range(len(endpoints)))
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        pairs = tree.query_pairs(config.intersection_snap_m)
        for a, b in pairs:
            union(a, b)
        
        clusters = defaultdict(list)
        for i in range(len(endpoints)):
            clusters[find(i)].append(i)
        
        intersections = []
        for root, members in clusters.items():
            touching_edges = []
            for ep_idx in members:
                touching_edges.extend(edge_map[ep_idx])
            
            # Deduplicate edges
            unique_edges = {}
            for (edge_idx, ep_type, heading) in touching_edges:
                if edge_idx not in unique_edges:
                    unique_edges[edge_idx] = (ep_type, heading)
            
            degree = len(unique_edges)
            if degree >= config.min_intersection_degree:
                cluster_coords = ep_arr[members]
                cx = float(np.mean(cluster_coords[:, 0]))
                cy = float(np.mean(cluster_coords[:, 1]))
                
                intersections.append(Intersection(
                    center_x=cx,
                    center_y=cy,
                    degree=degree,
                    edge_indices=list(unique_edges.keys()),
                    edge_headings=[h for (_, h) in unique_edges.values()],
                    endpoint_types=[t for (t, _) in unique_edges.values()],
                ))
        
        print(f"    Found {len(intersections)} intersections (degree >= {config.min_intersection_degree})")
        self.intersections = intersections
        return intersections

    # ------------------------------------------------------------------
    #  SPUR REMOVAL
    # ------------------------------------------------------------------

    def remove_intersection_spurs(self) -> int:
        """
        Remove short spur segments at intersections.
        A spur is a short dead-end segment near an intersection.
        """
        print("  Removing intersection spurs...")
        config = self.config
        
        if not self.intersections:
            return 0
        
        # Build endpoint degree map
        endpoint_counts = defaultdict(int)
        for i, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or geom.is_empty:
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            
            start = (round(coords[0][0], 1), round(coords[0][1], 1))
            end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
            endpoint_counts[start] += 1
            endpoint_counts[end] += 1
        
        spurs_to_remove = set()
        
        for intersection in self.intersections:
            for edge_idx, ep_type, heading in zip(
                intersection.edge_indices,
                intersection.endpoint_types,
                intersection.edge_headings
            ):
                if edge_idx in spurs_to_remove:
                    continue
                
                if edge_idx >= len(self.skeleton):
                    continue
                
                row = self.skeleton.iloc[edge_idx]
                geom = row.geometry
                if not isinstance(geom, LineString):
                    continue
                
                length = geom.length
                if length > config.spur_max_length_m:
                    continue
                
                coords = list(geom.coords)
                
                # Check if the other endpoint is dead-end (degree 1)
                if ep_type == "start":
                    other_end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
                else:
                    other_end = (round(coords[0][0], 1), round(coords[0][1], 1))
                
                if endpoint_counts[other_end] == 1:
                    spurs_to_remove.add(edge_idx)
                    continue
                
                # Check unique heading vs other edges at intersection
                other_headings = [h for i_h, h in enumerate(intersection.edge_headings) 
                                  if intersection.edge_indices[i_h] != edge_idx]
                
                if not other_headings:
                    continue
                
                min_angle_diff = min(angle_diff_deg(heading, h) for h in other_headings)
                
                if min_angle_diff > 60 and length < config.spur_max_length_m * 0.5:
                    spurs_to_remove.add(edge_idx)
        
        if spurs_to_remove:
            keep_mask = [i not in spurs_to_remove for i in range(len(self.skeleton))]
            self.skeleton = self.skeleton[keep_mask].reset_index(drop=True)
        
        self.removed_spurs = len(spurs_to_remove)
        print(f"    Removed {self.removed_spurs} spur segments")
        return self.removed_spurs

    # ------------------------------------------------------------------
    #  HAUSDORFF-BASED PARALLEL MERGE
    # ------------------------------------------------------------------

    def merge_parallel_segments(self) -> int:
        """
        Remove duplicate/sub-segment lines using corridor-based overlap.
        
        For each segment, check if it's contained within a longer, higher-quality
        segment's corridor. NO Union-Find to avoid transitive chains.
        
        Two types of duplicates:
        1. Sub-segments: shorter line is >60% inside longer line's buffer
        2. True parallels: similar-length lines are MUTUALLY >60% inside each other
        """
        print("  Removing duplicate/sub-segment lines (corridor overlap)...")
        config = self.config
        
        if len(self.skeleton) < 2:
            return 0
        
        n = len(self.skeleton)
        geoms = self.skeleton.geometry.values
        lengths = self.skeleton.geometry.length.values
        
        # Compute direction-agnostic headings
        headings = np.array([
            (np.degrees(np.arctan2(
                np.array(g.coords)[-1, 1] - np.array(g.coords)[0, 1],
                np.array(g.coords)[-1, 0] - np.array(g.coords)[0, 0]
            )) % 180.0) if isinstance(g, LineString) and len(g.coords) >= 2 else 0.0
            for g in geoms
        ])
        
        # Quality scores (prioritize VPD, then support × length)
        has_source = "source" in self.skeleton.columns
        has_support = "weighted_support" in self.skeleton.columns
        
        scores = np.zeros(n)
        for i in range(n):
            base_score = lengths[i]
            if has_support:
                support = self.skeleton.iloc[i].get("weighted_support", 1.0)
                support = float(support) if support is not None else 1.0
                base_score *= support
            if has_source:
                src = self.skeleton.iloc[i].get("source", "VPD")
                if src == "VPD":
                    base_score *= 1.5
                elif src == "roundabout":
                    base_score *= 10.0  # Never remove roundabouts
            scores[i] = base_score
        
        buf_dist = config.parallel_hausdorff_m
        buffered_geoms = [g.buffer(buf_dist) for g in geoms]
        
        # Spatial pre-filter
        tree = STRtree(geoms)
        query_distance = buf_dist + 5.0
        left_idx, right_idx = tree.query(geoms, predicate="dwithin", distance=query_distance)
        
        # Build adjacency
        neighbors = defaultdict(list)
        for k in range(len(left_idx)):
            i, j = int(left_idx[k]), int(right_idx[k])
            if i != j:
                neighbors[i].append(j)
                neighbors[j].append(i)
        
        to_remove = set()
        overlap_count = 0
        
        for i in range(n):
            if i in to_remove:
                continue
            
            len_i = lengths[i]
            if len_i < 1.0:
                continue
            
            # Never remove roundabouts
            if has_source:
                src_i = self.skeleton.iloc[i].get("source", "")
                if src_i == "roundabout":
                    continue
            
            for j in neighbors[i]:
                if j in to_remove:
                    continue
                
                len_j = lengths[j]
                if len_j < 1.0:
                    continue
                
                if has_source:
                    src_j = self.skeleton.iloc[j].get("source", "")
                    if src_j == "roundabout":
                        continue
                
                # Heading compatibility
                h_diff = abs(headings[i] - headings[j])
                h_diff = min(h_diff, 180.0 - h_diff)
                if h_diff > config.parallel_heading_tol_deg:
                    continue
                
                # Determine shorter/longer
                if len_i <= len_j:
                    shorter_idx, longer_idx = i, j
                    shorter_len, longer_len = len_i, len_j
                    shorter_geom = geoms[i]
                    longer_buf = buffered_geoms[j]
                else:
                    shorter_idx, longer_idx = j, i
                    shorter_len, longer_len = len_j, len_i
                    shorter_geom = geoms[j]
                    longer_buf = buffered_geoms[i]
                
                # Check corridor overlap
                try:
                    inside = shorter_geom.intersection(longer_buf)
                    corridor_ratio = inside.length / shorter_len
                except Exception:
                    continue
                
                if corridor_ratio < config.parallel_min_overlap:
                    continue
                
                # Length ratio check
                len_ratio = shorter_len / longer_len
                if len_ratio < 0.3 and corridor_ratio < 0.85:
                    continue
                
                # If similar lengths, require MUTUAL overlap
                if len_ratio > 0.7:
                    try:
                        longer_geom = geoms[longer_idx]
                        shorter_buf = buffered_geoms[shorter_idx]
                        reverse_inside = longer_geom.intersection(shorter_buf)
                        reverse_ratio = reverse_inside.length / longer_len
                    except Exception:
                        continue
                    
                    if reverse_ratio < config.parallel_min_overlap * 0.8:
                        continue
                
                # Remove the one with lower score
                if scores[shorter_idx] < scores[longer_idx]:
                    to_remove.add(shorter_idx)
                else:
                    to_remove.add(longer_idx)
                overlap_count += 1
        
        keep_mask = [i not in to_remove for i in range(n)]
        self.skeleton = self.skeleton[keep_mask].reset_index(drop=True)
        
        removed = len(to_remove)
        self.merged_parallels = removed
        print(f"    Found {overlap_count} duplicate pairs, removed {removed} segments")
        return removed

    # ------------------------------------------------------------------
    #  Z-LEVEL CROSSING RESOLUTION
    # ------------------------------------------------------------------

    def resolve_z_level_crossings(self) -> int:
        """
        Resolve Z-level crossings by splitting segments at crossing points
        and assigning different z_level attributes.
        
        Unlike the old detect-only approach, this actually disconnects crossing
        segments so they don't create false junctions. Uses altitude data from
        Phase 2 (mean_altitude column) when available to determine which
        segment is the overpass vs underpass.
        """
        print("  Resolving Z-level crossings...")
        config = self.config
        
        if len(self.skeleton) < 2:
            return 0
        
        # Initialize z_level column
        if "z_level" not in self.skeleton.columns:
            self.skeleton["z_level"] = 0
        
        has_altitude = "mean_altitude" in self.skeleton.columns
        n = len(self.skeleton)
        
        # Use spatial index for efficient crossing detection
        geoms = self.skeleton.geometry.values
        tree = STRtree(geoms)
        left_idx, right_idx = tree.query(geoms, predicate="crosses")
        
        crossings_resolved = 0
        segments_to_split = []  # (original_idx, split_point, assigned_z)
        z_assignments = {}  # idx -> z_level
        
        # Process unique pairs only
        seen_pairs = set()
        for k in range(len(left_idx)):
            i, j = int(left_idx[k]), int(right_idx[k])
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            geom_i = geoms[i]
            geom_j = geoms[j]
            
            if not isinstance(geom_i, LineString) or not isinstance(geom_j, LineString):
                continue
            
            try:
                intersection = geom_i.intersection(geom_j)
            except Exception:
                continue
            
            if intersection.is_empty:
                continue
            
            # Only process point crossings (not shared segments)
            if not isinstance(intersection, Point):
                continue
            
            # Verify it's a mid-segment crossing (not at endpoints)
            coords_i = list(geom_i.coords)
            coords_j = list(geom_j.coords)
            
            is_endpoint_i = (
                (abs(intersection.x - coords_i[0][0]) < 1 and abs(intersection.y - coords_i[0][1]) < 1) or
                (abs(intersection.x - coords_i[-1][0]) < 1 and abs(intersection.y - coords_i[-1][1]) < 1)
            )
            is_endpoint_j = (
                (abs(intersection.x - coords_j[0][0]) < 1 and abs(intersection.y - coords_j[0][1]) < 1) or
                (abs(intersection.x - coords_j[-1][0]) < 1 and abs(intersection.y - coords_j[-1][1]) < 1)
            )
            
            if is_endpoint_i or is_endpoint_j:
                continue  # Junction, not crossing
            
            # Check heading angle — real crossings have significant angle
            heading_i = bearing_from_xy(coords_i[0][0], coords_i[0][1],
                                         coords_i[-1][0], coords_i[-1][1])
            heading_j = bearing_from_xy(coords_j[0][0], coords_j[0][1],
                                         coords_j[-1][0], coords_j[-1][1])
            
            if angle_diff_deg(heading_i, heading_j) < config.z_crossing_angle_min_deg:
                continue
            
            # Assign z_levels: use altitude if available, otherwise heuristic
            if has_altitude:
                alt_i = self.skeleton.iloc[i].get("mean_altitude", 0.0)
                alt_j = self.skeleton.iloc[j].get("mean_altitude", 0.0)
                alt_i = float(alt_i) if alt_i is not None and not np.isnan(alt_i) else 0.0
                alt_j = float(alt_j) if alt_j is not None and not np.isnan(alt_j) else 0.0
                
                if alt_i >= alt_j:
                    z_assignments[i] = max(z_assignments.get(i, 0), 1)
                    z_assignments.setdefault(j, 0)
                else:
                    z_assignments[j] = max(z_assignments.get(j, 0), 1)
                    z_assignments.setdefault(i, 0)
            else:
                # Heuristic: longer segment is more likely the main road (ground level)
                if geom_i.length >= geom_j.length:
                    z_assignments.setdefault(i, 0)
                    z_assignments[j] = max(z_assignments.get(j, 0), 1)
                else:
                    z_assignments.setdefault(j, 0)
                    z_assignments[i] = max(z_assignments.get(i, 0), 1)
            
            crossings_resolved += 1
        
        # Apply z_level assignments
        for idx, z_val in z_assignments.items():
            if idx < len(self.skeleton):
                self.skeleton.at[self.skeleton.index[idx], "z_level"] = z_val
        
        self.z_conflicts_resolved = crossings_resolved
        print(f"    Resolved {crossings_resolved} Z-level crossings ({len(z_assignments)} segments assigned z_level)")
        return crossings_resolved

    # ------------------------------------------------------------------
    #  SHARP ANGLE REMOVAL
    # ------------------------------------------------------------------

    def remove_sharp_angles(self) -> int:
        """Remove vertices with >120° direction change (bird-nesting fix)."""
        print("  Removing sharp angle vertices...")
        
        max_angle = self.config.sharp_angle_max_deg
        vertices_removed = 0
        new_geoms = []
        
        for idx, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or geom.is_empty or len(geom.coords) < 3:
                new_geoms.append(geom)
                continue
            
            coords = list(geom.coords)
            clean_coords = [coords[0]]
            
            for i in range(1, len(coords) - 1):
                b1 = bearing_from_xy(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
                b2 = bearing_from_xy(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
                
                if angle_diff_deg(b1, b2) < max_angle:
                    clean_coords.append(coords[i])
                else:
                    vertices_removed += 1
            
            clean_coords.append(coords[-1])
            new_geoms.append(LineString(clean_coords) if len(clean_coords) >= 2 else geom)
        
        self.skeleton = self.skeleton.copy()
        self.skeleton["geometry"] = new_geoms
        self._filter_valid_geometries()
        
        print(f"    Removed {vertices_removed} sharp-angle vertices")
        return vertices_removed

    # ------------------------------------------------------------------
    #  SMOOTHING (Douglas-Peucker + B-spline with Hausdorff guard)
    # ------------------------------------------------------------------

    def smooth_geometry(self) -> int:
        """
        Two-stage smoothing:
          1. Douglas-Peucker simplification to remove zig-zag noise
          2. B-spline smoothing with tight Hausdorff guard to prevent drift
        """
        print("  Smoothing geometry (D-P + B-spline)...")
        config = self.config
        
        smoothed_count = 0
        dp_count = 0
        new_geoms = []
        
        for geom in self.skeleton.geometry:
            if not isinstance(geom, LineString) or geom.is_empty:
                new_geoms.append(geom)
                continue
            
            n_coords = len(geom.coords)
            
            # Stage 1: Douglas-Peucker
            if n_coords >= 4:
                try:
                    simplified = geom.simplify(config.dp_tolerance_m, preserve_topology=True)
                    if not simplified.is_empty and simplified.length > 0:
                        geom = simplified
                        n_coords = len(geom.coords)
                        dp_count += 1
                except Exception:
                    pass
            
            # Stage 2: B-spline (only for lines with enough points)
            if n_coords >= 4:
                try:
                    x, y = geom.xy
                    x, y = list(x), list(y)
                    
                    n_out = max(4, int(geom.length / 100.0 * config.points_per_100m))
                    n_out = min(n_out, 150)
                    
                    k = min(3, n_coords - 1)
                    tck, u = splprep([x, y], s=config.smoothing_factor, k=k)
                    new_pts = splev(np.linspace(0, 1, n_out), tck)
                    
                    new_line = LineString(zip(new_pts[0], new_pts[1]))
                    
                    if new_line.hausdorff_distance(geom) < config.max_hausdorff_dev:
                        geom = new_line
                        smoothed_count += 1
                except Exception:
                    pass
            
            new_geoms.append(geom)
        
        self.skeleton["geometry"] = new_geoms
        
        print(f"    D-P simplified: {dp_count}, B-spline smoothed: {smoothed_count}")
        return smoothed_count

    # ------------------------------------------------------------------
    #  ENDPOINT SNAPPING (aligns near-miss endpoints)
    # ------------------------------------------------------------------

    def snap_segment_endpoints(self) -> int:
        """
        Snap close segment endpoints to common locations.
        
        This fixes the "near-miss" problem where endpoints are 2-10m apart
        but should be at the same location. After snapping, segments sharing
        the same logical intersection will have identical endpoint coordinates.
        """
        print("  Snapping segment endpoints...")
        config = self.config
        
        if len(self.skeleton) < 2:
            return 0
        
        # Collect all endpoints
        endpoints = []  # (x, y, seg_idx, is_start)
        for i, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or len(geom.coords) < 2:
                continue
            coords = list(geom.coords)
            endpoints.append((coords[0][0], coords[0][1], i, True))
            endpoints.append((coords[-1][0], coords[-1][1], i, False))
        
        if len(endpoints) < 2:
            return 0
        
        ep_coords = np.array([(e[0], e[1]) for e in endpoints])
        tree = cKDTree(ep_coords)
        
        # Union-Find clustering
        parent = list(range(len(endpoints)))
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        
        # Cluster endpoints within snap radius
        pairs = tree.query_pairs(config.endpoint_snap_radius_m)
        for a, b in pairs:
            union(a, b)
        
        # Compute cluster centroids
        clusters = defaultdict(list)
        for i in range(len(endpoints)):
            clusters[find(i)].append(i)
        
        # Build snap map: endpoint index -> new (x, y)
        snap_map = {}
        snapped_count = 0
        
        for root, members in clusters.items():
            if len(members) <= 1:
                continue
            
            # Compute centroid
            member_coords = ep_coords[members]
            cx = np.mean(member_coords[:, 0])
            cy = np.mean(member_coords[:, 1])
            
            for idx in members:
                snap_map[idx] = (cx, cy)
                snapped_count += 1
        
        if not snap_map:
            print("    No endpoints to snap")
            return 0
        
        # Apply snapping to geometries
        new_geoms = []
        for i, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or len(geom.coords) < 2:
                new_geoms.append(geom)
                continue
            
            coords = list(geom.coords)
            
            # Find this segment's endpoint indices
            start_idx = None
            end_idx = None
            for ep_idx, ep in enumerate(endpoints):
                if ep[2] == i:
                    if ep[3]:  # is_start
                        start_idx = ep_idx
                    else:
                        end_idx = ep_idx
            
            # Apply snapping if needed
            modified = False
            if start_idx is not None and start_idx in snap_map:
                coords[0] = snap_map[start_idx]
                modified = True
            if end_idx is not None and end_idx in snap_map:
                coords[-1] = snap_map[end_idx]
                modified = True
            
            if modified:
                new_geoms.append(LineString(coords))
            else:
                new_geoms.append(geom)
        
        self.skeleton["geometry"] = new_geoms
        self._filter_valid_geometries()
        
        print(f"    Snapped {snapped_count} endpoints to {len([c for c in clusters.values() if len(c) > 1])} common locations")
        return snapped_count
    #  SEGMENT STITCHING (fixes fragmentation)
    # ------------------------------------------------------------------

    def stitch_fragmented_segments(self) -> int:
        """
        Stitch nearby segment endpoints to create continuous roads.
        
        Finds pairs of endpoints that are:
        1. Close together (within snap radius)
        2. Heading-compatible (would form smooth continuation)
        3. Not already connected
        
        Then merges them into single continuous LineStrings.
        """
        print("  Stitching fragmented segments...")
        config = self.config
        
        if len(self.skeleton) < 2:
            return 0
        
        # Build endpoint index
        endpoints = []  # (x, y, seg_idx, is_start, heading)
        
        for i, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or len(geom.coords) < 2:
                continue
            
            coords = list(geom.coords)
            
            # Start endpoint
            start_heading = bearing_from_xy(
                coords[0][0], coords[0][1],
                coords[1][0], coords[1][1]
            )
            endpoints.append((coords[0][0], coords[0][1], i, True, start_heading))
            
            # End endpoint  
            end_heading = bearing_from_xy(
                coords[-2][0], coords[-2][1],
                coords[-1][0], coords[-1][1]
            )
            endpoints.append((coords[-1][0], coords[-1][1], i, False, end_heading))
        
        if len(endpoints) < 2:
            return 0
        
        ep_coords = np.array([(e[0], e[1]) for e in endpoints])
        tree = cKDTree(ep_coords)
        
        # Find stitch candidates
        pairs = tree.query_pairs(config.stitch_snap_radius_m)
        
        # Build endpoint degree map (to identify dead ends)
        ep_degree = Counter()
        for ep in endpoints:
            key = (round(ep[0], 1), round(ep[1], 1))
            ep_degree[key] += 1
        
        # Union-Find for merging
        parent = {i: i for i in range(len(self.skeleton))}
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                return True
            return False
        
        merge_pairs = []  # (seg_a, is_start_a, seg_b, is_start_b, dist)
        
        for a_idx, b_idx in pairs:
            ep_a = endpoints[a_idx]
            ep_b = endpoints[b_idx]
            
            seg_a, is_start_a, heading_a = ep_a[2], ep_a[3], ep_a[4]
            seg_b, is_start_b, heading_b = ep_b[2], ep_b[3], ep_b[4]
            
            # Don't stitch same segment
            if seg_a == seg_b:
                continue
            
            # Only stitch dead-end endpoints (degree 1)
            key_a = (round(ep_a[0], 1), round(ep_a[1], 1))
            key_b = (round(ep_b[0], 1), round(ep_b[1], 1))
            if ep_degree[key_a] > 1 and ep_degree[key_b] > 1:
                continue  # Both endpoints are already connected elsewhere
            
            # Check lengths - don't stitch very short segments
            len_a = self.skeleton.iloc[seg_a].geometry.length
            len_b = self.skeleton.iloc[seg_b].geometry.length
            if len_a < config.stitch_min_length_m or len_b < config.stitch_min_length_m:
                continue
            
            # Check heading compatibility
            # For stitching: end of A should point toward start of B (or vice versa)
            if is_start_a:
                heading_a_eff = (heading_a + 180.0) % 360.0
            else:
                heading_a_eff = heading_a
            
            if is_start_b:
                heading_b_eff = heading_b
            else:
                heading_b_eff = (heading_b + 180.0) % 360.0
            
            angle_diff = angle_diff_deg(heading_a_eff, heading_b_eff)
            if angle_diff > config.stitch_max_angle_deg:
                continue
            
            # Compute distance
            dist = np.hypot(ep_a[0] - ep_b[0], ep_a[1] - ep_b[1])
            
            # Good stitch candidate
            merge_pairs.append((seg_a, is_start_a, seg_b, is_start_b, dist))
        
        if not merge_pairs:
            print("    No stitch candidates found")
            return 0
        
        # Sort by distance (stitch closest pairs first)
        merge_pairs.sort(key=lambda x: x[4])
        
        # Track which segments get merged
        merged_count = 0
        merge_map = {}  # seg_idx -> list of (other_seg, is_start_self, is_start_other)
        
        for seg_a, is_start_a, seg_b, is_start_b, dist in merge_pairs:
            if find(seg_a) == find(seg_b):
                continue  # Already in same group
            
            union(seg_a, seg_b)
            
            if seg_a not in merge_map:
                merge_map[seg_a] = []
            if seg_b not in merge_map:
                merge_map[seg_b] = []
            
            merge_map[seg_a].append((seg_b, is_start_a, is_start_b))
            merge_map[seg_b].append((seg_a, is_start_b, is_start_a))
            merged_count += 1
        
        # Group segments by merge set
        groups = defaultdict(set)
        for i in range(len(self.skeleton)):
            groups[find(i)].add(i)
        
        # Only process groups with multiple segments
        multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
        
        if not multi_groups:
            print("    No stitch candidates found")
            return 0
        
        # Merge each group into single LineString
        new_rows = []
        merged_indices = set()
        
        for root, members in multi_groups.items():
            merged_indices.update(members)
            members_list = list(members)
            
            # Simple chain: order segments by connectivity
            # Start with first segment, then find connected segments
            ordered_segments = []
            remaining = set(members_list)
            
            # Start with segment that has a dead-end (or any if none)
            current = members_list[0]
            remaining.remove(current)
            ordered_segments.append((current, False))  # (seg_idx, reverse)
            
            # Chain segments together
            while remaining:
                found = False
                current_seg = ordered_segments[-1][0]
                current_reversed = ordered_segments[-1][1]
                
                current_geom = self.skeleton.iloc[current_seg].geometry
                current_coords = list(current_geom.coords)
                if current_reversed:
                    current_end = current_coords[0]
                else:
                    current_end = current_coords[-1]
                
                best_next = None
                best_dist = float('inf')
                best_reverse = False
                
                for next_seg in remaining:
                    next_geom = self.skeleton.iloc[next_seg].geometry
                    next_coords = list(next_geom.coords)
                    
                    # Check distance to start and end of next segment
                    dist_to_start = np.hypot(
                        current_end[0] - next_coords[0][0],
                        current_end[1] - next_coords[0][1]
                    )
                    dist_to_end = np.hypot(
                        current_end[0] - next_coords[-1][0],
                        current_end[1] - next_coords[-1][1]
                    )
                    
                    if dist_to_start < best_dist and dist_to_start < config.stitch_snap_radius_m:
                        best_dist = dist_to_start
                        best_next = next_seg
                        best_reverse = False
                    if dist_to_end < best_dist and dist_to_end < config.stitch_snap_radius_m:
                        best_dist = dist_to_end
                        best_next = next_seg
                        best_reverse = True
                
                if best_next is not None:
                    ordered_segments.append((best_next, best_reverse))
                    remaining.remove(best_next)
                    found = True
                
                if not found:
                    break  # Can't chain further
            
            # Build merged geometry
            all_coords = []
            for seg_idx, reverse in ordered_segments:
                geom = self.skeleton.iloc[seg_idx].geometry
                coords = list(geom.coords)
                if reverse:
                    coords = coords[::-1]
                
                if all_coords:
                    # Skip first point if close to last (avoid duplicate)
                    if len(all_coords) > 0:
                        last_pt = all_coords[-1]
                        dist = np.hypot(last_pt[0] - coords[0][0], last_pt[1] - coords[0][1])
                        if dist < config.stitch_snap_radius_m:
                            coords = coords[1:]
                
                all_coords.extend(coords)
            
            if len(all_coords) >= 2:
                merged_geom = LineString(all_coords)
                
                # Copy attributes from first segment
                first_row = self.skeleton.iloc[members_list[0]].copy()
                first_row["geometry"] = merged_geom
                first_row["length_m"] = merged_geom.length
                new_rows.append(first_row)
        
        # Remove merged segments, add merged results
        if merged_indices:
            keep_mask = [i not in merged_indices for i in range(len(self.skeleton))]
            self.skeleton = self.skeleton[keep_mask].reset_index(drop=True)
            
            if new_rows:
                merged_gdf = gpd.GeoDataFrame(new_rows, crs=self.skeleton.crs)
                self.skeleton = pd.concat([self.skeleton, merged_gdf], ignore_index=True)
        
        segments_stitched = len(merged_indices) - len(new_rows)
        print(f"    Stitched {len(merged_indices)} segments into {len(new_rows)} merged roads")
        return segments_stitched

    # ------------------------------------------------------------------
    #  GAP BRIDGING (connects dead-ends to nearby segments)
    # ------------------------------------------------------------------

    def bridge_dead_end_gaps(self) -> int:
        """
        Bridge gaps between dead-end segments and nearby roads.
        
        For each dead-end (degree-1 endpoint), if it points toward
        another segment within bridging distance, create a small
        connector line to close the gap.
        """
        print("  Bridging dead-end gaps...")
        config = self.config
        
        if len(self.skeleton) < 2:
            return 0
        
        # Build endpoint degree map
        endpoint_degree = Counter()
        endpoint_to_seg = {}  # (x, y) -> list of (seg_idx, is_start, heading)
        
        for i, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or len(geom.coords) < 2:
                continue
            
            coords = list(geom.coords)
            
            # Start
            start_key = (round(coords[0][0], 1), round(coords[0][1], 1))
            endpoint_degree[start_key] += 1
            if start_key not in endpoint_to_seg:
                endpoint_to_seg[start_key] = []
            start_heading = bearing_from_xy(coords[0][0], coords[0][1], coords[1][0], coords[1][1])
            endpoint_to_seg[start_key].append((i, True, start_heading, coords[0]))
            
            # End
            end_key = (round(coords[-1][0], 1), round(coords[-1][1], 1))
            endpoint_degree[end_key] += 1
            if end_key not in endpoint_to_seg:
                endpoint_to_seg[end_key] = []
            end_heading = bearing_from_xy(coords[-2][0], coords[-2][1], coords[-1][0], coords[-1][1])
            endpoint_to_seg[end_key].append((i, False, end_heading, coords[-1]))
        
        # Find dead-ends (degree 1)
        dead_ends = []  # (exact_coord, heading pointing outward, seg_idx)
        for ep_key, count in endpoint_degree.items():
            if count != 1:
                continue
            for (seg_idx, is_start, heading, exact_coord) in endpoint_to_seg[ep_key]:
                # Outward heading: if start, reverse; if end, keep
                if is_start:
                    outward_heading = (heading + 180.0) % 360.0
                else:
                    outward_heading = heading
                dead_ends.append((exact_coord, outward_heading, seg_idx, is_start))
        
        if not dead_ends:
            print("    No dead-ends to bridge")
            return 0
        
        # Build spatial index of all segment geometries for nearest-segment queries
        geoms = list(self.skeleton.geometry)
        tree = STRtree(geoms)
        
        # For each dead-end, try to bridge to nearest segment
        bridges_created = 0
        new_connectors = []
        bridge_distance = config.stitch_snap_radius_m * 1.5  # Allow slightly longer bridges
        
        for (dead_coord, outward_heading, dead_seg_idx, is_start) in dead_ends:
            dead_pt = Point(dead_coord)
            
            # Find nearby segments
            nearby_idxs = tree.query(dead_pt.buffer(bridge_distance))
            
            best_target = None
            best_dist = float('inf')
            best_target_pt = None
            
            for idx in nearby_idxs:
                if idx == dead_seg_idx:
                    continue  # Don't bridge to self
                
                target_geom = geoms[idx]
                if not isinstance(target_geom, LineString):
                    continue
                
                # Find nearest point on target segment
                nearest_pt = target_geom.interpolate(target_geom.project(dead_pt))
                dist = dead_pt.distance(nearest_pt)
                
                if dist < 1.0:
                    continue  # Already touching, no bridge needed
                
                if dist >= bridge_distance:
                    continue
                
                # Check heading compatibility - dead-end should point roughly toward target
                to_target_heading = bearing_from_xy(
                    dead_coord[0], dead_coord[1],
                    nearest_pt.x, nearest_pt.y
                )
                angle_diff = angle_diff_deg(outward_heading, to_target_heading)
                
                if angle_diff > config.stitch_max_angle_deg:
                    continue  # Not pointing toward target
                
                if dist < best_dist:
                    best_dist = dist
                    best_target = idx
                    best_target_pt = (nearest_pt.x, nearest_pt.y)
            
            if best_target is not None and best_target_pt is not None:
                # Create connector line
                connector = LineString([dead_coord, best_target_pt])
                new_connectors.append({
                    "geometry": connector,
                    "length_m": connector.length,
                    "source": "bridge",
                    "weighted_support": 1.0,
                })
                bridges_created += 1
        
        # Add connectors to skeleton
        if new_connectors:
            connector_gdf = gpd.GeoDataFrame(new_connectors, crs=self.skeleton.crs)
            self.skeleton = pd.concat([self.skeleton, connector_gdf], ignore_index=True)
        
        print(f"    Created {bridges_created} gap bridges")
        return bridges_created

    # ------------------------------------------------------------------
    #  JITTER SPIKE REMOVAL
    # ------------------------------------------------------------------

    def remove_jitter_spikes(self) -> int:
        """Remove vertices deviating far from the main trajectory with sharp angles."""
        print("  Removing jitter spikes...")
        config = self.config
        
        spikes_removed = 0
        new_geoms = []
        
        for idx, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or geom.is_empty:
                new_geoms.append(geom)
                continue
            
            coords = list(geom.coords)
            if len(coords) < 4 or geom.length < config.jitter_min_segment_length_m:
                new_geoms.append(geom)
                continue
            
            main_line = LineString([coords[0], coords[-1]])
            clean_coords = [coords[0]]
            
            for i in range(1, len(coords) - 1):
                pt = Point(coords[i])
                deviation = pt.distance(main_line)
                
                if deviation <= config.jitter_max_deviation_m:
                    clean_coords.append(coords[i])
                elif len(clean_coords) >= 1:
                    # Only remove if also creates sharp angle
                    prev = clean_coords[-1]
                    next_pt = coords[i + 1] if i + 1 < len(coords) else coords[-1]
                    
                    b1 = bearing_from_xy(prev[0], prev[1], coords[i][0], coords[i][1])
                    b2 = bearing_from_xy(coords[i][0], coords[i][1], next_pt[0], next_pt[1])
                    
                    if angle_diff_deg(b1, b2) > config.jitter_sharp_angle_deg:
                        spikes_removed += 1
                    else:
                        clean_coords.append(coords[i])
                else:
                    clean_coords.append(coords[i])
            
            clean_coords.append(coords[-1])
            new_geoms.append(LineString(clean_coords) if len(clean_coords) >= 2 else geom)
        
        self.skeleton = self.skeleton.copy()
        self.skeleton["geometry"] = new_geoms
        self._filter_valid_geometries()
        
        print(f"    Removed {spikes_removed} jitter spike vertices")
        return spikes_removed

    # ------------------------------------------------------------------
    #  LOOP SEGMENT REMOVAL
    # ------------------------------------------------------------------

    def remove_loop_segments(self) -> int:
        """Remove segments forming tight loops or doubling back on themselves."""
        print("  Removing loop/self-intersecting segments...")
        config = self.config
        
        to_remove = []
        
        for idx, row in self.skeleton.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or geom.is_empty or len(geom.coords) < 3:
                continue
            
            start = Point(geom.coords[0])
            end = Point(geom.coords[-1])
            
            # Tiny closed loop
            if start.distance(end) < config.loop_close_distance_m and geom.length < config.loop_max_length_m:
                to_remove.append(idx)
                continue
            
            # Doubled-back shape
            straight_dist = start.distance(end)
            if straight_dist > 0:
                shape_factor = straight_dist / geom.length
                if shape_factor < config.loop_shape_factor and geom.length < config.loop_shape_max_length_m:
                    to_remove.append(idx)
        
        if to_remove:
            self.skeleton = self.skeleton.drop(to_remove).reset_index(drop=True)
        
        print(f"    Removed {len(to_remove)} loop/doubled-back segments")
        return len(to_remove)

    # ------------------------------------------------------------------
    #  ITERATIVE STUB PRUNING (consolidated from old Phase 5)
    # ------------------------------------------------------------------

    def prune_stubs(self) -> int:
        """
        Iteratively remove dead-end segments shorter than threshold.
        Uses NetworkX graph to correctly identify degree-1 nodes.
        """
        print(f"  Pruning stubs (< {self.config.stub_threshold_m}m)...")
        config = self.config
        
        def round_coord(c):
            return (round(c[0], 1), round(c[1], 1))
        
        has_source = "source" in self.skeleton.columns
        total_removed = 0
        iteration = 0
        
        while iteration < config.stub_max_iterations:
            iteration += 1
            
            G = nx.Graph()
            for idx, row in self.skeleton.iterrows():
                geom = row.geometry
                if not isinstance(geom, LineString) or len(geom.coords) < 2:
                    continue
                u = round_coord(geom.coords[0])
                v = round_coord(geom.coords[-1])
                if u == v:
                    continue
                source = row.get("source", "VPD") if has_source else "VPD"
                G.add_edge(u, v, idx=idx, length=geom.length, source=source)
            
            stubs = set()
            for node in G.nodes():
                if G.degree(node) == 1:
                    neighbor = list(G.neighbors(node))[0]
                    data = G[node][neighbor]
                    length = data.get("length", float("inf"))
                    source = data.get("source", "VPD")
                    
                    # Never prune roundabout segments
                    if source == "roundabout":
                        continue
                    
                    if length < config.stub_threshold_m:
                        stubs.add(data["idx"])
            
            if not stubs:
                break
            
            self.skeleton = self.skeleton.drop(index=list(stubs)).reset_index(drop=True)
            total_removed += len(stubs)
        
        self.stubs_pruned = total_removed
        print(f"    Removed {total_removed} stubs over {iteration} iteration(s)")
        return total_removed

    # ------------------------------------------------------------------
    #  GEOMETRY CLEANING
    # ------------------------------------------------------------------

    def _filter_valid_geometries(self):
        """Remove invalid/empty geometries (internal helper)."""
        valid_mask = (
            self.skeleton.geometry.notna()
            & ~self.skeleton.geometry.is_empty
            & self.skeleton.geometry.is_valid
        )
        self.skeleton = self.skeleton[valid_mask].reset_index(drop=True)

    def clean_geometry(self):
        """Final geometry cleaning: remove empty, invalid, and very short segments."""
        print("  Cleaning geometry...")
        before = len(self.skeleton)
        
        self._filter_valid_geometries()
        
        # Remove very short segments
        self.skeleton = self.skeleton[
            self.skeleton.geometry.length >= self.config.min_segment_length_m
        ].reset_index(drop=True)
        
        removed = before - len(self.skeleton)
        if removed > 0:
            print(f"    Removed {removed} invalid/short geometries")

    # ------------------------------------------------------------------
    #  EXPORT
    # ------------------------------------------------------------------

    def export(self):
        """Export refined skeleton."""
        output_path = os.path.join(self.output_dir, "interim_refined_skeleton_phase3.gpkg")
        print(f"  Exporting refined skeleton to {output_path}...")
        
        if self.skeleton.crs != CRS.from_epsg(4326):
            output_gdf = self.skeleton.to_crs("EPSG:4326")
        else:
            output_gdf = self.skeleton.copy()
        
        output_gdf.to_file(output_path, driver="GPKG")
        self.refined_skeleton = output_gdf
        
        print(f"    Exported {len(output_gdf)} segments")
        return output_path

    # ------------------------------------------------------------------
    #  MAIN PIPELINE
    # ------------------------------------------------------------------

    def run(self):
        """Execute consolidated VPD geometry refinement pipeline."""
        print("=" * 60)
        print("  PHASE 3: VPD Geometry Cleanup & Topology Refinement")
        print("=" * 60)
        t0 = time.time()
        
        self.load_data()
        
        if len(self.skeleton) == 0:
            print("  No skeleton segments found. Exiting.")
            return None
        
        original_count = len(self.skeleton)
        print(f"    [DIAG] After load: {len(self.skeleton)} segments")
        
        # Step 1: Detect intersections
        self.detect_intersections()
        print(f"    [DIAG] After detect_intersections: {len(self.skeleton)} segments")
        
        # Step 2: Remove spurs at intersections
        self.remove_intersection_spurs()
        print(f"    [DIAG] After remove_intersection_spurs: {len(self.skeleton)} segments")
        
        # Step 3: Merge parallel overlapping segments (Hausdorff-based)
        self.merge_parallel_segments()
        print(f"    [DIAG] After merge_parallel_segments: {len(self.skeleton)} segments")
        
        # Step 4: Resolve Z-level crossings (split + assign z_level)
        self.resolve_z_level_crossings()
        print(f"    [DIAG] After resolve_z_level_crossings: {len(self.skeleton)} segments")
        
        # Step 5: Remove sharp angle vertices
        self.remove_sharp_angles()
        print(f"    [DIAG] After remove_sharp_angles: {len(self.skeleton)} segments")
        
        # Step 6: Smooth geometry (D-P + B-spline)
        self.smooth_geometry()
        print(f"    [DIAG] After smooth_geometry: {len(self.skeleton)} segments")
        
        # Step 6.5: Snap endpoints + stitch fragmented segments (pass 1)
        self.snap_segment_endpoints()
        print(f"    [DIAG] After snap_segment_endpoints: {len(self.skeleton)} segments")
        self.segments_stitched = self.stitch_fragmented_segments()
        print(f"    [DIAG] After stitch_fragmented_segments: {len(self.skeleton)} segments")
        self.bridges_created = self.bridge_dead_end_gaps()
        print(f"    [DIAG] After bridge_dead_end_gaps: {len(self.skeleton)} segments")
        
        # Step 7: Remove jitter spikes
        self.remove_jitter_spikes()
        print(f"    [DIAG] After remove_jitter_spikes: {len(self.skeleton)} segments")
        
        # Step 8: Remove loop/self-intersecting segments
        self.remove_loop_segments()
        print(f"    [DIAG] After remove_loop_segments: {len(self.skeleton)} segments")
        
        # Step 9: Prune dead-end stubs (iterative)
        self.prune_stubs()
        print(f"    [DIAG] After prune_stubs: {len(self.skeleton)} segments")
        
        # Step 9.5: Snap endpoints + stitch + bridge again (pass 2)
        self.snap_segment_endpoints()
        self.segments_stitched += self.stitch_fragmented_segments()
        self.bridges_created += self.bridge_dead_end_gaps()
        print(f"    [DIAG] After pass 2 stitching: {len(self.skeleton)} segments")
        
        # Step 10: Clean geometry
        self.clean_geometry()
        
        # Export
        self.export()
        
        elapsed = time.time() - t0
        final_count = len(self.refined_skeleton)
        
        print(f"\nPhase 3 complete in {elapsed:.1f}s")
        print(f"  Input segments:      {original_count}")
        print(f"  Spurs removed:       {self.removed_spurs}")
        print(f"  Parallels merged:    {self.merged_parallels}")
        print(f"  Z-level resolved:    {self.z_conflicts_resolved}")
        print(f"  Segments stitched:   {self.segments_stitched}")
        print(f"  Bridges created:     {self.bridges_created}")
        print(f"  Stubs pruned:        {self.stubs_pruned}")
        print(f"  Output segments:     {final_count}")
        
        gc.collect()
        return self.refined_skeleton


# Aliases for backwards compatibility
ProbeRecovery = VPDGeometryRefiner
RoadsterProbeRecovery = VPDGeometryRefiner
RoadsterConfig = RefinementConfig


if __name__ == "__main__":
    SKELETON_FILE = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
    HPD_FILE = os.path.join(PROJECT_ROOT, "data", "interim_hpd_phase1.gpkg")
    
    refiner = VPDGeometryRefiner(SKELETON_FILE, HPD_FILE)
    refiner.run()
