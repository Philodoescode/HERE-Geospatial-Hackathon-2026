"""
Phase 2: VPD Centerline Generation via Kharita-style Clustering

Replaces KDE-based rasterization with heading-aware incremental clustering
for faster, cleaner centerline generation.

Algorithm:
  1. Load and project VPD traces
  2. Sample traces at regular intervals with heading computation
  3. Heading-aware incremental clustering (Kharita-style)
  4. Build directed co-occurrence graph from trace transitions
  5. Three-pass edge pruning (support, direction-conflict, transitive)
  6. Centerline stitching with turn-preserving smoothing
  7. Candidate selection and quality filtering
"""

import gc
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely import get_coordinates
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import algorithm utilities
from src.algorithms.dynamic_weighting import (
    DynamicWeightConfig,
    apply_dynamic_weighting_to_edges,
    compute_trace_weight,
)
from src.algorithms.centerline_utils import (
    angle_diff_deg,
    bearing_from_xy,
    sample_line_projected,
    smooth_polyline_preserve_turns,
    shortest_alternative_with_hop_limit,
    stitch_centerline_paths,
    interpolate_edge_with_traces,
    detect_high_curvature_zones,
    separate_z_levels,
    compute_curvature_at_point,
)
from src.algorithms.roundabout_detection import (
    RoundaboutConfig,
    RoundaboutDetector,
    RoundaboutPostFilterConfig,
    detect_roundabouts_from_gdf,
    validate_roundabouts_post_detection,
)
from src.algorithms.curve_smoothing import (
    simplify_and_smooth_centerline,
    merge_nearby_points,
)
from src.algorithms.intersection_detection import (
    IntersectionDetector,
    IntersectionNode,
    detect_intersections,
    split_lines_at_intersections,
)
from src.algorithms.segment_averaging import (
    SegmentAveragingConfig,
    SegmentGrouper,
    SegmentAverager,
    average_segment_groups,
)
from src.algorithms.topology_builder import (
    TopologyConfig,
    TopologyBuilder,
    build_topology,
)


@dataclass
class KharitaConfig:
    """Configuration for Kharita-style clustering algorithm."""
    
    # Trace sampling
    sample_spacing_m: float = 8.0
    max_points_per_trace: int = 120
    vpd_base_weight: float = 1.2
    
    # Heading-aware clustering
    cluster_radius_m: float = 10.0
    heading_tolerance_deg: float = 55.0  # Increased for better curve handling
    heading_distance_weight_m: float = 0.18  # Reduced to favor spatial proximity on curves
    min_cluster_points: int = 1
    
    # Z-level handling (for bridges/overpasses at interchanges)
    enable_z_level_separation: bool = True
    z_separation_threshold_m: float = 3.0  # Min altitude difference to consider separate
    z_level_cluster_penalty: float = 50.0  # Penalty for crossing Z-levels in clustering
    
    # Curve-aware centerline generation (fixes cloverleaf chord cutting)
    enable_curve_interpolation: bool = True
    curve_min_edge_length_m: float = 25.0  # Only interpolate edges longer than this
    curve_max_deviation_m: float = 3.0  # When trace points deviate more than this, use curve
    curve_curvature_threshold: float = 0.015  # 1/67m radius = typical interchange ramp
    
    # Edge extraction
    max_transition_distance_m: float = 50.0
    
    # Edge pruning (relaxed for better recovery)
    min_edge_support: float = 1.0  # Reduced from 1.5 for better low-density road recovery
    reverse_edge_ratio: float = 0.2
    enable_transitive_pruning: bool = True
    transitive_max_hops: int = 4
    transitive_ratio: float = 1.05  # Slightly relaxed from 1.03
    transitive_max_checks: int = 25000
    
    # Smoothing
    use_turn_preserving_smoothing: bool = True
    turn_smoothing_deg: float = 30.0
    turn_smoothing_neighbor_weight: float = 0.25
    smooth_iterations: int = 2
    
    # Centerline filtering
    min_centerline_length_m: float = 6.0  # Reduced from 10.0 for better recovery
    
    # Candidate selection (relaxed for better recovery)
    apply_candidate_selection: bool = True
    candidate_selection_threshold: float = 0.25  # Reduced from 0.40 for better recovery
    candidate_length_scale_m: float = 50.0  # Reduced from 60.0
    candidate_density_scale: float = 0.15  # Reduced from 0.20
    candidate_force_keep_weighted_support: float = 8.0  # Reduced from 15.0
    candidate_short_length_m: float = 5.0  # Reduced from 8.0
    candidate_low_weighted_support: float = 2.0  # Reduced from 4.0
    candidate_dangling_max_length_m: float = 45.0  # Increased from 25.0 for better recovery
    candidate_dangling_min_weighted_support: float = 3.0  # Reduced from 6.0
    
    # Dynamic weighting
    enable_dynamic_weighting: bool = True
    dyn_lambda_vpd: float = 1.6
    dyn_road_likeness_beta: float = 6.0
    dyn_road_likeness_tau: float = 0.45
    
    # Roundabout detection (relaxed for better detection)
    enable_roundabout_detection: bool = True
    roundabout_min_radius_m: float = 6.0  # Reduced from 8.0
    roundabout_max_radius_m: float = 60.0  # Increased from 50.0
    roundabout_min_traces: int = 2  # Reduced from 3
    roundabout_cluster_radius_m: float = 30.0  # Increased from 25.0
    
    # Roundabout post-filter (removes false positives after detection)
    enable_roundabout_postfilter: bool = True
    roundabout_postfilter_min_traces: int = 5  # Min traces near circle (tightened)
    roundabout_postfilter_min_points: int = 25  # Min points on circle (tightened)
    roundabout_postfilter_min_directions: int = 3  # Min entry directions (tightened)
    roundabout_postfilter_min_arc_coverage: float = 0.40  # Min circle coverage (tightened)
    roundabout_postfilter_merge_distance_m: float = 30.0  # Merge nearby duplicates
    
    # Curve smoothing (DISABLED - was causing fragmentation)
    # Post-processing merge is used instead
    enable_curve_smoothing: bool = False
    curve_simplify_tolerance: float = 2.5
    curve_merge_distance: float = 4.0
    curve_smooth_weight: float = 0.22
    
    # Post-processing: merge parallel overlapping centerlines
    enable_parallel_merge: bool = True
    parallel_merge_buffer_m: float = 10.0      # Lateral corridor buffer (meters)
    parallel_merge_heading_tol_deg: float = 25.0
    parallel_merge_min_overlap: float = 0.65   # Min fraction of shorter inside longer's buffer
    
    # Intersection-based topology (NEW: replaces simple parallel merge)
    enable_intersection_topology: bool = True
    intersection_snap_radius_m: float = 8.0  # Radius for clustering endpoints
    intersection_min_degree: int = 1  # Minimum node degree to keep
    
    # Fréchet averaging (NEW: replaces winner-takes-all deduplication)
    enable_frechet_averaging: bool = True
    frechet_corridor_buffer_m: float = 12.0  # Corridor for grouping parallels
    frechet_heading_tolerance_deg: float = 30.0  # Max heading difference
    frechet_min_overlap: float = 0.60  # Min corridor overlap for grouping
    frechet_resample_spacing_m: float = 5.0  # Target spacing for resampling
    frechet_eccentricity_power: float = 1.0  # Outlier penalization strength


class KharitaCenterlineGenerator:
    """
    Phase 2: Generate VPD centerlines using Kharita-style clustering.
    
    Replaces KDE skeletonization with a graph-based approach:
      - Cluster trace points by position AND heading
      - Build co-occurrence graph from trace transitions
      - Prune weak/redundant edges
      - Stitch remaining edges into centerlines
      - Smooth while preserving turns
    """
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        config: KharitaConfig = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.config = config or KharitaConfig()
        
        self.gdf = None
        self.crs_projected = None
        self.to_proj = None
        self.to_wgs = None
        
        # Algorithm state
        self.labels = None
        self.nodes_df = None
        self.edges_df = None
        self.centerlines_df = None
        self.roundabouts = []  # Detected roundabout geometries
    
    # ------------------------------------------------------------------
    #  DATA LOADING
    # ------------------------------------------------------------------
    
    def load_and_project(self):
        """Load VPD traces and set up coordinate transformations."""
        print(f"  Loading VPD from {self.input_path}...")
        self.gdf = gpd.read_file(self.input_path)
        
        # Dynamic local projection (azimuthal equidistant)
        bounds = self.gdf.total_bounds
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        
        self.crs_projected = CRS.from_proj4(
            f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
            f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        
        # Create transformers
        wgs84 = CRS.from_epsg(4326)
        self.to_proj = Transformer.from_crs(wgs84, self.crs_projected, always_xy=True)
        self.to_wgs = Transformer.from_crs(self.crs_projected, wgs84, always_xy=True)
        
        # Filter valid LineStrings
        self.gdf = self.gdf[
            self.gdf.geometry.notna()
            & ~self.gdf.geometry.is_empty
            & (self.gdf.geom_type == "LineString")
        ].reset_index(drop=True)
        
        print(f"  Loaded {len(self.gdf)} VPD traces.")
    
    # ------------------------------------------------------------------
    #  TRACE SAMPLING
    # ------------------------------------------------------------------
    
    def _sample_traces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[int, int]], Dict[int, dict]]:
        """
        Sample all traces at regular intervals.
        
        Returns:
            x_arr: X coordinates of all sample points
            y_arr: Y coordinates of all sample points
            heading_arr: Headings at all sample points
            weight_arr: Weights for all sample points
            altitude_arr: Altitude values for all sample points (NaN if unavailable)
            trace_ranges: Dict mapping trace_id -> (start_idx, end_idx)
            trace_meta: Dict mapping trace_id -> metadata
        """
        print("  Sampling traces...")
        t0 = time.time()
        config = self.config
        
        xs: List[float] = []
        ys: List[float] = []
        headings_list: List[float] = []
        point_weights: List[float] = []
        altitudes: List[float] = []  # Z-level data for interchange handling
        
        trace_ranges: Dict[int, Tuple[int, int]] = {}
        trace_meta: Dict[int, dict] = {}
        next_trace_id = 0
        
        for idx, row in self.gdf.iterrows():
            geom_wgs = row.geometry
            
            # Project geometry
            coords_wgs = np.array(geom_wgs.coords)
            coords_xy = np.column_stack(self.to_proj.transform(coords_wgs[:, 0], coords_wgs[:, 1]))
            line_xy = LineString(coords_xy)
            
            if line_xy.is_empty or line_xy.length <= 1.0:
                continue
            
            # Get altitude data if available (for Z-level separation at interchanges)
            trace_altitude = np.nan
            if "mean_altitude" in row.index and pd.notna(row.get("mean_altitude")):
                trace_altitude = float(row["mean_altitude"])
            
            # Sample the projected line
            sample_dists, sample_xy, sample_headings = sample_line_projected(
                line_xy=line_xy,
                sample_spacing_m=config.sample_spacing_m,
                max_points=config.max_points_per_trace,
            )
            
            if len(sample_xy) < 2:
                continue
            
            # Compute trace weight
            n_pts = len(coords_wgs)
            length_m = line_xy.length
            
            # Heading consistency from raw trace
            if n_pts >= 3:
                dx = coords_xy[1:, 0] - coords_xy[:-1, 0]
                dy = coords_xy[1:, 1] - coords_xy[:-1, 1]
                seg_headings = np.arctan2(dy, dx)
                sin_sum = np.sum(np.sin(seg_headings))
                cos_sum = np.sum(np.cos(seg_headings))
                r = np.sqrt(sin_sum**2 + cos_sum**2) / len(seg_headings)
                heading_consistency = float(np.clip(r, 0.0, 1.0))
            else:
                heading_consistency = 0.5
            
            base_weight = compute_trace_weight(
                source="VPD",
                length_m=length_m,
                n_points=n_pts,
                heading_consistency=heading_consistency,
                path_quality=row.get("path_quality_score", 0.5) if pd.notnull(row.get("path_quality_score", np.nan)) else 0.5,
                sensor_quality=row.get("sensor_quality_score", 0.5) if pd.notnull(row.get("sensor_quality_score", np.nan)) else 0.5,
                construction_percent=row.get("construction_percent", 0.0) if pd.notnull(row.get("construction_percent", np.nan)) else 0.0,
                vpd_base_weight=config.vpd_base_weight,
            )
            
            # Store sample points
            start = len(xs)
            for i in range(len(sample_xy)):
                xs.append(float(sample_xy[i, 0]))
                ys.append(float(sample_xy[i, 1]))
                headings_list.append(float(sample_headings[i]))
                point_weights.append(base_weight)
                altitudes.append(trace_altitude)  # Same altitude for all points in trace
            end = len(xs)
            
            trace_ranges[next_trace_id] = (start, end)
            trace_meta[next_trace_id] = {
                "trace_id": row.get("trace_id", idx),
                "source": "VPD",
                "construction_percent": float(row.get("construction_percent", 0.0) or 0.0),
                "traffic_signal_count": float(row.get("traffic_signal_count", 0.0) or 0.0),
                "day": int(row["day"]) if pd.notnull(row.get("day", np.nan)) else None,
                "hour": int(row["hour"]) if pd.notnull(row.get("hour", np.nan)) else None,
                "mean_altitude": trace_altitude,
            }
            next_trace_id += 1
        
        elapsed = time.time() - t0
        print(f"    Sampled {len(xs)} points from {next_trace_id} traces in {elapsed:.1f}s")
        
        # Log altitude coverage
        alt_arr = np.asarray(altitudes, dtype=np.float32)
        n_with_alt = np.sum(~np.isnan(alt_arr))
        if n_with_alt > 0:
            print(f"    Z-level data: {n_with_alt}/{len(alt_arr)} points have altitude")
        
        x_arr = np.asarray(xs, dtype=np.float32)
        y_arr = np.asarray(ys, dtype=np.float32)
        heading_arr = np.asarray(headings_list, dtype=np.float32)
        weight_arr = np.asarray(point_weights, dtype=np.float32)
        
        return x_arr, y_arr, heading_arr, weight_arr, alt_arr, trace_ranges, trace_meta
    
    # ------------------------------------------------------------------
    #  HEADING-AWARE CLUSTERING (with Z-level separation for interchanges)
    # ------------------------------------------------------------------
    
    def _kharita_clustering(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        headings: np.ndarray,
        point_weights: np.ndarray,
        altitudes: np.ndarray = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Heading-aware incremental clustering (Kharita-style).
        
        Points are assigned to clusters based on:
          - Spatial distance (within cluster_radius_m)
          - Heading similarity (within heading_tolerance_deg)
          - Z-level similarity (for bridge/overpass separation at interchanges)
        
        Cluster centroids and headings are updated incrementally.
        """
        print("  Performing heading-aware clustering (with Z-level separation)...")
        t0 = time.time()
        config = self.config
        
        n = len(xs)
        if n == 0:
            return np.array([], dtype=np.int32), pd.DataFrame(
                columns=["node_id", "x", "y", "heading", "weight", "point_count", "mean_altitude"]
            )
        
        labels = np.full(n, -1, dtype=np.int32)
        
        # Z-level handling for interchange separation
        use_z_level = config.enable_z_level_separation and altitudes is not None
        if use_z_level:
            n_valid_alt = np.sum(~np.isnan(altitudes))
            if n_valid_alt < n * 0.1:  # Less than 10% have altitude
                use_z_level = False
                print("    Note: Z-level separation disabled (insufficient altitude data)")
        
        # Cluster state
        cx: List[float] = []  # centroid x
        cy: List[float] = []  # centroid y
        cweight: List[float] = []  # total weight
        csin: List[float] = []  # weighted sin(heading)
        ccos: List[float] = []  # weighted cos(heading)
        ccount: List[int] = []  # point count
        calt: List[float] = []  # mean altitude (for Z-level separation)
        calt_count: List[int] = []  # count of points with valid altitude
        
        tree = None
        dirty_count = 0
        
        def rebuild_tree():
            nonlocal tree
            if cx:
                arr = np.column_stack([
                    np.asarray(cx, dtype=np.float32),
                    np.asarray(cy, dtype=np.float32)
                ])
                tree = cKDTree(arr)
        
        for i in range(n):
            x = float(xs[i])
            y = float(ys[i])
            heading = float(headings[i])
            w = float(max(point_weights[i], 1e-6))
            point_alt = float(altitudes[i]) if use_z_level else np.nan
            
            # First point creates first cluster
            if not cx:
                cx.append(x)
                cy.append(y)
                cweight.append(w)
                rad = math.radians(heading)
                csin.append(math.sin(rad) * w)
                ccos.append(math.cos(rad) * w)
                ccount.append(1)
                if use_z_level and not math.isnan(point_alt):
                    calt.append(point_alt)
                    calt_count.append(1)
                else:
                    calt.append(np.nan)
                    calt_count.append(0)
                labels[i] = 0
                rebuild_tree()
                continue
            
            # Rebuild tree periodically for efficiency
            if tree is None or dirty_count >= 2000 or i % 20000 == 0:
                rebuild_tree()
                dirty_count = 0
            
            # Find candidate clusters within radius
            candidate_ids = (
                tree.query_ball_point([x, y], r=config.cluster_radius_m)
                if tree is not None
                else []
            )
            
            # Find best matching cluster (closest in combined spatial+heading+z-level distance)
            best_cid = -1
            best_score = float("inf")
            
            for cid in candidate_ids:
                # Compute cluster heading from sin/cos sums
                cheading = (math.degrees(math.atan2(csin[cid], ccos[cid])) + 360.0) % 360.0
                
                # Check heading tolerance
                ad = angle_diff_deg(heading, cheading)
                if ad > config.heading_tolerance_deg:
                    continue
                
                # Z-level penalty for interchange separation (don't merge bridge with underpass)
                z_penalty = 0.0
                if use_z_level and not math.isnan(point_alt) and calt_count[cid] > 0:
                    cluster_alt = calt[cid]
                    if not math.isnan(cluster_alt):
                        alt_diff = abs(point_alt - cluster_alt)
                        if alt_diff > config.z_separation_threshold_m:
                            # Large Z difference - likely different road levels
                            z_penalty = config.z_level_cluster_penalty
                
                # Combined distance (spatial + heading penalty + z-level penalty)
                sd = math.hypot(x - cx[cid], y - cy[cid])
                score = sd + config.heading_distance_weight_m * ad + z_penalty
                
                if score < best_score:
                    best_score = score
                    best_cid = cid
            
            # If no matching cluster, create new one
            if best_cid == -1:
                cid = len(cx)
                cx.append(x)
                cy.append(y)
                cweight.append(w)
                rad = math.radians(heading)
                csin.append(math.sin(rad) * w)
                ccos.append(math.cos(rad) * w)
                ccount.append(1)
                if use_z_level and not math.isnan(point_alt):
                    calt.append(point_alt)
                    calt_count.append(1)
                else:
                    calt.append(np.nan)
                    calt_count.append(0)
                labels[i] = cid
                dirty_count += 1
                continue
            
            # Update existing cluster with weighted incremental average
            cid = best_cid
            old_w = cweight[cid]
            new_w = old_w + w
            cx[cid] = (cx[cid] * old_w + x * w) / new_w
            cy[cid] = (cy[cid] * old_w + y * w) / new_w
            cweight[cid] = new_w
            rad = math.radians(heading)
            csin[cid] += math.sin(rad) * w
            ccos[cid] += math.cos(rad) * w
            ccount[cid] += 1
            # Update altitude tracking
            if use_z_level and not math.isnan(point_alt):
                if calt_count[cid] == 0:
                    calt[cid] = point_alt
                else:
                    # Weighted average of altitude
                    old_alt = calt[cid] if not math.isnan(calt[cid]) else point_alt
                    calt[cid] = (old_alt * calt_count[cid] + point_alt) / (calt_count[cid] + 1)
                calt_count[cid] += 1
            labels[i] = cid
            dirty_count += 1
        
        # Collapse small clusters into nearby larger ones (consider Z-level)
        if config.min_cluster_points > 1 and cx:
            major_ids = [i for i, cnt in enumerate(ccount) if cnt >= config.min_cluster_points]
            if major_ids:
                major_xy = np.column_stack([
                    np.asarray([cx[i] for i in major_ids]),
                    np.asarray([cy[i] for i in major_ids]),
                ])
                major_tree = cKDTree(major_xy)
                remap = {cid: cid for cid in range(len(cx))}
                
                for cid, cnt in enumerate(ccount):
                    if cnt >= config.min_cluster_points:
                        continue
                    d, idx = major_tree.query([cx[cid], cy[cid]], k=1)
                    target = major_ids[int(idx)]
                    
                    # Check Z-level compatibility before merging
                    z_compatible = True
                    if use_z_level and calt_count[cid] > 0 and calt_count[target] > 0:
                        if not math.isnan(calt[cid]) and not math.isnan(calt[target]):
                            if abs(calt[cid] - calt[target]) > config.z_separation_threshold_m:
                                z_compatible = False
                    
                    if float(d) <= config.cluster_radius_m * 1.25 and z_compatible:
                        remap[cid] = target
                
                labels = np.asarray([remap[int(cid)] for cid in labels], dtype=np.int32)
        
        # Reindex labels to compact IDs
        unique = sorted(set(int(v) for v in labels))
        to_new = {old: new for new, old in enumerate(unique)}
        labels = np.asarray([to_new[int(v)] for v in labels], dtype=np.int32)
        
        # Recompute node stats from final assignments (including altitude)
        node_stats = defaultdict(
            lambda: {"xw": 0.0, "yw": 0.0, "w": 0.0, "sinw": 0.0, "cosw": 0.0, "n": 0, 
                     "alt_sum": 0.0, "alt_n": 0}
        )
        
        for i, nid in enumerate(labels):
            w = float(max(point_weights[i], 1e-6))
            node_stats[int(nid)]["xw"] += float(xs[i]) * w
            node_stats[int(nid)]["yw"] += float(ys[i]) * w
            node_stats[int(nid)]["w"] += w
            rad = math.radians(float(headings[i]))
            node_stats[int(nid)]["sinw"] += math.sin(rad) * w
            node_stats[int(nid)]["cosw"] += math.cos(rad) * w
            node_stats[int(nid)]["n"] += 1
            # Track altitude
            if use_z_level and not math.isnan(altitudes[i]):
                node_stats[int(nid)]["alt_sum"] += float(altitudes[i])
                node_stats[int(nid)]["alt_n"] += 1
        
        # Build nodes dataframe (with altitude for Z-level handling)
        rows = []
        for nid in sorted(node_stats):
            s = node_stats[nid]
            ww = s["w"] if s["w"] > 0 else 1.0
            mean_alt = s["alt_sum"] / s["alt_n"] if s["alt_n"] > 0 else np.nan
            rows.append({
                "node_id": nid,
                "x": s["xw"] / ww,
                "y": s["yw"] / ww,
                "heading": (math.degrees(math.atan2(s["sinw"], s["cosw"])) + 360.0) % 360.0,
                "weight": s["w"],
                "point_count": s["n"],
                "mean_altitude": mean_alt,
            })
        
        elapsed = time.time() - t0
        print(f"    Created {len(rows)} clusters in {elapsed:.1f}s")
        
        return labels, pd.DataFrame(rows)
    
    # ------------------------------------------------------------------
    #  CO-OCCURRENCE GRAPH CONSTRUCTION
    # ------------------------------------------------------------------
    
    def _build_edge_graph(
        self,
        labels: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        headings: np.ndarray,
        weights: np.ndarray,
        trace_ranges: Dict[int, Tuple[int, int]],
        trace_meta: Dict[int, dict],
    ) -> Dict[Tuple[int, int], dict]:
        """
        Build directed co-occurrence graph from trace transitions.
        
        An edge (u, v) is created when consecutive points in a trace
        are assigned to clusters u and v (with u != v).
        """
        print("  Building co-occurrence graph...")
        t0 = time.time()
        config = self.config
        
        edge_support: Dict[Tuple[int, int], dict] = {}
        
        for tid, (start, end) in trace_ranges.items():
            meta = trace_meta[tid]
            if end - start < 2:
                continue
            
            for i in range(start + 1, end):
                u = int(labels[i - 1])
                v = int(labels[i])
                
                if u == v:
                    continue
                
                # Check transition distance
                step_dist = float(np.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]))
                if step_dist <= 0.0 or step_dist > config.max_transition_distance_m:
                    continue
                
                key = (u, v)
                if key not in edge_support:
                    edge_support[key] = {
                        "support": 0.0,
                        "weighted_support": 0.0,
                        "length_sum": 0.0,
                        "vpd_support": 0.0,
                        "hpd_support": 0.0,
                        "construction_sum": 0.0,
                        "traffic_signal_sum": 0.0,
                        "day_counter": Counter(),
                        "hour_counter": Counter(),
                        "heading_sin_sum": 0.0,
                        "heading_cos_sum": 0.0,
                        "heading_count": 0.0,
                    }
                
                s = edge_support[key]
                s["support"] += 1.0
                w = float(weights[i])
                s["weighted_support"] += w
                s["length_sum"] += step_dist
                
                if str(meta["source"]).upper() == "VPD":
                    s["vpd_support"] += 1.0
                else:
                    s["hpd_support"] += 1.0
                
                s["construction_sum"] += float(meta["construction_percent"])
                s["traffic_signal_sum"] += float(meta["traffic_signal_count"])
                
                if meta.get("day") is not None:
                    s["day_counter"][int(meta["day"])] += 1
                if meta.get("hour") is not None:
                    s["hour_counter"][int(meta["hour"])] += 1
                
                h_step = float(headings[i])
                s["heading_sin_sum"] += math.sin(math.radians(h_step))
                s["heading_cos_sum"] += math.cos(math.radians(h_step))
                s["heading_count"] += 1.0
        
        elapsed = time.time() - t0
        print(f"    Created {len(edge_support)} edges in {elapsed:.1f}s")
        
        return edge_support
    
    # ------------------------------------------------------------------
    #  EDGE PRUNING
    # ------------------------------------------------------------------
    
    def _prune_edges(
        self,
        edge_support: Dict[Tuple[int, int], dict],
        node_xy: Dict[int, Tuple[float, float]],
    ) -> Dict[Tuple[int, int], dict]:
        """
        Three-pass edge pruning:
          1. Remove edges below minimum support
          2. Remove weak direction-conflict edges
          3. Transitive pruning (remove edges with short alternatives)
        """
        print("  Pruning edges (3 passes)...")
        t0 = time.time()
        config = self.config
        
        initial_count = len(edge_support)
        
        # Pass 1: Minimum support
        edge_support = {
            k: v for k, v in edge_support.items()
            if float(v.get("support", 0.0)) >= config.min_edge_support
        }
        after_pass1 = len(edge_support)
        print(f"    Pass 1 (min support): {initial_count} -> {after_pass1}")
        
        # Pass 2: Direction conflict
        to_drop: Set[Tuple[int, int]] = set()
        for (u, v), sv in list(edge_support.items()):
            if (v, u) not in edge_support:
                continue
            sr = edge_support[(v, u)]
            if float(sv["support"]) < float(sr["support"]) * config.reverse_edge_ratio:
                to_drop.add((u, v))
        
        for k in to_drop:
            edge_support.pop(k, None)
        
        after_pass2 = len(edge_support)
        print(f"    Pass 2 (direction conflict): {after_pass1} -> {after_pass2}")
        
        # Pass 3: Transitive pruning
        if config.enable_transitive_pruning and config.transitive_max_checks > 0:
            edge_lengths = {
                (u, v): float(np.hypot(
                    node_xy[u][0] - node_xy[v][0],
                    node_xy[u][1] - node_xy[v][1],
                ))
                for (u, v) in edge_support
            }
            
            # Build graph for shortest path queries
            graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            for (u, v), dist in edge_lengths.items():
                graph[u].append((v, dist))
            
            # Sort edges by support (ascending) for pruning candidates
            candidates = sorted(
                edge_support.keys(),
                key=lambda e: (float(edge_support[e]["support"]), -edge_lengths[e]),
            )[: config.transitive_max_checks]
            
            dropped_transitive: Set[Tuple[int, int]] = set()
            for e in candidates:
                if e not in edge_support:
                    continue
                direct = edge_lengths[e]
                if direct <= 1.0:
                    continue
                
                alt = shortest_alternative_with_hop_limit(
                    graph=graph,
                    source=e[0],
                    target=e[1],
                    skip_edge=e,
                    max_hops=config.transitive_max_hops,
                    max_dist=direct * config.transitive_ratio,
                )
                
                if np.isfinite(alt) and alt <= direct * config.transitive_ratio:
                    dropped_transitive.add(e)
            
            for e in dropped_transitive:
                edge_support.pop(e, None)
            
            after_pass3 = len(edge_support)
            print(f"    Pass 3 (transitive): {after_pass2} -> {after_pass3}")
        
        elapsed = time.time() - t0
        print(f"    Pruning complete in {elapsed:.1f}s")
        
        return edge_support
    
    # ------------------------------------------------------------------
    #  CENTERLINE STITCHING + CANDIDATE SELECTION (with curve interpolation)
    # ------------------------------------------------------------------
    
    def _precompute_edge_trace_points(self):
        """
        Pre-compute lookup table mapping edges to their trace points.
        
        Called once after clustering to enable O(1) edge point lookups
        during centerline stitching (vs O(n) per edge before).
        """
        if self._trace_labels is None or self._trace_points_xy is None:
            self._edge_trace_points = {}
            return
        
        print("    Pre-computing edge trace points lookup...")
        t0 = time.time()
        
        labels = self._trace_labels
        points = self._trace_points_xy
        edge_points = {}  # (u, v) -> list of (x, y)
        
        for i in range(1, len(labels)):
            u, v = int(labels[i-1]), int(labels[i])
            if u == v:
                continue
            
            # Canonical edge key (smaller node first for bidirectional)
            key = (min(u, v), max(u, v))
            
            if key not in edge_points:
                edge_points[key] = []
            
            # Add both endpoints of the transition
            edge_points[key].append((float(points[i-1, 0]), float(points[i-1, 1])))
            edge_points[key].append((float(points[i, 0]), float(points[i, 1])))
        
        self._edge_trace_points = edge_points
        print(f"    Pre-computed {len(edge_points)} edge lookups in {time.time() - t0:.1f}s")
    
    def _get_edge_trace_points(self, u: int, v: int) -> List[Tuple[float, float]]:
        """
        Get trace points that contributed to an edge (for curve interpolation).
        
        Uses precomputed lookup for O(1) access.
        """
        if not hasattr(self, '_edge_trace_points') or self._edge_trace_points is None:
            return []
        
        # Canonical key (smaller node first)
        key = (min(u, v), max(u, v))
        return self._edge_trace_points.get(key, [])
    
    def _stitch_centerlines(
        self,
        edge_support: Dict[Tuple[int, int], dict],
        node_xy: Dict[int, Tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Stitch edges into centerline paths with smoothing.
        
        Includes curve-aware interpolation to fix "chord cutting" on cloverleaf ramps.
        """
        print("  Stitching centerlines (with curve interpolation)...")
        t0 = time.time()
        config = self.config
        
        # Get centerline paths
        centerline_paths = stitch_centerline_paths(edge_support)
        print(f"    Found {len(centerline_paths)} centerline paths")
        
        centerline_rows = []
        curves_interpolated = 0
        
        for path_nodes in centerline_paths:
            if len(path_nodes) < 2:
                continue
            
            # Build path coordinates with curve-aware interpolation
            if config.enable_curve_interpolation:
                path_coords = [node_xy[path_nodes[0]]]
                
                for i in range(1, len(path_nodes)):
                    u, v = path_nodes[i-1], path_nodes[i]
                    start_xy = node_xy[u]
                    end_xy = node_xy[v]
                    edge_length = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
                    
                    # Only interpolate longer edges where curves matter
                    if edge_length >= config.curve_min_edge_length_m:
                        # Get trace points for this edge
                        edge_trace_pts = self._get_edge_trace_points(u, v)
                        
                        if len(edge_trace_pts) >= 3:
                            # Use curve interpolation
                            interpolated = interpolate_edge_with_traces(
                                start_xy=start_xy,
                                end_xy=end_xy,
                                trace_points=edge_trace_pts,
                                edge_length_m=edge_length,
                                min_intermediate_points=2,
                                max_deviation_m=config.curve_max_deviation_m,
                            )
                            # Add intermediate points (skip start as it's already in path)
                            for pt in interpolated[1:]:
                                path_coords.append(pt)
                            if len(interpolated) > 2:
                                curves_interpolated += 1
                        else:
                            path_coords.append(end_xy)
                    else:
                        path_coords.append(end_xy)
                
                raw_xy = np.asarray(path_coords, dtype=np.float64)
            else:
                # Original: just use node centroids
                raw_xy = np.asarray([node_xy[n] for n in path_nodes], dtype=np.float64)
            
            # Apply turn-preserving smoothing
            if config.use_turn_preserving_smoothing:
                smooth_xy = smooth_polyline_preserve_turns(
                    raw_xy,
                    passes=config.smooth_iterations,
                    turn_deg=config.turn_smoothing_deg,
                    neighbor_weight=config.turn_smoothing_neighbor_weight,
                )
            else:
                smooth_xy = raw_xy
            
            # Apply curve smoothing to fix overlapping clusters on curves
            if config.enable_curve_smoothing and len(smooth_xy) >= 3:
                smooth_xy = simplify_and_smooth_centerline(
                    smooth_xy,
                    simplify_tolerance=config.curve_simplify_tolerance,
                    merge_distance=config.curve_merge_distance,
                    smooth_iterations=1,
                    smooth_weight=config.curve_smooth_weight,
                )
            
            # Create LineString
            line_xy = LineString([(float(x), float(y)) for x, y in smooth_xy])
            if line_xy.length < config.min_centerline_length_m:
                continue
            
            # Convert to WGS84
            coords_xy = np.array(line_xy.coords)
            lon, lat = self.to_wgs.transform(coords_xy[:, 0], coords_xy[:, 1])
            line_wgs = LineString(zip(lon, lat))
            
            # Aggregate edge statistics
            support_sum = 0.0
            weighted_support_sum = 0.0
            construction_sum = 0.0
            signal_sum = 0.0
            
            for i in range(1, len(path_nodes)):
                a, b = path_nodes[i-1], path_nodes[i]
                for key in [(a, b), (b, a)]:
                    if key in edge_support:
                        es = edge_support[key]
                        support_sum += es["support"]
                        weighted_support_sum += es["weighted_support"]
                        construction_sum += es["construction_sum"]
                        signal_sum += es["traffic_signal_sum"]
            
            if support_sum <= 0:
                continue
            
            endpoint_dist_m = float(np.hypot(
                smooth_xy[-1, 0] - smooth_xy[0, 0],
                smooth_xy[-1, 1] - smooth_xy[0, 1],
            ))
            
            centerline_rows.append({
                "node_path": path_nodes,
                "support": float(support_sum),
                "weighted_support": float(weighted_support_sum),
                "construction_percent_mean": float(construction_sum / support_sum),
                "traffic_signal_count_mean": float(signal_sum / support_sum),
                "u": int(path_nodes[0]),
                "v": int(path_nodes[-1]),
                "length_m": float(line_xy.length),
                "endpoint_dist_m": endpoint_dist_m,
                "geometry": line_wgs,
                "source": "VPD",
            })
        
        elapsed = time.time() - t0
        print(f"    Created {len(centerline_rows)} centerlines in {elapsed:.1f}s")
        if curves_interpolated > 0:
            print(f"    Curve interpolation: {curves_interpolated} edges enhanced for cloverleaf/ramp handling")
        
        return pd.DataFrame(centerline_rows)
    
    def _candidate_selection(self, centerlines: pd.DataFrame) -> pd.DataFrame:
        """
        Apply heuristic candidate selection to filter low-quality centerlines.
        """
        config = self.config
        
        if centerlines.empty or not config.apply_candidate_selection:
            return centerlines
        
        print("  Applying candidate selection...")
        
        out = centerlines.copy()
        
        # Compute node degrees
        deg = Counter()
        for _, row in out.iterrows():
            deg[row["u"]] += 1
            deg[row["v"]] += 1
        
        u_deg = out["u"].map(lambda x: float(deg.get(x, 0)))
        v_deg = out["v"].map(lambda x: float(deg.get(x, 0)))
        dangling = ((u_deg <= 1.0) | (v_deg <= 1.0)).astype(np.float64)
        
        length_m = out["length_m"].values
        weighted_support = out["weighted_support"].values
        endpoint_dist_m = out["endpoint_dist_m"].values
        
        # Compute selection score
        safe_len = np.maximum(length_m, 1e-6)
        sinuosity = length_m / np.maximum(endpoint_dist_m, 1e-6)
        support_density = weighted_support / safe_len
        
        support_n = np.clip(weighted_support / max(config.candidate_force_keep_weighted_support, 1e-6), 0.0, 1.0)
        density_n = np.clip(support_density / max(config.candidate_density_scale, 1e-6), 0.0, 1.0)
        length_n = np.clip(length_m / max(config.candidate_length_scale_m, 1e-6), 0.0, 1.0)
        connectivity_n = np.clip((u_deg + v_deg) / 8.0, 0.0, 1.0)
        # Minimal penalty for dangling ends to improve recovery in sparse areas
        connectivity_n = np.where(dangling > 0.0, connectivity_n * 0.70, np.maximum(connectivity_n, 0.55))
        
        # Weights favor support and length for better recovery
        score = (
            0.30 * support_n
            + 0.15 * density_n  # Reduced importance of density
            + 0.30 * length_n   # High importance of length for coverage
            + 0.25 * connectivity_n  # Reduced connectivity weight
        )
        
        # Apply selection rules
        selected = np.ones(len(out), dtype=bool)
        reasons = []
        
        for i in range(len(out)):
            reason = "score_threshold"
            if weighted_support[i] >= config.candidate_force_keep_weighted_support:
                reason = "force_keep_support"
                selected[i] = True
            elif (length_m[i] < config.candidate_short_length_m and 
                  weighted_support[i] < config.candidate_low_weighted_support):
                reason = "drop_short_low_support"
                selected[i] = False
            elif (dangling.iloc[i] > 0.0 and 
                  length_m[i] < config.candidate_dangling_max_length_m and
                  weighted_support[i] < config.candidate_dangling_min_weighted_support):
                reason = "drop_dangling_weak"
                selected[i] = False
            else:
                selected[i] = bool(score[i] >= config.candidate_selection_threshold)
                if not selected[i]:
                    reason = "drop_low_score"
            reasons.append(reason)
        
        out["selection_score"] = score
        out["selection_reason"] = reasons
        out["is_selected"] = selected
        
        filtered = out[out["is_selected"]].copy()
        print(f"    Selected {len(filtered)} of {len(out)} centerlines")
        
        return filtered.reset_index(drop=True)
    
    # ------------------------------------------------------------------
    #  MERGE PARALLEL OVERLAPPING CENTERLINES
    # ------------------------------------------------------------------
    
    def _merge_parallel_centerlines(self, centerlines: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate/sub-segment centerlines using corridor-based overlap.
        
        For each segment, check if it's contained within a longer, higher-quality
        segment's corridor. If so, remove it. NO Union-Find to avoid transitive
        chains that incorrectly merge segments from different roads.
        
        Two types of duplicates:
        1. Sub-segments: shorter line is >70% inside longer line's buffer
        2. True parallels: similar-length lines are MUTUALLY >70% inside each other
        """
        config = self.config
        
        if not config.enable_parallel_merge or centerlines.empty:
            return centerlines
        
        print("  Removing duplicate/sub-segment centerlines (corridor overlap)...")
        
        gdf = gpd.GeoDataFrame(centerlines, crs="EPSG:4326")
        gdf = gdf.to_crs(self.crs_projected)
        n = len(gdf)
        
        def get_heading(geom):
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                return 0.0
            coords = np.array(geom.coords)
            dx = coords[-1, 0] - coords[0, 0]
            dy = coords[-1, 1] - coords[0, 1]
            return np.degrees(np.arctan2(dy, dx)) % 180.0
        
        headings = np.array([get_heading(g) for g in gdf.geometry])
        lengths = gdf.geometry.length.values
        supports = gdf["weighted_support"].values if "weighted_support" in gdf.columns else np.ones(n)
        scores = supports * lengths
        
        from shapely import STRtree
        geoms = gdf.geometry.values
        
        buf_dist = config.parallel_merge_buffer_m
        buffered_geoms = [g.buffer(buf_dist) for g in geoms]
        
        # Spatial pre-filter
        tree = STRtree(geoms)
        query_distance = buf_dist + 5.0
        left_idx, right_idx = tree.query(geoms, predicate="dwithin", distance=query_distance)
        
        # Build adjacency: for each segment, list of nearby segments
        neighbors = defaultdict(list)
        for k in range(len(left_idx)):
            i, j = int(left_idx[k]), int(right_idx[k])
            if i != j:
                neighbors[i].append(j)
                neighbors[j].append(i)
        
        to_remove = set()
        overlap_count = 0
        
        # For each segment, check if it should be removed
        for i in range(n):
            if i in to_remove:
                continue
            
            len_i = lengths[i]
            if len_i < 1.0:
                continue
            
            for j in neighbors[i]:
                if j in to_remove:
                    continue
                
                len_j = lengths[j]
                if len_j < 1.0:
                    continue
                
                # Heading compatibility
                h_diff = abs(headings[i] - headings[j])
                h_diff = min(h_diff, 180.0 - h_diff)
                if h_diff > config.parallel_merge_heading_tol_deg:
                    continue
                
                # Determine which is shorter/longer
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
                
                # Check corridor overlap: is shorter inside longer's buffer?
                try:
                    inside = shorter_geom.intersection(longer_buf)
                    corridor_ratio = inside.length / shorter_len
                except Exception:
                    continue
                
                if corridor_ratio < config.parallel_merge_min_overlap:
                    continue
                
                # Length ratio check: for very different lengths, require higher overlap
                len_ratio = shorter_len / longer_len
                if len_ratio < 0.3 and corridor_ratio < 0.85:
                    # Short stub that only partially overlaps - might be a side street
                    continue
                
                # If similar lengths, require MUTUAL overlap (true parallel duplicate)
                if len_ratio > 0.7:
                    try:
                        longer_geom = geoms[longer_idx]
                        shorter_buf = buffered_geoms[shorter_idx]
                        reverse_inside = longer_geom.intersection(shorter_buf)
                        reverse_ratio = reverse_inside.length / longer_len
                    except Exception:
                        continue
                    
                    if reverse_ratio < config.parallel_merge_min_overlap * 0.8:
                        # Not mutually overlapping - different roads that just touch
                        continue
                
                # This is a duplicate - remove the one with lower score
                if scores[shorter_idx] < scores[longer_idx]:
                    to_remove.add(shorter_idx)
                else:
                    to_remove.add(longer_idx)
                overlap_count += 1
        
        keep_mask = [i not in to_remove for i in range(n)]
        gdf = gdf[keep_mask].reset_index(drop=True)
        gdf = gdf.to_crs("EPSG:4326")
        
        removed = len(to_remove)
        print(f"    Found {overlap_count} duplicate pairs, removed {removed} centerlines")
        print(f"    Merged {removed} duplicate/sub-segment centerlines")
        
        return pd.DataFrame(gdf.drop(columns=["geometry"])).assign(geometry=gdf.geometry)
    
    # ------------------------------------------------------------------
    #  INTERSECTION-BASED TOPOLOGY DEDUPLICATION (NEW)
    # ------------------------------------------------------------------
    
    def _topology_based_deduplication(self, centerlines: pd.DataFrame) -> pd.DataFrame:
        """
        Apply intersection-based topology and Fréchet averaging for deduplication.
        
        This replaces the simple winner-takes-all parallel merge with:
        1. Detect all intersection points (endpoint + mid-segment)
        2. Split lines at intersection points
        3. Group overlapping segments between same node pairs
        4. Average each group using Fréchet-weighted median
        5. Build final topology network
        
        Preserves curvature and Z-level separation.
        """
        config = self.config
        
        if not config.enable_intersection_topology:
            return centerlines
        
        if centerlines.empty:
            return centerlines
        
        print("  Applying intersection-based topology deduplication...")
        
        # Convert to projected CRS for metric operations
        gdf = gpd.GeoDataFrame(centerlines, crs="EPSG:4326")
        gdf = gdf.to_crs(self.crs_projected)
        
        geometries = list(gdf.geometry)
        segment_ids = list(range(len(geometries)))
        supports = gdf["weighted_support"].values if "weighted_support" in gdf.columns else np.ones(len(gdf))
        altitudes = gdf["altitude_mean"].values if "altitude_mean" in gdf.columns else [np.nan] * len(gdf)
        sources = gdf["source"].values if "source" in gdf.columns else ["unknown"] * len(gdf)
        
        # Step 1: Detect intersections
        print("    Detecting intersection points...")
        intersections, segment_nodes = detect_intersections(
            geometries=geometries,
            segment_ids=segment_ids,
            snap_radius_m=config.intersection_snap_radius_m,
            min_degree=config.intersection_min_degree,
        )
        print(f"    Found {len(intersections)} intersection nodes")
        
        # Step 2: Split lines at mid-segment intersections
        print("    Splitting lines at intersection points...")
        split_geoms, split_ids, node_assignments = split_lines_at_intersections(
            geometries=geometries,
            segment_ids=segment_ids,
            intersections=intersections,
            tolerance_m=config.intersection_snap_radius_m,
        )
        print(f"    Split into {len(split_geoms)} atomic segments")
        
        # Step 2.5: Properly assign node IDs to ALL segment endpoints
        # The split function only assigns nodes to split points, not to original endpoints
        # We need to match each endpoint to the nearest intersection node
        print("    Assigning node IDs to all segment endpoints...")
        int_coords = np.array([(n.x, n.y) for n in intersections])
        int_tree = cKDTree(int_coords) if len(int_coords) > 0 else None
        
        proper_node_assignments = []
        for i, geom in enumerate(split_geoms):
            if geom is None or geom.is_empty:
                proper_node_assignments.append((None, None))
                continue
            
            coords = list(geom.coords)
            start_pt = np.array([coords[0][0], coords[0][1]])
            end_pt = np.array([coords[-1][0], coords[-1][1]])
            
            start_node = None
            end_node = None
            
            if int_tree is not None:
                # Find nearest intersection for start point
                dist_start, idx_start = int_tree.query(start_pt)
                if dist_start < config.intersection_snap_radius_m * 1.5:
                    start_node = intersections[idx_start].node_id
                
                # Find nearest intersection for end point
                dist_end, idx_end = int_tree.query(end_pt)
                if dist_end < config.intersection_snap_radius_m * 1.5:
                    end_node = intersections[idx_end].node_id
            
            proper_node_assignments.append((start_node, end_node))
        
        # Map original attributes to split segments
        split_supports = []
        split_altitudes = []
        split_sources = []
        
        orig_id_to_idx = {sid: i for i, sid in enumerate(segment_ids)}
        for sid in split_ids:
            if sid in orig_id_to_idx:
                orig_idx = orig_id_to_idx[sid]
                split_supports.append(supports[orig_idx])
                split_altitudes.append(altitudes[orig_idx])
                split_sources.append(sources[orig_idx])
            else:
                split_supports.append(1.0)
                split_altitudes.append(np.nan)
                split_sources.append("unknown")
        
        # Step 3: Group overlapping segments
        if config.enable_frechet_averaging:
            print("    Grouping parallel segments for averaging...")
            avg_config = SegmentAveragingConfig(
                corridor_buffer_m=config.frechet_corridor_buffer_m,
                heading_tolerance_deg=config.frechet_heading_tolerance_deg,
                min_corridor_overlap=config.frechet_min_overlap,
                resample_spacing_m=config.frechet_resample_spacing_m,
                frechet_eccentricity_power=config.frechet_eccentricity_power,
                z_separation_threshold_m=config.z_separation_threshold_m,
            )
            
            grouper = SegmentGrouper(avg_config)
            groups = grouper.group_segments(
                geometries=split_geoms,
                segment_ids=split_ids,
                supports=split_supports,
                altitudes=split_altitudes,
                node_pairs=proper_node_assignments,
            )
            
            n_groups = len(groups)
            n_multi = sum(1 for g in groups if g["member_count"] > 1)
            print(f"    Created {n_groups} groups ({n_multi} with multiple members)")
            
            # Step 4: Average each group
            print("    Computing Fréchet-weighted averages...")
            averaged = average_segment_groups(
                geometries=split_geoms,
                segment_ids=split_ids,
                supports=split_supports,
                groups=groups,
                altitudes=split_altitudes,
                source_types=split_sources,
                config=avg_config,
            )
            print(f"    Averaged to {len(averaged)} segments")
        else:
            # No averaging - just use split segments directly
            averaged = []
            for i, (geom, sid) in enumerate(zip(split_geoms, split_ids)):
                if geom is not None and not geom.is_empty:
                    averaged.append({
                        "geometry": geom,
                        "weighted_support": split_supports[i],
                        "member_count": 1,
                        "altitude_mean": split_altitudes[i],
                        "source": split_sources[i],
                    })
        
        # Step 5: Build topology network
        print("    Building topology network...")
        topo_config = TopologyConfig(
            snap_radius_m=config.intersection_snap_radius_m,
            min_segment_length_m=config.min_centerline_length_m,
        )
        
        topo_geoms = [r["geometry"] for r in averaged]
        topo_supports = [r["weighted_support"] for r in averaged]
        topo_alts = [r.get("altitude_mean", np.nan) for r in averaged]
        topo_sources = [r.get("source", "unknown") for r in averaged]
        
        nodes, edges = build_topology(
            geometries=topo_geoms,
            supports=topo_supports,
            altitudes=topo_alts,
            sources=topo_sources,
            config=topo_config,
        )
        print(f"    Built network: {len(nodes)} nodes, {len(edges)} edges")
        
        # Convert back to DataFrame
        result_records = []
        for edge in edges:
            # Convert geometry back to WGS84
            geom_wgs = shapely_transform(
                lambda x, y: self.to_wgs.transform(x, y),
                edge.geometry
            )
            
            result_records.append({
                "geometry": geom_wgs,
                "length_m": edge.length_m,
                "weighted_support": edge.support,
                "altitude_mean": edge.altitude_mean,
                "source": edge.source,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "z_level": edge.z_level,
            })
        
        if not result_records:
            print("    Warning: No segments after topology building, returning original")
            return centerlines
        
        result_df = pd.DataFrame(result_records)
        result_gdf = gpd.GeoDataFrame(result_df, geometry="geometry", crs="EPSG:4326")
        
        reduction = len(centerlines) - len(result_df)
        print(f"    Topology deduplication: {len(centerlines)} -> {len(result_df)} ({reduction} removed)")
        
        return result_gdf

    # ------------------------------------------------------------------
    #  ROUNDABOUT ZONE CLEANUP
    # ------------------------------------------------------------------
    
    def _suppress_roundabout_centerlines(
        self,
        centerlines: pd.DataFrame,
        roundabouts: List[dict],
    ) -> pd.DataFrame:
        """
        Remove centerlines that fall mostly inside detected roundabout zones.
        
        A centerline is suppressed if >70% of its length lies within the
        buffered roundabout circle. This prevents the overlapping spoke/through
        lines that the clustering creates from remaining after roundabout detection.
        """
        if not roundabouts or centerlines.empty:
            return centerlines
        
        print("  Suppressing centerlines inside roundabout zones...")
        
        gdf = gpd.GeoDataFrame(centerlines, crs="EPSG:4326")
        gdf = gdf.to_crs(self.crs_projected)
        
        # Build buffered roundabout zones (buffer = 1.3× radius to catch spoke lines)
        ra_zones = []
        for ra in roundabouts:
            # Convert roundabout geometry to projected CRS
            ra_geom_wgs = ra["geometry"]
            coords_wgs = np.array(ra_geom_wgs.coords)
            proj_x, proj_y = self.to_proj.transform(coords_wgs[:, 0], coords_wgs[:, 1])
            ra_geom_proj = LineString(zip(proj_x, proj_y))
            
            # Compute centroid and buffer
            centroid = ra_geom_proj.centroid
            radius = ra.get("radius", 20.0)
            buffer_radius = radius * 1.3 + 5.0  # Extra margin for spoke lines
            zone = centroid.buffer(buffer_radius)
            ra_zones.append({
                "zone": zone,
                "centroid": centroid,
                "radius": radius,
                "geom_proj": ra_geom_proj,
            })
        
        from shapely.ops import unary_union
        combined_zone = unary_union([z["zone"] for z in ra_zones])
        
        to_remove = set()
        for i, row in gdf.iterrows():
            geom = row.geometry
            if not isinstance(geom, LineString) or geom.is_empty:
                continue
            
            try:
                inside = geom.intersection(combined_zone)
                inside_ratio = inside.length / geom.length if geom.length > 0 else 0
            except Exception:
                continue
            
            if inside_ratio > 0.70:
                to_remove.add(i)
        
        if to_remove:
            keep_mask = [i not in to_remove for i in gdf.index]
            gdf = gdf[keep_mask].reset_index(drop=True)
        
        # Convert back
        gdf = gdf.to_crs("EPSG:4326")
        
        print(f"    Suppressed {len(to_remove)} centerlines inside roundabout zones")
        return pd.DataFrame(gdf.drop(columns=["geometry"])).assign(geometry=gdf.geometry)
    
    def _stitch_to_roundabouts(
        self,
        centerlines: pd.DataFrame,
        roundabouts: List[dict],
    ) -> pd.DataFrame:
        """
        Extend approaching road centerlines to connect to roundabout circles.
        
        For each roundabout, find centerline endpoints that are close to
        (but not on) the roundabout circle, and extend them to snap onto it.
        """
        if not roundabouts or centerlines.empty:
            return centerlines
        
        print("  Stitching road endpoints to roundabout circles...")
        
        gdf = gpd.GeoDataFrame(centerlines, crs="EPSG:4326")
        gdf = gdf.to_crs(self.crs_projected)
        
        snap_distance = 25.0  # Max distance to snap an endpoint to a roundabout
        stitched_count = 0
        
        # Build projected roundabout geometries
        ra_proj_list = []
        for ra in roundabouts:
            ra_geom_wgs = ra["geometry"]
            coords_wgs = np.array(ra_geom_wgs.coords)
            proj_x, proj_y = self.to_proj.transform(coords_wgs[:, 0], coords_wgs[:, 1])
            ra_geom_proj = LineString(zip(proj_x, proj_y))
            ra_proj_list.append({
                "geom": ra_geom_proj,
                "radius": ra.get("radius", 20.0),
            })
        
        new_geoms = list(gdf.geometry)
        
        for i, geom in enumerate(new_geoms):
            if not isinstance(geom, LineString) or geom.is_empty or len(geom.coords) < 2:
                continue
            
            coords = list(geom.coords)
            modified = False
            
            for ra_info in ra_proj_list:
                ra_geom = ra_info["geom"]
                
                # Check start endpoint
                start_pt = geom.coords[0]
                start_dist = ra_geom.distance(Point(start_pt))
                
                if start_dist < snap_distance and start_dist > 2.0:
                    # Snap to nearest point on roundabout circle
                    from shapely.ops import nearest_points
                    snap_pt = nearest_points(ra_geom, Point(start_pt))[0]
                    coords[0] = (snap_pt.x, snap_pt.y) + coords[0][2:]
                    modified = True
                
                # Check end endpoint
                end_pt = geom.coords[-1]
                end_dist = ra_geom.distance(Point(end_pt))
                
                if end_dist < snap_distance and end_dist > 2.0:
                    from shapely.ops import nearest_points
                    snap_pt = nearest_points(ra_geom, Point(end_pt))[0]
                    coords[-1] = (snap_pt.x, snap_pt.y) + coords[-1][2:]
                    modified = True
            
            if modified:
                try:
                    new_line = LineString(coords)
                    if not new_line.is_empty and new_line.length > 0:
                        new_geoms[i] = new_line
                        stitched_count += 1
                except Exception:
                    pass
        
        gdf = gdf.copy()
        gdf["geometry"] = new_geoms
        gdf = gdf.to_crs("EPSG:4326")
        
        print(f"    Stitched {stitched_count} centerline endpoints to roundabouts")
        return pd.DataFrame(gdf.drop(columns=["geometry"])).assign(geometry=gdf.geometry)
    
    # ------------------------------------------------------------------
    #  ROUNDABOUT DETECTION
    # ------------------------------------------------------------------
    
    def _detect_roundabouts(self) -> List[dict]:
        """
        Detect roundabouts from VPD traces using arc-based curvature analysis.
        
        Returns list of roundabout dictionaries with geometry in WGS84.
        """
        config = self.config
        
        if not config.enable_roundabout_detection:
            return []
        
        print("  Detecting roundabouts from trace curvature...")
        
        # Configure roundabout detector with balanced settings
        # (not too strict to miss roundabouts, not too relaxed to have false positives)
        ra_config = RoundaboutConfig(
            min_radius_m=config.roundabout_min_radius_m,
            max_radius_m=config.roundabout_max_radius_m,
            min_unique_traces=3,  # Require at least 3 unique traces
            cluster_radius_m=config.roundabout_cluster_radius_m,
            # Arc detection parameters - TIGHTENED to remove false positives
            min_arc_heading_deg=160.0,  # Real roundabouts need ~180° turn
            min_arc_segments=4,  # Require more segments for robust detection
            max_radius_std_ratio=0.20,  # Strict radius consistency
            min_arc_bbox_m=12.0,  # Require reasonable arc spatial extent
            arc_zero_threshold_rad=0.02,  # Balanced threshold
            arc_min_segment_length_m=0.3,  # Standard segment length
            arc_max_zero_run=3,  # Fewer straight segments in arcs
            # Validation thresholds - TIGHTENED
            min_avg_heading_deg=150.0,  # Real roundabouts have high heading change
            min_entry_directions=2,  # At least 2 entry directions required
            direction_separation_deg=50.0,  # Good angular separation between entries
            min_arcs_for_radius_only=5,  # Need 5+ arcs if fewer unique traces
        )
        
        detector = RoundaboutDetector(ra_config)
        
        # Prepare traces for detection (in projected coordinates)
        # Limit processing: sample traces and downsample coordinates for performance
        traces = []
        max_traces_for_roundabout = 3000  # Limit to avoid excessive processing
        sample_stride = max(1, len(self.gdf) // max_traces_for_roundabout)  # Sample every Nth trace
        coord_decimate_factor = 5  # Keep every 5th point (reduced from 20 for accuracy)
        
        for idx, row in self.gdf.iloc[::sample_stride].iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            # Project geometry
            coords_wgs = np.array(geom.coords)
            coords_xy = np.column_stack(
                self.to_proj.transform(coords_wgs[:, 0], coords_wgs[:, 1])
            )
            
            if len(coords_xy) < 10:  # Need enough points after decimation
                continue
            
            # Downsample coordinates for faster processing
            coords_xy_decimated = coords_xy[::coord_decimate_factor]
            if len(coords_xy_decimated) < 4:
                continue
            
            traces.append({"coords": coords_xy_decimated})
        
        if not traces:
            print("    No valid traces for roundabout detection")
            return []
        
        print(f"    Processing {len(traces)} traces (sampled from {len(self.gdf)})")
        
        # Detect roundabouts using arc-based method only (curl is too slow for large datasets)
        # The arc method is more reliable for fine-sampled GPS data anyway
        roundabouts_proj = detector.detect(traces, use_curl=False, use_arc=True)
        
        if not roundabouts_proj:
            print("    No roundabouts detected")
            return []
        
        print(f"    Initial detection: {len(roundabouts_proj)} roundabout candidates")
        
        # Apply post-filter validation to remove false positives
        if config.enable_roundabout_postfilter:
            print("    Applying post-filter validation to remove false positives...")
            postfilter_config = RoundaboutPostFilterConfig(
                min_traces_near_circle=config.roundabout_postfilter_min_traces,
                min_points_on_circle=config.roundabout_postfilter_min_points,
                min_entry_directions=config.roundabout_postfilter_min_directions,
                min_arc_coverage_ratio=config.roundabout_postfilter_min_arc_coverage,
                merge_distance_m=config.roundabout_postfilter_merge_distance_m,
                min_radius_m=config.roundabout_min_radius_m,
                max_radius_m=config.roundabout_max_radius_m,
            )
            roundabouts_proj = validate_roundabouts_post_detection(
                roundabouts_proj, traces, postfilter_config
            )
            print(f"    After post-filter: {len(roundabouts_proj)} validated roundabouts")
        
        if not roundabouts_proj:
            print("    No roundabouts passed validation")
            return []
        
        # Convert roundabout geometries to WGS84
        roundabouts_wgs = []
        for ra in roundabouts_proj:
            geom_proj = ra["geometry"]
            coords_proj = np.array(geom_proj.coords)
            
            # Transform to WGS84
            lon, lat = self.to_wgs.transform(coords_proj[:, 0], coords_proj[:, 1])
            geom_wgs = LineString(zip(lon, lat))
            
            roundabouts_wgs.append({
                "center_x": ra["center_x"],
                "center_y": ra["center_y"],
                "radius": ra["radius"],
                "geometry": geom_wgs,
                "source": "roundabout",
                "length_m": geom_proj.length,
                "support": 10.0,  # High support for roundabouts
                "weighted_support": 20.0,
            })
        
        print(f"    Final: {len(roundabouts_wgs)} roundabouts")
        return roundabouts_wgs
    
    # ------------------------------------------------------------------
    #  MAIN PIPELINE
    # ------------------------------------------------------------------
    
    def run(self):
        """Execute the Kharita-style centerline generation pipeline."""
        print("=" * 60)
        print("  PHASE 2: Kharita-style Centerline Generation")
        print("=" * 60)
        t0 = time.time()
        
        # Load data
        self.load_and_project()
        
        if len(self.gdf) == 0:
            print("  No valid traces found. Exiting.")
            return None
        
        # Sample traces (now includes altitude for Z-level handling at interchanges)
        xs, ys, headings, weights, altitudes, trace_ranges, trace_meta = self._sample_traces()
        
        if len(xs) == 0:
            print("  No sample points generated. Exiting.")
            return None
        
        # Store trace point coordinates for curve-aware interpolation
        self._trace_points_xy = np.column_stack([xs, ys])
        self._trace_labels = None  # Will be set after clustering
        
        # Cluster points (with Z-level separation for bridge/overpass handling)
        labels, nodes = self._kharita_clustering(xs, ys, headings, weights, altitudes)
        self.labels = labels
        self._trace_labels = labels  # Store for curve interpolation
        self.nodes_df = nodes
        
        if nodes.empty:
            print("  No clusters created. Exiting.")
            return None
        
        # Pre-compute edge trace points for fast curve interpolation
        self._precompute_edge_trace_points()
        
        # Build node coordinate lookup
        node_xy = {
            int(r.node_id): (float(r.x), float(r.y))
            for r in nodes.itertuples(index=False)
        }
        
        # Build edge graph
        edge_support = self._build_edge_graph(
            labels, xs, ys, headings, weights, trace_ranges, trace_meta
        )
        
        # Apply dynamic weighting
        if self.config.enable_dynamic_weighting:
            dyn_cfg = DynamicWeightConfig(
                enabled=True,
                lambda_vpd=self.config.dyn_lambda_vpd,
                road_likeness_beta=self.config.dyn_road_likeness_beta,
                road_likeness_tau=self.config.dyn_road_likeness_tau,
            )
            apply_dynamic_weighting_to_edges(
                edge_support=edge_support,
                node_xy=node_xy,
                config=dyn_cfg,
            )
        
        # Prune edges
        edge_support = self._prune_edges(edge_support, node_xy)
        
        # Build edges dataframe
        edge_rows = []
        for (u, v), s in edge_support.items():
            x1, y1 = node_xy[u]
            x2, y2 = node_xy[v]
            lon1, lat1 = self.to_wgs.transform(x1, y1)
            lon2, lat2 = self.to_wgs.transform(x2, y2)
            line_wgs = LineString([(lon1, lat1), (lon2, lat2)])
            
            support = max(float(s["support"]), 1.0)
            edge_rows.append({
                "u": u,
                "v": v,
                "support": float(s["support"]),
                "weighted_support": float(s["weighted_support"]),
                "vpd_support": float(s["vpd_support"]),
                "hpd_support": float(s["hpd_support"]),
                "mean_step_length_m": float(s["length_sum"] / support),
                "geometry": line_wgs,
            })
        
        self.edges_df = pd.DataFrame(edge_rows)
        
        # Stitch centerlines
        centerlines = self._stitch_centerlines(edge_support, node_xy)
        
        # Candidate selection
        centerlines = self._candidate_selection(centerlines)
        
        # Merge parallel overlapping centerlines (fixes curve clustering issue)
        centerlines = self._merge_parallel_centerlines(centerlines)
        
        # NEW: Apply intersection-based topology and Fréchet averaging
        centerlines = self._topology_based_deduplication(centerlines)
        self.centerlines_df = centerlines
        
        # Detect roundabouts
        roundabouts = self._detect_roundabouts()
        self.roundabouts = roundabouts
        
        # Combine centerlines with roundabouts
        if roundabouts:
            # CRITICAL FIX: suppress centerlines inside roundabout zones BEFORE combining
            centerlines = self._suppress_roundabout_centerlines(centerlines, roundabouts)
            
            # Stitch approaching road endpoints to roundabout circles
            centerlines = self._stitch_to_roundabouts(centerlines, roundabouts)
            self.centerlines_df = centerlines
            
            ra_df = pd.DataFrame(roundabouts)
            # Ensure roundabouts have required columns
            for col in ["node_path", "construction_percent_mean", "traffic_signal_count_mean", 
                        "u", "v", "endpoint_dist_m"]:
                if col not in ra_df.columns:
                    ra_df[col] = None if col == "node_path" else 0.0
            
            combined = pd.concat([centerlines, ra_df], ignore_index=True)
        else:
            combined = centerlines
        
        # Export
        print(f"  Exporting to {self.output_path}...")
        if len(combined) > 0:
            out_gdf = gpd.GeoDataFrame(combined, crs="EPSG:4326")
            out_gdf.to_file(self.output_path, driver="GPKG")
        else:
            empty_gdf = gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")
            empty_gdf.to_file(self.output_path, driver="GPKG")
        
        elapsed = time.time() - t0
        total_km = combined["length_m"].sum() / 1000.0 if len(combined) > 0 else 0
        
        print(f"\nPhase 2 complete in {elapsed:.1f}s")
        print(f"  Nodes:       {len(nodes)}")
        print(f"  Edges:       {len(self.edges_df)}")
        print(f"  Centerlines: {len(centerlines)} ({total_km:.1f} km)")
        print(f"  Roundabouts: {len(roundabouts)}")
        
        gc.collect()
        return self.centerlines_df


# Alias for backwards compatibility
KDESkeletonizer = KharitaCenterlineGenerator


if __name__ == "__main__":
    VPD_INPUT = os.path.join(PROJECT_ROOT, "data", "interim_sample_phase1.gpkg")
    OUTPUT = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
    
    generator = KharitaCenterlineGenerator(VPD_INPUT, OUTPUT)
    generator.run()
