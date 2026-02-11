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
from shapely.geometry import LineString
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
    parallel_merge_buffer_m: float = 6.0
    parallel_merge_heading_tol_deg: float = 25.0


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
    
    def _sample_traces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[int, int]], Dict[int, dict]]:
        """
        Sample all traces at regular intervals.
        
        Returns:
            x_arr: X coordinates of all sample points
            y_arr: Y coordinates of all sample points
            heading_arr: Headings at all sample points
            weight_arr: Weights for all sample points
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
            end = len(xs)
            
            trace_ranges[next_trace_id] = (start, end)
            trace_meta[next_trace_id] = {
                "trace_id": row.get("trace_id", idx),
                "source": "VPD",
                "construction_percent": float(row.get("construction_percent", 0.0) or 0.0),
                "traffic_signal_count": float(row.get("traffic_signal_count", 0.0) or 0.0),
                "day": int(row["day"]) if pd.notnull(row.get("day", np.nan)) else None,
                "hour": int(row["hour"]) if pd.notnull(row.get("hour", np.nan)) else None,
            }
            next_trace_id += 1
        
        elapsed = time.time() - t0
        print(f"    Sampled {len(xs)} points from {next_trace_id} traces in {elapsed:.1f}s")
        
        x_arr = np.asarray(xs, dtype=np.float32)
        y_arr = np.asarray(ys, dtype=np.float32)
        heading_arr = np.asarray(headings_list, dtype=np.float32)
        weight_arr = np.asarray(point_weights, dtype=np.float32)
        
        return x_arr, y_arr, heading_arr, weight_arr, trace_ranges, trace_meta
    
    # ------------------------------------------------------------------
    #  HEADING-AWARE CLUSTERING
    # ------------------------------------------------------------------
    
    def _kharita_clustering(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        headings: np.ndarray,
        point_weights: np.ndarray,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Heading-aware incremental clustering (Kharita-style).
        
        Points are assigned to clusters based on:
          - Spatial distance (within cluster_radius_m)
          - Heading similarity (within heading_tolerance_deg)
        
        Cluster centroids and headings are updated incrementally.
        """
        print("  Performing heading-aware clustering...")
        t0 = time.time()
        config = self.config
        
        n = len(xs)
        if n == 0:
            return np.array([], dtype=np.int32), pd.DataFrame(
                columns=["node_id", "x", "y", "heading", "weight", "point_count"]
            )
        
        labels = np.full(n, -1, dtype=np.int32)
        
        # Cluster state
        cx: List[float] = []  # centroid x
        cy: List[float] = []  # centroid y
        cweight: List[float] = []  # total weight
        csin: List[float] = []  # weighted sin(heading)
        ccos: List[float] = []  # weighted cos(heading)
        ccount: List[int] = []  # point count
        
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
            
            # First point creates first cluster
            if not cx:
                cx.append(x)
                cy.append(y)
                cweight.append(w)
                rad = math.radians(heading)
                csin.append(math.sin(rad) * w)
                ccos.append(math.cos(rad) * w)
                ccount.append(1)
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
            
            # Find best matching cluster (closest in combined spatial+heading distance)
            best_cid = -1
            best_score = float("inf")
            
            for cid in candidate_ids:
                # Compute cluster heading from sin/cos sums
                cheading = (math.degrees(math.atan2(csin[cid], ccos[cid])) + 360.0) % 360.0
                
                # Check heading tolerance
                ad = angle_diff_deg(heading, cheading)
                if ad > config.heading_tolerance_deg:
                    continue
                
                # Combined distance (spatial + heading penalty)
                sd = math.hypot(x - cx[cid], y - cy[cid])
                score = sd + config.heading_distance_weight_m * ad
                
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
            labels[i] = cid
            dirty_count += 1
        
        # Collapse small clusters into nearby larger ones
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
                    if float(d) <= config.cluster_radius_m * 1.25:
                        remap[cid] = target
                
                labels = np.asarray([remap[int(cid)] for cid in labels], dtype=np.int32)
        
        # Reindex labels to compact IDs
        unique = sorted(set(int(v) for v in labels))
        to_new = {old: new for new, old in enumerate(unique)}
        labels = np.asarray([to_new[int(v)] for v in labels], dtype=np.int32)
        
        # Recompute node stats from final assignments
        node_stats = defaultdict(
            lambda: {"xw": 0.0, "yw": 0.0, "w": 0.0, "sinw": 0.0, "cosw": 0.0, "n": 0}
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
        
        # Build nodes dataframe
        rows = []
        for nid in sorted(node_stats):
            s = node_stats[nid]
            ww = s["w"] if s["w"] > 0 else 1.0
            rows.append({
                "node_id": nid,
                "x": s["xw"] / ww,
                "y": s["yw"] / ww,
                "heading": (math.degrees(math.atan2(s["sinw"], s["cosw"])) + 360.0) % 360.0,
                "weight": s["w"],
                "point_count": s["n"],
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
    #  CENTERLINE STITCHING + CANDIDATE SELECTION
    # ------------------------------------------------------------------
    
    def _stitch_centerlines(
        self,
        edge_support: Dict[Tuple[int, int], dict],
        node_xy: Dict[int, Tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Stitch edges into centerline paths with smoothing.
        """
        print("  Stitching centerlines...")
        t0 = time.time()
        config = self.config
        
        # Get centerline paths
        centerline_paths = stitch_centerline_paths(edge_support)
        print(f"    Found {len(centerline_paths)} centerline paths")
        
        centerline_rows = []
        
        for path_nodes in centerline_paths:
            if len(path_nodes) < 2:
                continue
            
            # Get raw coordinates
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
        Merge parallel overlapping centerlines on curves.
        
        This addresses the issue where Kharita clustering creates multiple
        closely-spaced clusters on curves, resulting in overlapping centerlines.
        We identify pairs of parallel lines within a buffer and keep the stronger one.
        """
        config = self.config
        
        if not config.enable_parallel_merge or centerlines.empty:
            return centerlines
        
        print("  Merging parallel overlapping centerlines...")
        
        # Convert to GeoDataFrame for spatial operations
        gdf = gpd.GeoDataFrame(centerlines, crs="EPSG:4326")
        
        # Project to local CRS for accurate distance calculations
        gdf = gdf.to_crs(self.crs_projected)
        
        # Compute headings and lengths
        def get_heading(geom):
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                return 0.0
            coords = np.array(geom.coords)
            dx = coords[-1, 0] - coords[0, 0]
            dy = coords[-1, 1] - coords[0, 1]
            h = np.degrees(np.arctan2(dy, dx)) % 180.0  # Direction-agnostic
            return h
        
        headings = np.array([get_heading(g) for g in gdf.geometry])
        lengths = gdf.geometry.length.values
        supports = gdf["weighted_support"].values if "weighted_support" in gdf.columns else np.ones(len(gdf))
        
        # Build spatial index
        from shapely import STRtree
        tree = STRtree(gdf.geometry.values)
        
        # Find parallel overlapping pairs
        to_remove = set()
        
        for i in range(len(gdf)):
            if i in to_remove:
                continue
            
            geom_i = gdf.geometry.iloc[i]
            buffered = geom_i.buffer(config.parallel_merge_buffer_m)
            
            # Query nearby geometries
            candidates = tree.query(buffered)
            
            for j in candidates:
                if j <= i or j in to_remove:
                    continue
                
                geom_j = gdf.geometry.iloc[j]
                
                # Check heading compatibility (parallel)
                h_diff = abs(headings[i] - headings[j])
                h_diff = min(h_diff, 180 - h_diff)
                
                if h_diff > config.parallel_merge_heading_tol_deg:
                    continue
                
                # Check overlap
                try:
                    intersection = geom_i.buffer(config.parallel_merge_buffer_m).intersection(geom_j)
                    overlap_ratio = intersection.length / min(lengths[i], lengths[j]) if min(lengths[i], lengths[j]) > 0 else 0
                except Exception:
                    continue
                
                if overlap_ratio < 0.5:
                    continue
                
                # Remove the weaker one (lower support or shorter)
                score_i = supports[i] * lengths[i]
                score_j = supports[j] * lengths[j]
                
                if score_i >= score_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # Move to next i since current i is removed
        
        if to_remove:
            keep_mask = [i not in to_remove for i in range(len(gdf))]
            gdf = gdf[keep_mask].reset_index(drop=True)
        
        # Convert back to WGS84
        gdf = gdf.to_crs("EPSG:4326")
        
        removed = len(centerlines) - len(gdf)
        print(f"    Merged {removed} parallel overlapping centerlines")
        
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
        
        # Sample traces
        xs, ys, headings, weights, trace_ranges, trace_meta = self._sample_traces()
        
        if len(xs) == 0:
            print("  No sample points generated. Exiting.")
            return None
        
        # Cluster points
        labels, nodes = self._kharita_clustering(xs, ys, headings, weights)
        self.labels = labels
        self.nodes_df = nodes
        
        if nodes.empty:
            print("  No clusters created. Exiting.")
            return None
        
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
        self.centerlines_df = centerlines
        
        # Detect roundabouts
        roundabouts = self._detect_roundabouts()
        self.roundabouts = roundabouts
        
        # Combine centerlines with roundabouts
        if roundabouts:
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
