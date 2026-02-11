"""
Phase 3: Probe-based Gap Filling via Roadster-style Clustering

Replaces DBSCAN-based clustering with Fréchet-aware trajectory clustering
for more accurate gap detection and centerline generation.

Algorithm:
  1. Load VPD skeleton and HPD probes
  2. Extract gap portions (probe segments not covered by VPD)
  3. Extract subtrajectories via sliding window
  4. Cluster subtrajectories using Fréchet distance + heading
  5. Construct representative centerlines from clusters
  6. Infer vertices from endpoints and crossings
  7. Build edge graph with quality filtering
"""

import gc
import math
import os
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring, unary_union
from shapely.ops import transform as shapely_transform

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import algorithm utilities
from src.algorithms.centerline_utils import (
    angle_diff_deg,
    bearing_from_xy,
    resample_polyline,
    discrete_frechet_distance,
    weighted_median,
    smooth_polyline_preserve_turns,
)
from src.algorithms.dynamic_weighting import compute_trace_weight


@dataclass
class RoadsterConfig:
    """Configuration for Roadster-style trajectory clustering."""
    
    # Subtrajectory extraction
    subtraj_window_m: float = 65.0
    subtraj_step_m: float = 18.0
    subtraj_resample_points: int = 14
    max_windows_per_segment: int = 250
    
    # Subtrajectory clustering
    cluster_center_radius_m: float = 35.0
    cluster_heading_tolerance_deg: float = 35.0
    cluster_frechet_eps_m: float = 20.0
    cluster_altitude_eps_m: float = 5.0
    cluster_min_members: int = 2  # Reduced from 3 for better gap recovery
    cluster_min_weighted_support: float = 3.0  # Reduced from 5.5
    
    # Direction handling
    allow_opposite_direction_merge: bool = False
    opposite_direction_tolerance_deg: float = 25.0
    
    # Representative construction
    representative_spacing_m: float = 5.0
    representative_turn_deg: float = 28.0
    representative_smooth_passes: int = 1
    
    # Vertex inference
    vertex_snap_m: float = 18.0
    vertex_altitude_eps_m: float = 5.0
    edge_node_snap_m: float = 20.0
    
    # Edge filtering
    min_edge_length_m: float = 10.0  # Reduced from 18.0
    min_edge_support: float = 2.0  # Reduced from 4.5
    dangling_length_m: float = 20.0  # Reduced from 30.0
    dangling_min_support: float = 3.5  # Reduced from 8.0
    
    # Source weighting
    hpd_base_weight: float = 0.9
    quality_weight_boost: float = 0.45
    
    # Gap extraction
    skeleton_buffer_m: float = 7.0  # Reduced from 10.0 to find more gaps
    min_gap_length_m: float = 6.0  # Reduced from 8.0


class RoadsterProbeRecovery:
    """
    Phase 3: Fill gaps in VPD skeleton using HPD probes.
    
    Uses Roadster-style trajectory clustering:
      - Extract sliding-window subtrajectories
      - Cluster by Fréchet similarity + heading
      - Build representative centerlines from clusters
      - Infer intersections and connectivity
    """
    
    def __init__(
        self,
        skeleton_path: str,
        hpd_path: str,
        output_dir: str = "data",
        config: RoadsterConfig = None,
    ):
        self.skeleton_path = skeleton_path
        self.hpd_path = hpd_path
        self.output_dir = output_dir
        self.config = config or RoadsterConfig()
        
        self.local_crs = None
        self.to_proj = None
        self.to_wgs = None
        
        self.skeleton = None
        self.probes = None
        self.probe_skeleton_gdf = None
    
    # ------------------------------------------------------------------
    #  DATA LOADING
    # ------------------------------------------------------------------
    
    def load_data(self):
        """Load VPD skeleton and HPD probes."""
        print("  Loading data...")
        
        # Load skeleton
        if not os.path.exists(self.skeleton_path):
            raise FileNotFoundError(f"Skeleton not found: {self.skeleton_path}")
        
        self.skeleton = gpd.read_file(self.skeleton_path)
        
        # Set up projection
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
        
        # Create transformers
        wgs84 = CRS.from_epsg(4326)
        self.to_proj = Transformer.from_crs(wgs84, self.local_crs, always_xy=True)
        self.to_wgs = Transformer.from_crs(self.local_crs, wgs84, always_xy=True)
        
        # Project skeleton
        self.skeleton = self.skeleton.to_crs(self.local_crs)
        print(f"    Skeleton: {len(self.skeleton)} segments")
        
        # Load probes
        if not os.path.exists(self.hpd_path):
            print(f"    WARNING: HPD not found at {self.hpd_path}")
            self._load_hpd_from_csvs()
        else:
            self.probes = gpd.read_file(self.hpd_path)
            if len(self.probes) > 0 and self.probes.crs != self.local_crs:
                self.probes = self.probes.to_crs(self.local_crs)
            print(f"    Probes: {len(self.probes)} traces")
    
    def _load_hpd_from_csvs(self):
        """Fallback: load HPD from raw CSVs."""
        hpd_dir = os.path.join(PROJECT_ROOT, "data", "Kosovo_HPD")
        files = [
            os.path.join(hpd_dir, "XKO_HPD_week_1.csv"),
            os.path.join(hpd_dir, "XKO_HPD_week_2.csv"),
        ]
        
        all_probes = []
        for probe_path in files:
            if not os.path.exists(probe_path):
                continue
            
            df = pd.read_csv(probe_path)
            df.sort_values(by=["traceid", "time"], inplace=True)
            
            # Relaxed speed filter
            if "speed" in df.columns:
                trace_avg = df.groupby("traceid")["speed"].mean()
                fast = trace_avg[trace_avg >= 3.0].index
                df = df[df["traceid"].isin(fast)]
            
            for tid, group in df.groupby("traceid"):
                if len(group) < 2:
                    continue
                coords = list(zip(group["longitude"], group["latitude"]))
                geom = LineString(coords)
                if not geom.is_empty:
                    avg_spd = group["speed"].mean() if "speed" in group.columns else None
                    all_probes.append({
                        "traceid": tid,
                        "geometry": geom,
                        "avg_speed": avg_spd,
                    })
        
        self.probes = gpd.GeoDataFrame(all_probes, crs="EPSG:4326")
        if len(self.probes) > 0:
            self.probes = self.probes.to_crs(self.local_crs)
        print(f"    Probes (CSV fallback): {len(self.probes)} traces")
    
    # ------------------------------------------------------------------
    #  GAP EXTRACTION
    # ------------------------------------------------------------------
    
    def extract_gaps(self) -> List[dict]:
        """
        Extract probe portions NOT covered by VPD skeleton.
        
        Returns list of gap segment dictionaries with projected coordinates.
        """
        print(f"  Extracting gaps (outside {self.config.skeleton_buffer_m}m of VPD)...")
        config = self.config
        
        if len(self.probes) == 0:
            return []
        
        # Build skeleton buffer
        skel_buffered = self.skeleton.geometry.buffer(config.skeleton_buffer_m)
        skel_sindex = gpd.GeoDataFrame(geometry=skel_buffered.values, crs=self.local_crs).sindex
        
        gap_segments = []
        n_probes = len(self.probes)
        
        for i in range(n_probes):
            row = self.probes.iloc[i]
            probe_geom = row.geometry
            
            if not isinstance(probe_geom, (LineString, MultiLineString)):
                continue
            
            # Find nearby skeleton buffers
            candidate_idxs = list(skel_sindex.intersection(probe_geom.bounds))
            
            if not candidate_idxs:
                # No nearby skeleton — keep entire probe as gap
                if isinstance(probe_geom, LineString) and probe_geom.length >= config.min_gap_length_m:
                    coords = np.array(probe_geom.coords)
                    gap_segments.append({
                        "geometry": probe_geom,
                        "coords_xy": coords,
                        "length_m": probe_geom.length,
                        "traceid": row.get("traceid", i),
                        "avg_speed": row.get("avg_speed", None),
                    })
                continue
            
            # Compute difference with skeleton buffer
            try:
                local_buffer = unary_union(skel_buffered.iloc[candidate_idxs].values)
                diff = probe_geom.difference(local_buffer)
            except Exception:
                continue
            
            # Extract LineStrings from difference
            if diff.is_empty:
                continue
            
            lines = []
            if isinstance(diff, LineString):
                lines = [diff]
            elif isinstance(diff, MultiLineString):
                lines = list(diff.geoms)
            elif hasattr(diff, "geoms"):
                lines = [g for g in diff.geoms if isinstance(g, LineString)]
            
            for line in lines:
                if line.length >= config.min_gap_length_m:
                    coords = np.array(line.coords)
                    gap_segments.append({
                        "geometry": line,
                        "coords_xy": coords,
                        "length_m": line.length,
                        "traceid": row.get("traceid", i),
                        "avg_speed": row.get("avg_speed", None),
                    })
            
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{n_probes} probes, gaps: {len(gap_segments)}")
        
        print(f"    Extracted {len(gap_segments)} gap segments")
        return gap_segments
    
    # ------------------------------------------------------------------
    #  SUBTRAJECTORY EXTRACTION
    # ------------------------------------------------------------------
    
    def _extract_subtrajectories(self, gap_segments: List[dict]) -> List[dict]:
        """Extract sliding-window subtrajectories from gap segments."""
        print("  Extracting subtrajectories...")
        config = self.config
        
        subs = []
        sid = 0
        
        for seg in gap_segments:
            coords = seg["coords_xy"]
            if len(coords) < 2:
                continue
            
            length_m = seg["length_m"]
            spacing = max(length_m / max(len(coords) - 1, 1), 0.5)
            
            # Window parameters
            win_pts = max(4, int(round(config.subtraj_window_m / spacing)))
            step_pts = max(2, int(round(config.subtraj_step_m / spacing)))
            
            if len(coords) < win_pts:
                win_pts = len(coords)
                step_pts = len(coords)
            
            # Generate window start positions
            starts = list(range(0, max(len(coords) - win_pts + 1, 1), step_pts))
            if not starts:
                starts = [0]
            last_start = max(0, len(coords) - win_pts)
            if starts[-1] != last_start:
                starts.append(last_start)
            if len(starts) > config.max_windows_per_segment:
                starts = starts[:config.max_windows_per_segment]
            
            # Compute trace weight
            pweight = compute_trace_weight(
                source="HPD",
                length_m=length_m,
                n_points=len(coords),
                heading_consistency=0.5,
                hpd_base_weight=config.hpd_base_weight,
            )
            
            for s in starts:
                e = min(s + win_pts, len(coords))
                if e - s < 4:
                    continue
                
                c = coords[s:e]
                rep = resample_polyline(c, config.subtraj_resample_points)
                
                # Compute heading
                h = bearing_from_xy(c[0, 0], c[0, 1], c[-1, 0], c[-1, 1])
                center = np.mean(rep, axis=0)
                
                subs.append({
                    "sub_id": sid,
                    "segment_id": seg.get("traceid", sid),
                    "coords": c,
                    "rep": rep,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "heading": float(h),
                    "weight": float(pweight),
                })
                sid += 1
        
        print(f"    Extracted {len(subs)} subtrajectories")
        return subs
    
    # ------------------------------------------------------------------
    #  FRÉCHET-AWARE CLUSTERING
    # ------------------------------------------------------------------
    
    def _cluster_subtrajectories(self, subtrajs: List[dict]) -> List[dict]:
        """
        Cluster subtrajectories using Fréchet distance + heading.
        
        Uses connected components algorithm:
          1. Build adjacency based on center distance, heading, and Fréchet
          2. Find connected components
          3. Filter components by size and support
        """
        print("  Clustering subtrajectories (Fréchet + heading)...")
        t0 = time.time()
        config = self.config
        
        if not subtrajs:
            return []
        
        n = len(subtrajs)
        
        # Build spatial index on centers
        centers = np.column_stack([
            np.asarray([s["center_x"] for s in subtrajs], dtype=np.float64),
            np.asarray([s["center_y"] for s in subtrajs], dtype=np.float64),
        ])
        tree = cKDTree(centers)
        
        # Build adjacency graph
        adj: List[List[int]] = [[] for _ in range(n)]
        edges_checked = 0
        
        for i in range(n):
            # Find nearby candidates
            cand = tree.query_ball_point(centers[i], r=config.cluster_center_radius_m)
            si = subtrajs[i]
            
            for j in cand:
                if j <= i:
                    continue
                
                sj = subtrajs[j]
                
                # Check heading compatibility
                hd = angle_diff_deg(si["heading"], sj["heading"])
                opposite = hd >= (180.0 - config.opposite_direction_tolerance_deg)
                
                if opposite and not config.allow_opposite_direction_merge:
                    continue
                if hd > config.cluster_heading_tolerance_deg and not opposite:
                    continue
                
                # Compute Fréchet distance
                d_f = discrete_frechet_distance(
                    np.asarray(si["rep"], dtype=np.float64),
                    np.asarray(sj["rep"], dtype=np.float64),
                )
                edges_checked += 1
                
                if d_f > config.cluster_frechet_eps_m:
                    continue
                
                # Add edge
                adj[i].append(j)
                adj[j].append(i)
        
        print(f"    Checked {edges_checked} Fréchet pairs")
        
        # Find connected components
        visited = np.zeros(n, dtype=bool)
        clusters = []
        cid = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            # BFS to find component
            stack = [i]
            visited[i] = True
            comp = []
            
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            
            # Filter by size and support
            if len(comp) < config.cluster_min_members:
                continue
            
            members = [subtrajs[idx] for idx in comp]
            weighted_support = float(np.sum([m["weight"] for m in members]))
            
            if weighted_support < config.cluster_min_weighted_support:
                continue
            
            clusters.append({
                "cluster_id": cid,
                "member_idx": comp,
                "members": members,
                "weighted_support": weighted_support,
                "member_count": len(members),
            })
            cid += 1
        
        elapsed = time.time() - t0
        print(f"    Created {len(clusters)} clusters in {elapsed:.1f}s")
        return clusters
    
    # ------------------------------------------------------------------
    #  REPRESENTATIVE CONSTRUCTION
    # ------------------------------------------------------------------
    
    def _build_representatives(self, clusters: List[dict]) -> List[dict]:
        """Build representative centerline for each cluster."""
        print("  Building representative centerlines...")
        config = self.config
        
        reps = []
        
        for cluster in clusters:
            members = cluster["members"]
            if not members:
                continue
            
            # Determine number of points from median member length
            lengths = [LineString(m["coords"]).length for m in members]
            med_len = float(np.median(lengths)) if lengths else 0.0
            n_points = max(int(round(med_len / max(config.representative_spacing_m, 0.5))) + 1, 8)
            
            # Resample all members
            reps_arr = []
            weights = []
            headings = []
            
            for m in members:
                rep = resample_polyline(np.asarray(m["coords"], dtype=np.float64), n_points)
                reps_arr.append(rep)
                weights.append(float(m["weight"]))
                headings.append(float(m["heading"]))
            
            reps_arr = np.asarray(reps_arr, dtype=np.float64)  # (M, N, 2)
            w = np.asarray(weights, dtype=np.float64)
            
            # Weighted median for each point
            x = np.zeros(n_points, dtype=np.float64)
            y = np.zeros(n_points, dtype=np.float64)
            
            for i in range(n_points):
                x[i] = weighted_median(reps_arr[:, i, 0], w)
                y[i] = weighted_median(reps_arr[:, i, 1], w)
            
            rep_coords = np.column_stack([x, y])
            
            # Detect turn points
            turn_idx = {0, n_points - 1}
            for i in range(1, n_points - 1):
                h1 = bearing_from_xy(rep_coords[i-1, 0], rep_coords[i-1, 1], 
                                      rep_coords[i, 0], rep_coords[i, 1])
                h2 = bearing_from_xy(rep_coords[i, 0], rep_coords[i, 1],
                                      rep_coords[i+1, 0], rep_coords[i+1, 1])
                if angle_diff_deg(h1, h2) >= config.representative_turn_deg:
                    turn_idx.add(i)
            
            # Smooth while preserving turns
            smooth_coords = smooth_polyline_preserve_turns(
                rep_coords,
                passes=config.representative_smooth_passes,
                turn_deg=config.representative_turn_deg,
                neighbor_weight=0.25,
            )
            
            rep_line = LineString([(float(xx), float(yy)) for xx, yy in smooth_coords])
            rep_heading = bearing_from_xy(smooth_coords[0, 0], smooth_coords[0, 1],
                                          smooth_coords[-1, 0], smooth_coords[-1, 1])
            
            reps.append({
                "cluster_id": int(cluster["cluster_id"]),
                "member_count": int(cluster["member_count"]),
                "weighted_support": float(cluster["weighted_support"]),
                "line_xy": rep_line,
                "coords": smooth_coords,
                "heading": float(rep_heading),
                "turn_indices": sorted(turn_idx),
            })
        
        print(f"    Built {len(reps)} representatives")
        return reps
    
    # ------------------------------------------------------------------
    #  CENTERLINE CONSTRUCTION
    # ------------------------------------------------------------------
    
    def _build_centerlines(self, reps: List[dict]) -> pd.DataFrame:
        """Convert representatives to centerlines with quality filtering."""
        print("  Building centerlines...")
        config = self.config
        
        if not reps:
            return pd.DataFrame()
        
        # Compute node degrees for dangling detection
        # Use representative endpoints as pseudo-nodes
        endpoints = []
        for rep in reps:
            coords = rep["coords"]
            endpoints.append((coords[0, 0], coords[0, 1]))
            endpoints.append((coords[-1, 0], coords[-1, 1]))
        
        if endpoints:
            ep_arr = np.array(endpoints, dtype=np.float64)
            ep_tree = cKDTree(ep_arr)
            
            # Count nearby endpoints for each representative end
            deg = []
            for i, rep in enumerate(reps):
                coords = rep["coords"]
                start = (coords[0, 0], coords[0, 1])
                end = (coords[-1, 0], coords[-1, 1])
                
                start_neighbors = len(ep_tree.query_ball_point(start, r=config.vertex_snap_m))
                end_neighbors = len(ep_tree.query_ball_point(end, r=config.vertex_snap_m))
                
                deg.append((start_neighbors, end_neighbors))
        else:
            deg = [(1, 1)] * len(reps)
        
        # Build centerline rows
        rows = []
        for i, rep in enumerate(reps):
            line_xy = rep["line_xy"]
            length_m = line_xy.length
            
            if length_m < config.min_edge_length_m:
                continue
            
            weighted_support = rep["weighted_support"]
            
            # Check if dangling (low connectivity endpoints)
            start_deg, end_deg = deg[i]
            is_dangling = start_deg <= 2 or end_deg <= 2
            
            # Filter weak dangling segments
            if is_dangling:
                if length_m < config.dangling_length_m and weighted_support < config.dangling_min_support:
                    continue
            
            # Filter low support
            if weighted_support < config.min_edge_support:
                continue
            
            # Convert to WGS84
            coords_xy = np.array(line_xy.coords)
            lon, lat = self.to_wgs.transform(coords_xy[:, 0], coords_xy[:, 1])
            line_wgs = LineString(zip(lon, lat))
            
            rows.append({
                "geometry": line_wgs,
                "length_m": length_m,
                "weighted_support": weighted_support,
                "cluster_members": rep["member_count"],
                "heading": rep["heading"],
                "source": "Probe",
            })
        
        print(f"    Created {len(rows)} centerlines")
        return pd.DataFrame(rows)
    
    # ------------------------------------------------------------------
    #  MAIN PIPELINE
    # ------------------------------------------------------------------
    
    def run(self):
        """Execute the Roadster-style probe recovery pipeline."""
        print("=" * 60)
        print("  PHASE 3: Roadster-style Probe Gap Filling")
        print("=" * 60)
        t0 = time.time()
        
        # Load data
        self.load_data()
        
        if len(self.probes) == 0:
            print("  No probes found. Exiting.")
            self._save_empty_output()
            return None
        
        # Extract gaps
        gap_segments = self.extract_gaps()
        
        if not gap_segments:
            print("  No gaps found. Exiting.")
            self._save_empty_output()
            return None
        
        # Extract subtrajectories
        subtrajs = self._extract_subtrajectories(gap_segments)
        
        if not subtrajs:
            print("  No subtrajectories extracted. Exiting.")
            self._save_empty_output()
            return None
        
        # Cluster subtrajectories
        clusters = self._cluster_subtrajectories(subtrajs)
        
        if not clusters:
            print("  No clusters formed. Exiting.")
            self._save_empty_output()
            return None
        
        # Build representatives
        reps = self._build_representatives(clusters)
        
        if not reps:
            print("  No representatives built. Exiting.")
            self._save_empty_output()
            return None
        
        # Build centerlines
        centerlines = self._build_centerlines(reps)
        
        # Export
        output_path = os.path.join(self.output_dir, "interim_probe_skeleton_phase3.gpkg")
        print(f"  Exporting to {output_path}...")
        
        if len(centerlines) > 0:
            out_gdf = gpd.GeoDataFrame(centerlines, crs="EPSG:4326")
            out_gdf.to_file(output_path, driver="GPKG")
            self.probe_skeleton_gdf = out_gdf
        else:
            self._save_empty_output()
        
        elapsed = time.time() - t0
        total_km = centerlines["length_m"].sum() / 1000.0 if len(centerlines) > 0 else 0
        
        print(f"\nPhase 3 complete in {elapsed:.1f}s")
        print(f"  Gap segments:  {len(gap_segments)}")
        print(f"  Subtrajectories: {len(subtrajs)}")
        print(f"  Clusters:      {len(clusters)}")
        print(f"  Centerlines:   {len(centerlines)} ({total_km:.1f} km)")
        
        gc.collect()
        return self.probe_skeleton_gdf
    
    def _save_empty_output(self):
        """Save empty output file."""
        output_path = os.path.join(self.output_dir, "interim_probe_skeleton_phase3.gpkg")
        empty_gdf = gpd.GeoDataFrame(
            columns=["geometry", "source", "length_m", "weighted_support"],
            crs="EPSG:4326"
        )
        empty_gdf.to_file(output_path, driver="GPKG")
        self.probe_skeleton_gdf = empty_gdf


# Alias for backwards compatibility
ProbeRecovery = RoadsterProbeRecovery


if __name__ == "__main__":
    SKELETON_FILE = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
    HPD_FILE = os.path.join(PROJECT_ROOT, "data", "interim_hpd_phase1.gpkg")
    
    recovery = RoadsterProbeRecovery(SKELETON_FILE, HPD_FILE)
    recovery.run()
