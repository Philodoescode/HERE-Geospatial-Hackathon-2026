"""
Trajectory Clustering Module (Roadster-inspired)

Implements subtrajectory clustering using discrete Fréchet distance
for more accurate geometric similarity measurement.

Key concepts:
  - Subtrajectory extraction: Window-based sampling of trajectories
  - Fréchet similarity: Measures maximum deviation between trajectory pairs
  - Weighted representative: Robust median-based centerline construction
  - Vertex inference: Detect intersections from crossing trajectories
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring


@dataclass
class TrajectoryClusterConfig:
    """Configuration for trajectory clustering."""
    
    # Subtrajectory extraction
    window_length_m: float = 65.0
    window_step_m: float = 18.0
    resample_points: int = 14
    max_windows_per_trace: int = 250
    
    # Clustering parameters
    center_radius_m: float = 35.0
    heading_tolerance_deg: float = 30.0
    frechet_eps_m: float = 16.0
    altitude_eps_m: float = 5.0
    min_cluster_members: int = 3
    min_cluster_support: float = 5.5
    
    # Representative construction
    representative_spacing_m: float = 5.0
    turn_angle_deg: float = 28.0
    smooth_passes: int = 1
    
    # Direction handling
    allow_opposite_merge: bool = False
    opposite_tolerance_deg: float = 25.0


def _resample_polyline(coords: np.ndarray, n_points: int) -> np.ndarray:
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


def _discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute discrete Fréchet distance between two 2D polylines.
    
    Fréchet distance measures the maximum deviation between two curves,
    considering the order of points ("walking the dog" metaphor).
    
    Lower values mean more similar trajectories.
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


def _bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute bearing from (x1,y1) to (x2,y2) in degrees (0=north, clockwise)."""
    dx = x2 - x1
    dy = y2 - y1
    # atan2 gives angle from east, convert to bearing from north
    return (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    """Compute minimal angle difference between two headings."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _line_heading(coords: np.ndarray) -> float:
    """Get heading of a polyline from first to last point."""
    if len(coords) < 2:
        return 0.0
    return _bearing_from_xy(coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
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


def _smooth_polyline(coords: np.ndarray, keep_idx: set, passes: int) -> np.ndarray:
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


class SubtrajectoryClusterer:
    """
    Cluster trajectories using subtrajectory windows and Fréchet distance.
    
    This produces more accurate centerlines than point-based clustering
    because it considers the full shape of trajectory segments.
    """
    
    def __init__(self, config: Optional[TrajectoryClusterConfig] = None):
        self.config = config or TrajectoryClusterConfig()
    
    def extract_subtrajectories(
        self,
        traces: List[Dict],
    ) -> List[Dict]:
        """
        Extract windowed subtrajectories from full traces.
        
        Each trace is split into overlapping windows for clustering.
        This allows long traces to contribute to multiple road segments.
        """
        cfg = self.config
        subs = []
        sid = 0
        
        for trace in traces:
            coords = np.asarray(trace.get("coords_xy", []), dtype=np.float64)
            if len(coords) < 2:
                continue
            
            trace_len = float(np.sum(np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))))
            if trace_len <= 0:
                continue
            
            # Compute window parameters
            spacing = max(trace_len / max(len(coords) - 1, 1), 0.5)
            win_pts = max(4, int(round(cfg.window_length_m / spacing)))
            step_pts = max(2, int(round(cfg.window_step_m / spacing)))
            
            if len(coords) < win_pts:
                win_pts = len(coords)
                step_pts = len(coords)
            
            # Generate window starts
            starts = list(range(0, max(len(coords) - win_pts + 1, 1), step_pts))
            if not starts:
                starts = [0]
            
            last_start = max(0, len(coords) - win_pts)
            if starts[-1] != last_start:
                starts.append(last_start)
            
            starts = starts[:cfg.max_windows_per_trace]
            
            # Get altitudes if available
            alt = np.asarray(trace.get("altitudes", []), dtype=np.float64)
            if len(alt) != len(coords):
                alt = np.full(len(coords), np.nan, dtype=np.float64)
            
            weight = float(trace.get("weight", 1.0))
            
            # Extract each window
            for s in starts:
                e = min(s + win_pts, len(coords))
                if e - s < 4:
                    continue
                
                win_coords = coords[s:e]
                rep = _resample_polyline(win_coords, cfg.resample_points)
                h = _line_heading(win_coords)
                center = np.mean(rep, axis=0)
                a = float(np.nanmedian(alt[s:e])) if np.isfinite(alt[s:e]).any() else np.nan
                
                subs.append({
                    "sub_id": sid,
                    "trace_id": trace.get("trace_id"),
                    "source": trace.get("source", "VPD"),
                    "coords": win_coords,
                    "rep": rep,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "heading": float(h),
                    "altitude": a,
                    "weight": weight,
                    "day": trace.get("day"),
                    "hour": trace.get("hour"),
                })
                sid += 1
        
        return subs
    
    def cluster_subtrajectories(
        self,
        subtrajs: List[Dict]
    ) -> List[Dict]:
        """
        Cluster subtrajectories by spatial proximity, heading, and Fréchet distance.
        
        Returns list of clusters, each containing member subtrajectories.
        """
        cfg = self.config
        
        if not subtrajs:
            return []
        
        n = len(subtrajs)
        
        # Build spatial index
        centers = np.column_stack([
            np.asarray([s["center_x"] for s in subtrajs], dtype=np.float64),
            np.asarray([s["center_y"] for s in subtrajs], dtype=np.float64),
        ])
        tree = cKDTree(centers)
        
        # Build adjacency graph (connected subtrajectories that are similar)
        adj: List[List[int]] = [[] for _ in range(n)]
        
        for i in range(n):
            cand = tree.query_ball_point(centers[i], r=cfg.center_radius_m)
            si = subtrajs[i]
            
            for j in cand:
                if j <= i:
                    continue
                
                sj = subtrajs[j]
                
                # Heading check
                hd = _angle_diff_deg(si["heading"], sj["heading"])
                opposite = hd >= (180.0 - cfg.opposite_tolerance_deg)
                
                if opposite and not cfg.allow_opposite_merge:
                    continue
                if hd > cfg.heading_tolerance_deg and not opposite:
                    continue
                
                # Altitude check
                if np.isfinite(si["altitude"]) and np.isfinite(sj["altitude"]):
                    if abs(float(si["altitude"]) - float(sj["altitude"])) > cfg.altitude_eps_m:
                        continue
                
                # Fréchet distance check
                d_f = _discrete_frechet(
                    np.asarray(si["rep"], dtype=np.float64),
                    np.asarray(sj["rep"], dtype=np.float64),
                )
                
                if d_f > cfg.frechet_eps_m:
                    continue
                
                # Similar enough - connect
                adj[i].append(j)
                adj[j].append(i)
        
        # Find connected components
        visited = np.zeros(n, dtype=bool)
        clusters = []
        cid = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
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
            
            # Filter small/weak clusters
            if len(comp) < cfg.min_cluster_members:
                continue
            
            members = [subtrajs[idx] for idx in comp]
            weighted_support = float(np.sum([m["weight"] for m in members]))
            
            if weighted_support < cfg.min_cluster_support:
                continue
            
            clusters.append({
                "cluster_id": cid,
                "member_idx": comp,
                "members": members,
                "weighted_support": weighted_support,
                "member_count": len(members),
            })
            cid += 1
        
        return clusters
    
    def build_representative(self, cluster: Dict) -> Optional[Dict]:
        """
        Build a representative centerline for a cluster.
        
        Uses weighted median of resampled coordinates for robustness.
        """
        cfg = self.config
        members = cluster.get("members", [])
        
        if not members:
            return None
        
        # Determine representative complexity
        lengths = [LineString(m["coords"]).length for m in members]
        med_len = float(np.median(lengths)) if lengths else 0.0
        n_points = max(int(round(med_len / max(cfg.representative_spacing_m, 0.5))) + 1, 8)
        
        # Resample all members to same point count
        reps = []
        weights = []
        headings = []
        altitudes = []
        
        for m in members:
            rep = _resample_polyline(np.asarray(m["coords"], dtype=np.float64), n_points)
            reps.append(rep)
            weights.append(float(m["weight"]))
            headings.append(float(m["heading"]))
            altitudes.append(float(m["altitude"]) if np.isfinite(m["altitude"]) else np.nan)
        
        reps_arr = np.asarray(reps, dtype=np.float64)  # (M, N, 2)
        w = np.asarray(weights, dtype=np.float64)
        
        # Weighted median per point
        x = np.zeros(n_points, dtype=np.float64)
        y = np.zeros(n_points, dtype=np.float64)
        
        for i in range(n_points):
            x[i] = _weighted_median(reps_arr[:, i, 0], w)
            y[i] = _weighted_median(reps_arr[:, i, 1], w)
        
        rep = np.column_stack([x, y])
        
        # Detect turn points
        turn_idx = {0, n_points - 1}
        for i in range(1, n_points - 1):
            h1 = _bearing_from_xy(rep[i-1, 0], rep[i-1, 1], rep[i, 0], rep[i, 1])
            h2 = _bearing_from_xy(rep[i, 0], rep[i, 1], rep[i+1, 0], rep[i+1, 1])
            if _angle_diff_deg(h1, h2) >= cfg.turn_angle_deg:
                turn_idx.add(i)
        
        # Smooth
        rep = _smooth_polyline(rep, keep_idx=turn_idx, passes=cfg.smooth_passes)
        
        rep_line = LineString([(float(xx), float(yy)) for xx, yy in rep])
        rep_heading = _line_heading(rep)
        
        source_counts = Counter([m.get("source", "VPD") for m in members])
        
        return {
            "cluster_id": int(cluster["cluster_id"]),
            "member_count": int(cluster["member_count"]),
            "weighted_support": float(cluster["weighted_support"]),
            "line_xy": rep_line,
            "coords": rep,
            "heading": float(rep_heading),
            "turn_indices": sorted(turn_idx),
            "altitude_mean": float(np.nanmedian(np.asarray(altitudes)))
                if np.isfinite(altitudes).any() else np.nan,
            "source_counts": source_counts,
            "day_mode": Counter([m["day"] for m in members if m.get("day")]).most_common(1)[0][0]
                if any(m.get("day") for m in members) else None,
            "hour_mode": Counter([m["hour"] for m in members if m.get("hour")]).most_common(1)[0][0]
                if any(m.get("hour") for m in members) else None,
        }
    
    def cluster_and_build(
        self,
        traces: List[Dict],
        min_length_m: float = 12.0
    ) -> List[Dict]:
        """
        Full pipeline: extract subtrajectories, cluster, and build representatives.
        
        Returns list of representative centerlines.
        """
        subtrajs = self.extract_subtrajectories(traces)
        
        if not subtrajs:
            return []
        
        clusters = self.cluster_subtrajectories(subtrajs)
        
        representatives = []
        for cluster in clusters:
            rep = self.build_representative(cluster)
            if rep and rep["line_xy"].length >= min_length_m:
                representatives.append(rep)
        
        return representatives


class IntersectionDetector:
    """
    Detect road intersections from representative trajectories.
    
    Uses line crossing analysis and endpoint clustering.
    """
    
    def __init__(
        self,
        snap_radius_m: float = 18.0,
        altitude_eps_m: float = 5.0,
        min_crossing_angle_deg: float = 15.0
    ):
        self.snap_radius_m = snap_radius_m
        self.altitude_eps_m = altitude_eps_m
        self.min_crossing_angle_deg = min_crossing_angle_deg
    
    def detect_vertices(
        self,
        representatives: List[Dict]
    ) -> pd.DataFrame:
        """
        Detect vertices (intersections and endpoints) from representatives.
        
        Returns DataFrame with node_id, x, y, z, support columns.
        """
        if not representatives:
            return pd.DataFrame(columns=["node_id", "x", "y", "z", "support"])
        
        candidates = []
        
        for rep in representatives:
            coords = np.asarray(rep["coords"], dtype=np.float64)
            if len(coords) < 2:
                continue
            
            w = float(rep["weighted_support"])
            z = rep.get("altitude_mean", np.nan)
            
            # Endpoints
            h_start = _bearing_from_xy(
                coords[0, 0], coords[0, 1],
                coords[min(1, len(coords)-1), 0], coords[min(1, len(coords)-1), 1]
            )
            h_end = _bearing_from_xy(
                coords[max(0, len(coords)-2), 0], coords[max(0, len(coords)-2), 1],
                coords[-1, 0], coords[-1, 1]
            )
            
            candidates.append({
                "x": coords[0, 0], "y": coords[0, 1], "z": z,
                "heading": h_start, "weight": w, "kind": "endpoint"
            })
            candidates.append({
                "x": coords[-1, 0], "y": coords[-1, 1], "z": z,
                "heading": h_end, "weight": w, "kind": "endpoint"
            })
            
            # Turn points
            for i in rep.get("turn_indices", []):
                if i <= 0 or i >= len(coords) - 1:
                    continue
                h = _bearing_from_xy(
                    coords[i-1, 0], coords[i-1, 1],
                    coords[i+1, 0], coords[i+1, 1]
                )
                candidates.append({
                    "x": coords[i, 0], "y": coords[i, 1], "z": z,
                    "heading": h, "weight": w * 1.2, "kind": "turn"
                })
        
        # Detect crossings between representatives
        for i in range(len(representatives)):
            li = representatives[i]["line_xy"]
            for j in range(i + 1, len(representatives)):
                lj = representatives[j]["line_xy"]
                
                if li.distance(lj) > self.snap_radius_m:
                    continue
                
                inter = li.intersection(lj)
                if inter.is_empty:
                    continue
                
                pts = []
                if isinstance(inter, Point):
                    pts = [inter]
                elif hasattr(inter, "geoms"):
                    pts = [g for g in inter.geoms if isinstance(g, Point)]
                
                if not pts:
                    continue
                
                p = pts[0]
                hd = _angle_diff_deg(
                    representatives[i]["heading"],
                    representatives[j]["heading"]
                )
                
                if hd < self.min_crossing_angle_deg:
                    continue
                
                w = min(
                    representatives[i]["weighted_support"],
                    representatives[j]["weighted_support"]
                ) * max(math.sin(math.radians(hd)), 0.2)
                
                zvals = [
                    representatives[i].get("altitude_mean"),
                    representatives[j].get("altitude_mean")
                ]
                zvals = [z for z in zvals if z is not None and np.isfinite(z)]
                z = float(np.mean(zvals)) if zvals else np.nan
                
                candidates.append({
                    "x": float(p.x), "y": float(p.y), "z": z,
                    "heading": float((representatives[i]["heading"] + representatives[j]["heading"]) / 2.0),
                    "weight": w, "kind": "cross"
                })
        
        if not candidates:
            return pd.DataFrame(columns=["node_id", "x", "y", "z", "support"])
        
        # Cluster nearby candidates into nodes
        groups: List[List[Dict]] = []
        
        for c in sorted(candidates, key=lambda d: d["weight"], reverse=True):
            assigned = False
            
            for g in groups:
                gx = float(np.mean([u["x"] for u in g]))
                gy = float(np.mean([u["y"] for u in g]))
                dist = float(np.hypot(c["x"] - gx, c["y"] - gy))
                
                if dist > self.snap_radius_m:
                    continue
                
                # Altitude compatibility
                zc = c["z"]
                gz_vals = [u["z"] for u in g if np.isfinite(u.get("z", np.nan))]
                if np.isfinite(zc) and gz_vals:
                    if abs(zc - float(np.mean(gz_vals))) > self.altitude_eps_m:
                        continue
                
                g.append(c)
                assigned = True
                break
            
            if not assigned:
                groups.append([c])
        
        # Build output
        rows = []
        for nid, g in enumerate(groups):
            wx = np.asarray([u["weight"] for u in g], dtype=np.float64)
            xs = np.asarray([u["x"] for u in g], dtype=np.float64)
            ys = np.asarray([u["y"] for u in g], dtype=np.float64)
            zvals = np.asarray([u.get("z", np.nan) for u in g], dtype=np.float64)
            
            x = float(np.average(xs, weights=wx))
            y = float(np.average(ys, weights=wx))
            zf = float(np.nanmedian(zvals)) if np.isfinite(zvals).any() else np.nan
            
            rows.append({
                "node_id": int(nid),
                "x": x,
                "y": y,
                "z": zf,
                "support": float(np.sum(wx)),
                "candidate_count": int(len(g)),
            })
        
        return pd.DataFrame(rows)


# Convenience function for integration
def cluster_traces_roadster_style(
    gdf: pd.DataFrame,
    config: Optional[TrajectoryClusterConfig] = None,
    min_centerline_length_m: float = 12.0
) -> List[Dict]:
    """
    Cluster traces using Roadster-style subtrajectory clustering.
    
    Args:
        gdf: GeoDataFrame with geometry column (LineStrings)
        config: Clustering configuration
        min_centerline_length_m: Minimum output length
        
    Returns:
        List of representative dictionaries with 'line_xy' geometry
    """
    # Convert GeoDataFrame to trace list
    traces = []
    for row in gdf.itertuples(index=False):
        geom = row.geometry
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            continue
        
        traces.append({
            "trace_id": getattr(row, "trace_id", None) or getattr(row, "traceid", None),
            "source": getattr(row, "source", "VPD"),
            "coords_xy": list(geom.coords),
            "altitudes": getattr(row, "altitudes", []) if hasattr(row, "altitudes") else [],
            "weight": getattr(row, "weight", 1.0) if hasattr(row, "weight") else 1.0,
            "day": getattr(row, "day", None) if hasattr(row, "day") else None,
            "hour": getattr(row, "hour", None) if hasattr(row, "hour") else None,
        })
    
    if not traces:
        return []
    
    clusterer = SubtrajectoryClusterer(config)
    return clusterer.cluster_and_build(traces, min_centerline_length_m)
