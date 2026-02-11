"""Roadster-adapted centerline generation algorithm.

This implementation follows the Roadster pipeline structure while adapting
it to HERE Hackathon data characteristics (high-quality VPD + supplemental HPD):

1) Subtrajectory extraction + clustering (windowed trajectories)
2) Representative construction and refinement
3) Vertex (intersection/end-point) inference
4) Edge construction with topology cleanup

Clustering uses a practical approximation of Fréchet-style similarity:
windowed subtrajectories are compared with discrete Fréchet distance on
resampled points, with direction and altitude constraints.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass, fields
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring, transform

from .base import AlgorithmConfig, BaseCenterlineAlgorithm
from ..preprocessing import (
    SourcePreprocessConfig,
    default_source_preprocess_config,
    preprocess_traces_dataframe,
)
from ..utils import angle_diff_deg, bearing_from_xy


@dataclass
class RoadsterConfig(AlgorithmConfig):
    """Configuration for the Roadster-adapted pipeline."""

    # Subtrajectory extraction
    subtraj_window_m: float = 65.0
    subtraj_step_m: float = 18.0
    subtraj_resample_points: int = 14
    max_windows_per_segment: int = 250

    # Subtrajectory clustering
    cluster_center_radius_m: float = 35.0
    cluster_heading_tolerance_deg: float = 30.0
    cluster_frechet_eps_m: float = 16.0
    cluster_altitude_eps_m: float = 5.0
    cluster_min_members: int = 3
    cluster_min_weighted_support: float = 5.5

    # Direction + altitude safeguards
    allow_opposite_direction_merge: bool = False
    opposite_direction_tolerance_deg: float = 25.0
    altitude_band_m: float = 4.0

    # Representative refinement
    representative_spacing_m: float = 5.0
    representative_turn_deg: float = 28.0
    representative_smooth_passes: int = 1

    # Vertex inference and graph construction
    vertex_snap_m: float = 18.0
    vertex_altitude_eps_m: float = 5.0
    edge_node_snap_m: float = 20.0
    min_edge_length_m: float = 18.0
    min_edge_support: float = 4.5
    dangling_length_m: float = 30.0
    dangling_min_support: float = 8.0

    # Source weighting
    vpd_base_weight: float = 2.4
    hpd_base_weight: float = 0.9
    quality_weight_boost: float = 0.45
    construction_penalty: float = 0.65  # 100% construction -> 35% remaining weight

    # Temporal weighting (optional)
    temporal_slice_day: int | None = None
    temporal_slice_hour: int | None = None
    temporal_mismatch_penalty: float = 0.35


def _resample_polyline(coords: np.ndarray, n_points: int) -> np.ndarray:
    if len(coords) <= 1:
        return coords.copy()
    if n_points <= 2:
        return np.asarray([coords[0], coords[-1]], dtype=np.float64)
    d = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    cd = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cd[-1])
    if total <= 0.0:
        return np.repeat(coords[:1], n_points, axis=0)
    target = np.linspace(0.0, total, n_points, dtype=np.float64)
    x = np.interp(target, cd, coords[:, 0])
    y = np.interp(target, cd, coords[:, 1])
    return np.column_stack([x, y])


def _discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Fréchet distance between two 2D polylines."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
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


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
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


def _smooth_polyline(coords: np.ndarray, keep_idx: set[int], passes: int) -> np.ndarray:
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


def _line_heading(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    return bearing_from_xy(coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1])


def _line_from_point_heading(x: float, y: float, heading_deg: float, span_m: float = 120.0) -> LineString:
    # Heading is 0=north clockwise. Convert to dx,dy in projected XY.
    rad = math.radians(float(heading_deg))
    dx = math.sin(rad) * span_m
    dy = math.cos(rad) * span_m
    return LineString([(x - dx, y - dy), (x + dx, y + dy)])


def _intersection_point(l1: LineString, l2: LineString) -> Point | None:
    inter = l1.intersection(l2)
    if inter.is_empty:
        return None
    if isinstance(inter, Point):
        return inter
    if hasattr(inter, "geoms"):
        pts = [g for g in inter.geoms if isinstance(g, Point)]
        if pts:
            return pts[0]
    return None


class RoadsterAlgorithm(BaseCenterlineAlgorithm):
    name = "roadster"
    description = (
        "Roadster-adapted subtrajectory clustering with representative refinement, "
        "intersection inference, and topology-aware edge graph construction."
    )

    def __init__(self, config: RoadsterConfig | None = None) -> None:
        self.config = config or RoadsterConfig()

    def add_cli_args(self, parser) -> None:
        g = parser.add_argument_group("Roadster algorithm parameters")
        g.add_argument("--subtraj-window-m", type=float, default=65.0)
        g.add_argument("--subtraj-step-m", type=float, default=18.0)
        g.add_argument("--cluster-center-radius-m", type=float, default=35.0)
        g.add_argument("--cluster-frechet-eps-m", type=float, default=16.0)
        g.add_argument("--cluster-heading-tolerance-deg", type=float, default=30.0)
        g.add_argument("--cluster-altitude-eps-m", type=float, default=5.0)
        g.add_argument("--cluster-min-members", type=int, default=3)
        g.add_argument("--cluster-min-weighted-support", type=float, default=5.5)
        g.add_argument("--vertex-snap-m", type=float, default=18.0)
        g.add_argument("--edge-node-snap-m", type=float, default=20.0)
        g.add_argument("--min-edge-length-m", type=float, default=18.0)
        g.add_argument("--min-edge-support", type=float, default=4.5)
        g.add_argument("--allow-opposite-direction-merge", action=argparse.BooleanOptionalAction, default=False)
        g.add_argument("--temporal-slice-day", type=int, default=None)
        g.add_argument("--temporal-slice-hour", type=int, default=None)

    def configure(self, args) -> None:
        d: dict = {}
        for f in fields(RoadsterConfig):
            if hasattr(args, f.name):
                val = getattr(args, f.name)
                if val is not None:
                    d[f.name] = val
        self.config = RoadsterConfig.from_dict(d)

    def _point_weight(self, seg_row) -> float:
        cfg = self.config
        source = str(seg_row.source).upper()
        w = cfg.vpd_base_weight if source == "VPD" else cfg.hpd_base_weight
        if pd.notnull(seg_row.path_quality_score):
            w += cfg.quality_weight_boost * float(np.clip(seg_row.path_quality_score, 0.0, 1.0))
        if pd.notnull(seg_row.sensor_quality_score):
            w += 0.5 * cfg.quality_weight_boost * float(np.clip(seg_row.sensor_quality_score, 0.0, 1.0))

        construction = float(seg_row.construction_percent or 0.0)
        w *= max(0.15, 1.0 - cfg.construction_penalty * (construction / 100.0))

        if cfg.temporal_slice_day is not None and seg_row.day is not None and int(seg_row.day) != int(cfg.temporal_slice_day):
            w *= (1.0 - cfg.temporal_mismatch_penalty)
        if cfg.temporal_slice_hour is not None and seg_row.hour is not None and int(seg_row.hour) != int(cfg.temporal_slice_hour):
            w *= (1.0 - cfg.temporal_mismatch_penalty)
        return float(max(w, 0.05))

    def _extract_subtrajectories(self, segments: pd.DataFrame) -> List[dict]:
        cfg = self.config
        subs: List[dict] = []
        sid = 0

        for row in segments.itertuples(index=False):
            coords = np.asarray(row.coords_xy, dtype=np.float64)
            if len(coords) < 2:
                continue

            spacing = max(float(row.length_m) / max(len(coords) - 1, 1), 0.5)
            win_pts = max(4, int(round(cfg.subtraj_window_m / spacing)))
            step_pts = max(2, int(round(cfg.subtraj_step_m / spacing)))
            if len(coords) < win_pts:
                win_pts = len(coords)
                step_pts = len(coords)

            starts = list(range(0, max(len(coords) - win_pts + 1, 1), step_pts))
            if not starts:
                starts = [0]
            last_start = max(0, len(coords) - win_pts)
            if starts[-1] != last_start:
                starts.append(last_start)
            if len(starts) > cfg.max_windows_per_segment:
                starts = starts[: cfg.max_windows_per_segment]

            alt = np.asarray(row.altitudes, dtype=np.float64)
            if len(alt) != len(coords):
                alt = np.full(len(coords), np.nan, dtype=np.float64)

            pweight = self._point_weight(row)
            crosswalk_count = len(row.crosswalk_types) if isinstance(row.crosswalk_types, list) else 0

            for s in starts:
                e = min(s + win_pts, len(coords))
                if e - s < 4:
                    continue
                c = coords[s:e]
                rep = _resample_polyline(c, cfg.subtraj_resample_points)
                h = _line_heading(c)
                center = np.mean(rep, axis=0)
                a = float(np.nanmedian(alt[s:e])) if np.isfinite(alt[s:e]).any() else float("nan")
                subs.append(
                    {
                        "sub_id": sid,
                        "segment_id": int(row.segment_id),
                        "trace_id": str(row.trace_id),
                        "source": str(row.source),
                        "coords": c,
                        "rep": rep,
                        "center_x": float(center[0]),
                        "center_y": float(center[1]),
                        "heading": float(h),
                        "altitude": a,
                        "weight": float(pweight),
                        "construction_percent": float(row.construction_percent),
                        "traffic_signal_count": float(row.traffic_signal_count),
                        "crosswalk_count": int(crosswalk_count),
                        "day": row.day,
                        "hour": row.hour,
                    }
                )
                sid += 1
        return subs

    def _cluster_subtrajectories(self, subtrajs: List[dict]) -> List[dict]:
        cfg = self.config
        if not subtrajs:
            return []

        centers = np.column_stack(
            [
                np.asarray([s["center_x"] for s in subtrajs], dtype=np.float64),
                np.asarray([s["center_y"] for s in subtrajs], dtype=np.float64),
            ]
        )
        tree = cKDTree(centers)
        n = len(subtrajs)

        # Build adjacency graph over windows that satisfy similarity constraints.
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            cand = tree.query_ball_point(centers[i], r=cfg.cluster_center_radius_m)
            si = subtrajs[i]
            for j in cand:
                if j <= i:
                    continue
                sj = subtrajs[j]

                hd = angle_diff_deg(si["heading"], sj["heading"])
                opposite = hd >= (180.0 - cfg.opposite_direction_tolerance_deg)
                if opposite and not cfg.allow_opposite_direction_merge:
                    continue
                if hd > cfg.cluster_heading_tolerance_deg and not opposite:
                    continue

                if np.isfinite(si["altitude"]) and np.isfinite(sj["altitude"]):
                    if abs(float(si["altitude"]) - float(sj["altitude"])) > cfg.cluster_altitude_eps_m:
                        continue

                d_f = _discrete_frechet(
                    np.asarray(si["rep"], dtype=np.float64),
                    np.asarray(sj["rep"], dtype=np.float64),
                )
                if d_f > cfg.cluster_frechet_eps_m:
                    continue

                adj[i].append(j)
                adj[j].append(i)

        visited = np.zeros(n, dtype=bool)
        clusters: List[dict] = []
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

            if len(comp) < cfg.cluster_min_members:
                continue

            members = [subtrajs[idx] for idx in comp]
            weighted_support = float(np.sum([m["weight"] for m in members]))
            if weighted_support < cfg.cluster_min_weighted_support:
                continue

            clusters.append(
                {
                    "cluster_id": cid,
                    "member_idx": comp,
                    "members": members,
                    "weighted_support": weighted_support,
                    "member_count": int(len(members)),
                }
            )
            cid += 1

        return clusters

    def _refine_representative(self, cluster: dict) -> dict:
        cfg = self.config
        members = cluster["members"]
        if not members:
            return {}

        # Choose representative complexity from median member length.
        lengths = [LineString(m["coords"]).length for m in members]
        med_len = float(np.median(lengths)) if lengths else 0.0
        n_points = max(int(round(med_len / max(cfg.representative_spacing_m, 0.5))) + 1, 8)

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

        x = np.zeros(n_points, dtype=np.float64)
        y = np.zeros(n_points, dtype=np.float64)
        for i in range(n_points):
            x[i] = _weighted_median(reps_arr[:, i, 0], w)
            y[i] = _weighted_median(reps_arr[:, i, 1], w)
        rep = np.column_stack([x, y])

        turn_idx: set[int] = {0, n_points - 1}
        for i in range(1, n_points - 1):
            h1 = bearing_from_xy(rep[i - 1, 0], rep[i - 1, 1], rep[i, 0], rep[i, 1])
            h2 = bearing_from_xy(rep[i, 0], rep[i, 1], rep[i + 1, 0], rep[i + 1, 1])
            if angle_diff_deg(h1, h2) >= cfg.representative_turn_deg:
                turn_idx.add(i)

        rep = _smooth_polyline(rep, keep_idx=turn_idx, passes=cfg.representative_smooth_passes)
        rep_line = LineString([(float(xx), float(yy)) for xx, yy in rep])
        rep_heading = _line_heading(rep)

        return {
            "cluster_id": int(cluster["cluster_id"]),
            "member_count": int(cluster["member_count"]),
            "weighted_support": float(cluster["weighted_support"]),
            "line_xy": rep_line,
            "coords": rep,
            "heading": float(rep_heading),
            "turn_indices": sorted(turn_idx),
            "altitude_mean": float(np.nanmedian(np.asarray(altitudes, dtype=np.float64)))
            if np.isfinite(altitudes).any()
            else np.nan,
            "source_counts": Counter([m["source"] for m in members]),
            "construction_percent_mean": float(np.mean([m["construction_percent"] for m in members])),
            "traffic_signal_mean": float(np.mean([m["traffic_signal_count"] for m in members])),
            "crosswalk_mean": float(np.mean([m["crosswalk_count"] for m in members])),
            "day_mode": Counter([m["day"] for m in members if m["day"] is not None]).most_common(1)[0][0]
            if any(m["day"] is not None for m in members)
            else None,
            "hour_mode": Counter([m["hour"] for m in members if m["hour"] is not None]).most_common(1)[0][0]
            if any(m["hour"] is not None for m in members)
            else None,
        }

    def _infer_vertices(self, reps: List[dict]) -> pd.DataFrame:
        cfg = self.config
        if not reps:
            return pd.DataFrame(columns=["node_id", "x", "y", "z", "support"])

        candidates = []
        for rep in reps:
            coords = np.asarray(rep["coords"], dtype=np.float64)
            if len(coords) < 2:
                continue
            w = float(rep["weighted_support"])
            z = rep["altitude_mean"]

            # Endpoints
            h_start = bearing_from_xy(coords[0, 0], coords[0, 1], coords[min(1, len(coords) - 1), 0], coords[min(1, len(coords) - 1), 1])
            h_end = bearing_from_xy(coords[max(0, len(coords) - 2), 0], coords[max(0, len(coords) - 2), 1], coords[-1, 0], coords[-1, 1])
            candidates.append({"x": coords[0, 0], "y": coords[0, 1], "z": z, "heading": h_start, "weight": w, "kind": "endpoint"})
            candidates.append({"x": coords[-1, 0], "y": coords[-1, 1], "z": z, "heading": h_end, "weight": w, "kind": "endpoint"})

            # Turns
            for i in rep["turn_indices"]:
                if i <= 0 or i >= len(coords) - 1:
                    continue
                h = bearing_from_xy(coords[i - 1, 0], coords[i - 1, 1], coords[i + 1, 0], coords[i + 1, 1])
                turn_weight = w * (1.0 + 0.1 * rep["traffic_signal_mean"] + 0.05 * rep["crosswalk_mean"])
                candidates.append({"x": coords[i, 0], "y": coords[i, 1], "z": z, "heading": h, "weight": turn_weight, "kind": "turn"})

        # Candidate intersections from line crossings.
        for i in range(len(reps)):
            li = reps[i]["line_xy"]
            for j in range(i + 1, len(reps)):
                lj = reps[j]["line_xy"]
                if li.distance(lj) > cfg.vertex_snap_m:
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
                hd = angle_diff_deg(reps[i]["heading"], reps[j]["heading"])
                if hd < 12.0:
                    continue
                w = min(reps[i]["weighted_support"], reps[j]["weighted_support"]) * max(math.sin(math.radians(hd)), 0.2)
                zvals = [reps[i]["altitude_mean"], reps[j]["altitude_mean"]]
                zvals = [z for z in zvals if np.isfinite(z)]
                z = float(np.mean(zvals)) if zvals else np.nan
                candidates.append({"x": float(p.x), "y": float(p.y), "z": z, "heading": float((reps[i]["heading"] + reps[j]["heading"]) / 2.0), "weight": w, "kind": "cross"})

        if not candidates:
            return pd.DataFrame(columns=["node_id", "x", "y", "z", "support"])

        # Merge candidates into node groups by distance + altitude compatibility.
        groups: List[List[dict]] = []
        for c in sorted(candidates, key=lambda d: d["weight"], reverse=True):
            assigned = False
            for g in groups:
                gx = float(np.mean([u["x"] for u in g]))
                gy = float(np.mean([u["y"] for u in g]))
                dist = float(np.hypot(c["x"] - gx, c["y"] - gy))
                if dist > cfg.vertex_snap_m:
                    continue
                zc = c["z"]
                gz_vals = [u["z"] for u in g if np.isfinite(u["z"])]
                if np.isfinite(zc) and gz_vals:
                    if abs(zc - float(np.mean(gz_vals))) > cfg.vertex_altitude_eps_m:
                        continue
                g.append(c)
                assigned = True
                break
            if not assigned:
                groups.append([c])

        rows = []
        for nid, g in enumerate(groups):
            wx = np.asarray([u["weight"] for u in g], dtype=np.float64)
            xs = np.asarray([u["x"] for u in g], dtype=np.float64)
            ys = np.asarray([u["y"] for u in g], dtype=np.float64)
            zvals = np.asarray([u["z"] for u in g], dtype=np.float64)

            # Optional line-intersection refinement from heading lines.
            pair_pts = []
            pair_w = []
            for i in range(len(g)):
                li = _line_from_point_heading(g[i]["x"], g[i]["y"], g[i]["heading"])
                for j in range(i + 1, len(g)):
                    hd = angle_diff_deg(g[i]["heading"], g[j]["heading"])
                    if hd < 15.0:
                        continue
                    lj = _line_from_point_heading(g[j]["x"], g[j]["y"], g[j]["heading"])
                    p = _intersection_point(li, lj)
                    if p is None:
                        continue
                    if float(np.hypot(p.x - np.mean(xs), p.y - np.mean(ys))) > 2.0 * cfg.vertex_snap_m:
                        continue
                    pair_pts.append((float(p.x), float(p.y)))
                    pair_w.append(float(g[i]["weight"] * g[j]["weight"] * max(math.sin(math.radians(hd)), 0.2)))

            if pair_pts:
                pw = np.asarray(pair_w, dtype=np.float64)
                px = np.asarray([p[0] for p in pair_pts], dtype=np.float64)
                py = np.asarray([p[1] for p in pair_pts], dtype=np.float64)
                x = float(np.average(px, weights=pw))
                y = float(np.average(py, weights=pw))
            else:
                x = float(np.average(xs, weights=wx))
                y = float(np.average(ys, weights=wx))

            zf = float(np.nanmedian(zvals)) if np.isfinite(zvals).any() else np.nan
            rows.append(
                {
                    "node_id": int(nid),
                    "x": x,
                    "y": y,
                    "z": zf,
                    "support": float(np.sum(wx)),
                    "candidate_count": int(len(g)),
                }
            )
        return pd.DataFrame(rows)

    def _build_edges(self, reps: List[dict], nodes: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if not reps or nodes.empty:
            return pd.DataFrame()

        node_rows = list(nodes.itertuples(index=False))

        edge_acc: Dict[tuple, dict] = {}

        for rep in reps:
            line: LineString = rep["line_xy"]
            coords = np.asarray(rep["coords"], dtype=np.float64)
            if line.length <= 0 or len(coords) < 2:
                continue

            # Find nearby nodes for this representative.
            hits = []
            for n in node_rows:
                if np.isfinite(n.z) and np.isfinite(rep["altitude_mean"]):
                    if abs(float(n.z) - float(rep["altitude_mean"])) > cfg.vertex_altitude_eps_m:
                        continue
                d = float(line.distance(Point(float(n.x), float(n.y))))
                if d <= cfg.edge_node_snap_m:
                    proj = float(line.project(Point(float(n.x), float(n.y))))
                    hits.append((proj, int(n.node_id)))

            if len(hits) < 2:
                continue
            hits.sort(key=lambda x: x[0])

            # Unique ordered nodes by projected position.
            ordered = []
            seen = set()
            for proj, nid in hits:
                if nid in seen:
                    continue
                ordered.append((proj, nid))
                seen.add(nid)

            if len(ordered) < 2:
                continue

            for i in range(1, len(ordered)):
                d0, a = ordered[i - 1]
                d1, b = ordered[i]
                if a == b:
                    continue
                if d1 - d0 < cfg.min_edge_length_m:
                    continue
                seg = substring(line, d0, d1)
                if seg.is_empty or not isinstance(seg, LineString):
                    continue
                seg_len = float(seg.length)
                if seg_len < cfg.min_edge_length_m:
                    continue

                u, v = (a, b) if a < b else (b, a)
                alt_band = (
                    int(round(float(rep["altitude_mean"]) / max(cfg.altitude_band_m, 0.1)))
                    if np.isfinite(rep["altitude_mean"])
                    else 0
                )
                key = (u, v, alt_band)
                if key not in edge_acc:
                    edge_acc[key] = {
                        "u": u,
                        "v": v,
                        "altitude_band": alt_band,
                        "support_k": 0.0,
                        "weighted_support": 0.0,
                        "dir_support": Counter(),
                        "ref_dir": (a, b),
                        "vpd_support": 0.0,
                        "hpd_support": 0.0,
                        "construction_wsum": 0.0,
                        "traffic_signal_wsum": 0.0,
                        "crosswalk_wsum": 0.0,
                        "attr_weight_sum": 0.0,
                        "day_counter": Counter(),
                        "hour_counter": Counter(),
                        "length_m": seg_len,
                        "geometry_xy": seg,
                    }
                s = edge_acc[key]
                s["support_k"] += float(rep["member_count"])
                s["weighted_support"] += float(rep["weighted_support"])

                rep_weight = float(rep["weighted_support"])
                s["dir_support"][(int(a), int(b))] += rep_weight

                s["vpd_support"] += float(rep["source_counts"].get("VPD", 0))
                s["hpd_support"] += float(rep["source_counts"].get("HPD", 0))
                s["construction_wsum"] += float(rep["construction_percent_mean"]) * rep_weight
                s["traffic_signal_wsum"] += float(rep["traffic_signal_mean"]) * rep_weight
                s["crosswalk_wsum"] += float(rep["crosswalk_mean"]) * rep_weight
                s["attr_weight_sum"] += rep_weight
                if rep["day_mode"] is not None:
                    s["day_counter"][int(rep["day_mode"])] += 1
                if rep["hour_mode"] is not None:
                    s["hour_counter"][int(rep["hour_mode"])] += 1

                # Keep higher-support geometry if keys collide.
                if float(rep["weighted_support"]) > s.get("_best_geom_weight", -1.0):
                    s["_best_geom_weight"] = float(rep["weighted_support"])
                    s["geometry_xy"] = seg
                    s["length_m"] = seg_len
                    s["ref_dir"] = (int(a), int(b))

        if not edge_acc:
            return pd.DataFrame()

        rows = []
        for s in edge_acc.values():
            if s["weighted_support"] < cfg.min_edge_support:
                continue
            ref_a, ref_b = s.get("ref_dir", (int(s["u"]), int(s["v"])))
            fw = float(s["dir_support"].get((int(ref_a), int(ref_b)), 0.0))
            rv = float(s["dir_support"].get((int(ref_b), int(ref_a)), 0.0))
            tot = fw + rv
            if tot <= 0:
                continue
            one_way_conf = abs(fw - rv) / tot
            if min(fw, rv) / max(tot, 1e-6) >= 0.2:
                dir_travel = "B"
            elif fw >= rv:
                dir_travel = "T"
            else:
                dir_travel = "F"

            attr_weight = float(s.get("attr_weight_sum", 0.0))
            if attr_weight > 0.0:
                construction_mean = float(s["construction_wsum"] / attr_weight)
                traffic_signal_mean = float(s["traffic_signal_wsum"] / attr_weight)
                crosswalk_mean = float(s["crosswalk_wsum"] / attr_weight)
            else:
                construction_mean = float("nan")
                traffic_signal_mean = float("nan")
                crosswalk_mean = float("nan")

            rows.append(
                {
                    "u": int(ref_a),
                    "v": int(ref_b),
                    "support_k": float(s["support_k"]),
                    "weighted_support": float(s["weighted_support"]),
                    "length_m": float(s["length_m"]),
                    "vpd_support": float(s["vpd_support"]),
                    "hpd_support": float(s["hpd_support"]),
                    "construction_percent_mean": construction_mean,
                    "traffic_signal_count_mean": traffic_signal_mean,
                    "crosswalk_count_mean": crosswalk_mean,
                    "day_mode": s["day_counter"].most_common(1)[0][0] if s["day_counter"] else None,
                    "hour_mode": s["hour_counter"].most_common(1)[0][0] if s["hour_counter"] else None,
                    "altitude_band": int(s["altitude_band"]),
                    "dir_travel": dir_travel,
                    "one_way_confidence": float(one_way_conf),
                    "confidence": float(min(1.0, s["weighted_support"] / max(cfg.cluster_min_weighted_support * 2.0, 1e-6))),
                    "geometry_xy": s["geometry_xy"],
                }
            )

        if not rows:
            return pd.DataFrame()
        edges = pd.DataFrame(rows)

        # Remove weak short dangling edges.
        deg = Counter()
        for r in edges.itertuples(index=False):
            deg[int(r.u)] += 1
            deg[int(r.v)] += 1

        keep = []
        for r in edges.itertuples(index=False):
            is_dangle = deg[int(r.u)] == 1 or deg[int(r.v)] == 1
            if is_dangle and float(r.length_m) < cfg.dangling_length_m and float(r.weighted_support) < cfg.dangling_min_support:
                keep.append(False)
                continue
            keep.append(True)
        edges = edges.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)
        return edges

    def generate(
            self,
            traces: pd.DataFrame,
            projected_crs: CRS,
            to_proj: Transformer,
            to_wgs: Transformer,
    ) -> dict:
        cfg = self.config
        source_cfgs: Dict[str, SourcePreprocessConfig] = {
            "VPD": default_source_preprocess_config("VPD"),
            "HPD": default_source_preprocess_config("HPD"),
        }
        segments = preprocess_traces_dataframe(
            traces=traces,
            to_proj=to_proj,
            to_wgs=to_wgs,
            source_cfgs=source_cfgs,
        )
        if segments.empty:
            return {
                "projected_crs": projected_crs,
                "nodes": pd.DataFrame(),
                "edges": pd.DataFrame(),
                "centerlines": pd.DataFrame(),
                "trace_count": int(len(traces)),
                "sample_point_count": 0,
            }

        subtrajs = self._extract_subtrajectories(segments)
        clusters = self._cluster_subtrajectories(subtrajs)
        reps = [self._refine_representative(c) for c in clusters]
        reps = [r for r in reps if r and r["line_xy"].length >= cfg.min_centerline_length_m]

        nodes = self._infer_vertices(reps)
        edges_xy = self._build_edges(reps, nodes)

        if nodes.empty or edges_xy.empty:
            return {
                "projected_crs": projected_crs,
                "nodes": pd.DataFrame(),
                "edges": pd.DataFrame(),
                "centerlines": pd.DataFrame(),
                "trace_count": int(traces["trace_id"].nunique() if "trace_id" in traces.columns else len(traces)),
                "sample_point_count": int(sum(len(np.asarray(v)) for v in segments["coords_xy"])),
            }

        # Project nodes/edges back to WGS84.
        lon, lat = to_wgs.transform(nodes["x"].to_numpy(), nodes["y"].to_numpy())
        nodes = nodes.copy()
        nodes["lon"] = lon
        nodes["lat"] = lat

        edges = edges_xy.copy()
        edges["geometry"] = edges["geometry_xy"].map(lambda g: transform(to_wgs.transform, g))
        edges = edges.drop(columns=["geometry_xy"])
        edges["edge_id"] = np.arange(len(edges), dtype=np.int64)

        # Centerlines as edge-level output (graph edge polylines).
        centerlines = edges.copy()
        centerlines["node_path"] = centerlines.apply(lambda r: [int(r["u"]), int(r["v"])], axis=1)
        centerlines = centerlines.rename(columns={"support_k": "support"})

        return {
            "projected_crs": projected_crs,
            "nodes": nodes,
            "edges": edges,
            "centerlines": centerlines,
            "trace_count": int(traces["trace_id"].nunique() if "trace_id" in traces.columns else len(traces)),
            "sample_point_count": int(sum(len(np.asarray(v)) for v in segments["coords_xy"])),
        }
