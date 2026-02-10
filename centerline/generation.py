from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import transform

from .io_utils import infer_local_projected_crs, load_hpd_traces, load_vpd_traces


@dataclass
class CenterlineConfig:
    # Kharita-style heading-aware clustering
    cluster_radius_m: float = 10.0
    heading_tolerance_deg: float = 45.0
    heading_distance_weight_m: float = 0.22
    min_cluster_points: int = 1

    # Trace sampling and edge extraction
    sample_spacing_m: float = 8.0
    max_points_per_trace: int = 120
    max_transition_distance_m: float = 50.0

    # Edge pruning
    min_edge_support: float = 2.0
    reverse_edge_ratio: float = 0.2
    transitive_max_hops: int = 4
    transitive_ratio: float = 1.03
    transitive_max_checks: int = 25000

    # Centerline smoothing/cleanup
    smooth_iterations: int = 2
    min_centerline_length_m: float = 12.0

    # Trace weighting (use richer VPD quality instead of degrading to legacy format)
    vpd_base_weight: float = 1.2
    hpd_base_weight: float = 1.0


def _angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    # 0 degrees = north, clockwise positive.
    return (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 360.0) % 360.0


def _sample_line_projected(
    line_xy: LineString, sample_spacing_m: float, max_points: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if line_xy.length == 0:
        arr = np.array([], dtype=np.float32)
        return arr, arr.reshape(0, 2), arr

    effective_step = max(sample_spacing_m, line_xy.length / max(max_points, 1))
    n_segments = max(int(math.ceil(line_xy.length / effective_step)), 1)
    dists = np.linspace(0.0, float(line_xy.length), n_segments + 1, dtype=np.float32)

    coords = np.zeros((len(dists), 2), dtype=np.float32)
    for i, d in enumerate(dists):
        p = line_xy.interpolate(float(d))
        coords[i, 0] = float(p.x)
        coords[i, 1] = float(p.y)

    headings = np.zeros(len(dists), dtype=np.float32)
    for i in range(len(dists)):
        i0 = max(i - 1, 0)
        i1 = min(i + 1, len(dists) - 1)
        if i0 == i1:
            headings[i] = 0.0
        else:
            headings[i] = _bearing_from_xy(
                coords[i0, 0], coords[i0, 1], coords[i1, 0], coords[i1, 1]
            )
    return dists, coords, headings


def _interpolate_altitudes(
    altitudes: List[float],
    line_xy: LineString,
    sample_dists: np.ndarray,
) -> np.ndarray:
    if not altitudes:
        return np.full(len(sample_dists), np.nan, dtype=np.float32)

    cleaned = []
    for a in altitudes:
        try:
            cleaned.append(float(a))
        except Exception:
            continue
    if len(cleaned) < 2:
        return np.full(len(sample_dists), np.nan, dtype=np.float32)

    anchor_dists = np.linspace(
        0.0, float(line_xy.length), len(cleaned), dtype=np.float32
    )
    return np.interp(
        sample_dists, anchor_dists, np.asarray(cleaned, dtype=np.float32)
    ).astype(np.float32)


def _kharita_style_incremental_clustering(
    xs: np.ndarray,
    ys: np.ndarray,
    headings: np.ndarray,
    point_weights: np.ndarray,
    config: CenterlineConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Adapted from Kharita/Kharita*: heading-aware radius clustering.

    Adaptation:
    - Uses metric projection for robust distance in meters.
    - Uses incremental weighted centroid updates to preserve high-quality VPD detail.
    - Keeps heading tolerance as a first-class condition.
    """
    n = len(xs)
    if n == 0:
        return np.array([], dtype=np.int32), pd.DataFrame(
            columns=["node_id", "x", "y", "heading", "weight", "point_count"]
        )

    labels = np.full(n, -1, dtype=np.int32)

    cx: List[float] = []
    cy: List[float] = []
    cweight: List[float] = []
    csin: List[float] = []
    ccos: List[float] = []
    ccount: List[int] = []

    tree = None
    dirty_count = 0  # tracks both new clusters and centroid updates

    def rebuild_tree() -> None:
        nonlocal tree
        if cx:
            arr = np.column_stack(
                [np.asarray(cx, dtype=np.float32), np.asarray(cy, dtype=np.float32)]
            )
            tree = cKDTree(arr)

    for i in range(n):
        x = float(xs[i])
        y = float(ys[i])
        heading = float(headings[i])
        w = float(max(point_weights[i], 1e-6))

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

        if tree is None or dirty_count >= 2000 or i % 20000 == 0:
            rebuild_tree()
            dirty_count = 0

        candidate_ids = (
            tree.query_ball_point([x, y], r=config.cluster_radius_m)
            if tree is not None
            else []
        )

        best_cid = -1
        best_score = float("inf")
        for cid in candidate_ids:
            cheading = (math.degrees(math.atan2(csin[cid], ccos[cid])) + 360.0) % 360.0
            ad = _angle_diff_deg(heading, cheading)
            if ad > config.heading_tolerance_deg:
                continue
            sd = math.hypot(x - cx[cid], y - cy[cid])
            score = sd + config.heading_distance_weight_m * ad
            if score < best_score:
                best_score = score
                best_cid = cid

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

        # Weighted incremental update of centroid and heading components.
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

    # Optionally collapse tiny clusters into nearby larger clusters.
    if config.min_cluster_points > 1 and cx:
        major_ids = [
            i for i, cnt in enumerate(ccount) if cnt >= config.min_cluster_points
        ]
        if major_ids:
            major_xy = np.column_stack(
                [
                    np.asarray([cx[i] for i in major_ids]),
                    np.asarray([cy[i] for i in major_ids]),
                ]
            )
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

    # Reindex labels to compact IDs and recompute node stats from assigned points.
    unique = sorted(set(int(v) for v in labels))
    to_new = {old: new for new, old in enumerate(unique)}
    labels = np.asarray([to_new[int(v)] for v in labels], dtype=np.int32)

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

    rows = []
    for nid in sorted(node_stats):
        s = node_stats[nid]
        ww = s["w"] if s["w"] > 0 else 1.0
        rows.append(
            {
                "node_id": nid,
                "x": s["xw"] / ww,
                "y": s["yw"] / ww,
                "heading": (math.degrees(math.atan2(s["sinw"], s["cosw"])) + 360.0)
                % 360.0,
                "weight": s["w"],
                "point_count": s["n"],
            }
        )
    return labels, pd.DataFrame(rows)


def _chaikin(coords: np.ndarray, iterations: int) -> np.ndarray:
    if len(coords) < 3 or iterations <= 0:
        return coords

    out = coords.copy()
    for _ in range(iterations):
        if len(out) < 3:
            break
        next_coords = [out[0]]
        for i in range(len(out) - 1):
            p = out[i]
            q = out[i + 1]
            q1 = 0.75 * p + 0.25 * q
            q2 = 0.25 * p + 0.75 * q
            next_coords.append(q1)
            next_coords.append(q2)
        next_coords.append(out[-1])
        out = np.asarray(next_coords, dtype=np.float64)
    return out


def _shortest_alternative_with_hop_limit(
    graph: Dict[int, List[Tuple[int, float]]],
    source: int,
    target: int,
    skip_edge: Tuple[int, int],
    max_hops: int,
    max_dist: float,
) -> float:
    # Dijkstra-like search constrained by hop count.
    best = {(source, 0): 0.0}
    frontier: list[tuple[float, int, int]] = [(0.0, source, 0)]

    while frontier:
        dist, node, hops = heapq.heappop(frontier)
        if dist > max_dist:
            continue
        if node == target and hops > 0:
            return dist
        if hops >= max_hops:
            continue

        for nbr, w in graph.get(node, []):
            if (node, nbr) == skip_edge:
                continue
            nd = dist + w
            key = (nbr, hops + 1)
            if nd < best.get(key, float("inf")):
                best[key] = nd
                heapq.heappush(frontier, (nd, nbr, hops + 1))

    return float("inf")


def _stitch_centerline_paths(
    edge_support: Dict[Tuple[int, int], dict],
) -> List[List[int]]:
    undirected_adj: Dict[int, set] = defaultdict(set)
    undirected_edges = set()
    for u, v in edge_support:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        undirected_edges.add((a, b))
        undirected_adj[u].add(v)
        undirected_adj[v].add(u)

    visited = set()
    paths: List[List[int]] = []

    def walk(start: int, nxt: int) -> List[int]:
        path = [start, nxt]
        visited.add((min(start, nxt), max(start, nxt)))
        prev, cur = start, nxt
        while len(undirected_adj[cur]) == 2:
            candidates = [z for z in undirected_adj[cur] if z != prev]
            if not candidates:
                break
            nn = candidates[0]
            ek = (min(cur, nn), max(cur, nn))
            if ek in visited:
                break
            path.append(nn)
            visited.add(ek)
            prev, cur = cur, nn
        return path

    endpoints = [n for n, neigh in undirected_adj.items() if len(neigh) != 2]
    for s in endpoints:
        for n in list(undirected_adj[s]):
            e = (min(s, n), max(s, n))
            if e in visited:
                continue
            paths.append(walk(s, n))

    # Remaining loops.
    for a, b in list(undirected_edges):
        if (a, b) in visited:
            continue
        loop = walk(a, b)
        paths.append(loop)

    return paths


def generate_centerlines(
    vpd_csv: str | Path,
    hpd_csvs: Iterable[str | Path],
    config: CenterlineConfig | None = None,
    fused_only: bool = True,
    max_vpd_rows: int | None = None,
    max_hpd_rows_per_file: int | None = None,
) -> dict:
    config = config or CenterlineConfig()

    vpd_df = load_vpd_traces(vpd_csv, fused_only=fused_only, max_rows=max_vpd_rows)
    hpd_df = load_hpd_traces(hpd_csvs, max_rows_per_file=max_hpd_rows_per_file)
    traces = pd.concat([vpd_df, hpd_df], ignore_index=True)
    traces = traces[traces["geometry"].notnull()].reset_index(drop=True)

    if traces.empty:
        return {
            "projected_crs": None,
            "nodes": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "centerlines": pd.DataFrame(),
            "trace_count": 0,
            "sample_point_count": 0,
        }

    projected_crs: CRS = infer_local_projected_crs(list(traces["geometry"]))
    to_proj = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    to_wgs = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)

    # Trace metadata and sampled points containers.
    trace_meta: Dict[int, dict] = {}
    trace_ranges: Dict[int, Tuple[int, int]] = {}

    xs: List[float] = []
    ys: List[float] = []
    headings: List[float] = []
    point_weights: List[float] = []
    altitudes: List[float] = []
    trace_index_for_point: List[int] = []

    next_trace_id = 0

    for _, row in traces.iterrows():
        line_wgs = row["geometry"]
        line_xy = transform(to_proj.transform, line_wgs)
        if line_xy.is_empty or line_xy.length <= 1.0:
            continue

        sample_dists, sample_xy, sample_headings = _sample_line_projected(
            line_xy=line_xy,
            sample_spacing_m=config.sample_spacing_m,
            max_points=config.max_points_per_trace,
        )
        if len(sample_xy) < 2:
            continue

        alts = _interpolate_altitudes(row.get("altitudes", []), line_xy, sample_dists)

        source = str(row.get("source", "VPD"))
        base_w = (
            config.vpd_base_weight
            if source.upper() == "VPD"
            else config.hpd_base_weight
        )
        path_q = row.get("path_quality_score", np.nan)
        sensor_q = row.get("sensor_quality_score", np.nan)
        if pd.notnull(path_q):
            base_w += 0.25 * float(np.clip(path_q, 0.0, 1.0))
        if pd.notnull(sensor_q):
            base_w += 0.15 * float(np.clip(sensor_q, 0.0, 1.0))

        start = len(xs)
        for i in range(len(sample_xy)):
            xs.append(float(sample_xy[i, 0]))
            ys.append(float(sample_xy[i, 1]))
            headings.append(float(sample_headings[i]))
            point_weights.append(base_w)
            altitudes.append(float(alts[i]) if i < len(alts) else float("nan"))
            trace_index_for_point.append(next_trace_id)
        end = len(xs)

        trace_ranges[next_trace_id] = (start, end)
        trace_meta[next_trace_id] = {
            "trace_id": row.get("trace_id"),
            "source": source,
            "construction_percent": float(row.get("construction_percent", 0.0) or 0.0),
            "traffic_signal_count": float(row.get("traffic_signal_count", 0.0) or 0.0),
            "crosswalk_types": row.get("crosswalk_types", [])
            if isinstance(row.get("crosswalk_types", []), list)
            else [],
            "day": int(row["day"]) if pd.notnull(row.get("day", np.nan)) else None,
            "hour": int(row["hour"]) if pd.notnull(row.get("hour", np.nan)) else None,
        }
        next_trace_id += 1

    if not xs:
        return {
            "projected_crs": projected_crs,
            "nodes": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "centerlines": pd.DataFrame(),
            "trace_count": 0,
            "sample_point_count": 0,
        }

    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    heading_arr = np.asarray(headings, dtype=np.float32)
    weight_arr = np.asarray(point_weights, dtype=np.float32)
    altitude_arr = np.asarray(altitudes, dtype=np.float32)

    labels, nodes = _kharita_style_incremental_clustering(
        xs=x_arr,
        ys=y_arr,
        headings=heading_arr,
        point_weights=weight_arr,
        config=config,
    )

    # Build directed co-occurrence edges from consecutive sampled points in each trace.
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
            step_dist = float(
                math.hypot(x_arr[i] - x_arr[i - 1], y_arr[i] - y_arr[i - 1])
            )
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
                    "altitude_sum": 0.0,
                    "altitude_count": 0,
                    "crosswalk_types": set(),
                    "day_counter": Counter(),
                    "hour_counter": Counter(),
                }

            s = edge_support[key]
            s["support"] += 1.0
            w = float(weight_arr[i])
            s["weighted_support"] += w
            s["length_sum"] += step_dist

            if str(meta["source"]).upper() == "VPD":
                s["vpd_support"] += 1.0
            else:
                s["hpd_support"] += 1.0

            s["construction_sum"] += float(meta["construction_percent"])
            s["traffic_signal_sum"] += float(meta["traffic_signal_count"])

            a0 = altitude_arr[i - 1]
            a1 = altitude_arr[i]
            if np.isfinite(a0) and np.isfinite(a1):
                s["altitude_sum"] += 0.5 * (float(a0) + float(a1))
                s["altitude_count"] += 1

            s["crosswalk_types"].update(meta.get("crosswalk_types", []))
            if meta.get("day") is not None:
                s["day_counter"][int(meta["day"])] += 1
            if meta.get("hour") is not None:
                s["hour_counter"][int(meta["hour"])] += 1

    # Initial pruning by support.
    edge_support = {
        k: v for k, v in edge_support.items() if v["support"] >= config.min_edge_support
    }

    # Direction conflict pruning (keep both only when both directions are reasonably supported).
    to_drop = set()
    for (u, v), sv in list(edge_support.items()):
        if (v, u) not in edge_support:
            continue
        sr = edge_support[(v, u)]
        if sv["support"] < sr["support"] * config.reverse_edge_ratio:
            to_drop.add((u, v))
    for k in to_drop:
        edge_support.pop(k, None)

    # Transitive pruning similar in spirit to Kharita's graph pruning.
    node_xy = {
        int(r.node_id): (float(r.x), float(r.y)) for r in nodes.itertuples(index=False)
    }
    edge_lengths = {
        (u, v): float(
            math.hypot(node_xy[u][0] - node_xy[v][0], node_xy[u][1] - node_xy[v][1])
        )
        for (u, v) in edge_support
    }

    graph = defaultdict(list)
    for (u, v), dist in edge_lengths.items():
        graph[u].append((v, dist))

    candidates = sorted(
        edge_support.keys(),
        key=lambda e: (edge_support[e]["support"], -edge_lengths[e]),
    )[: config.transitive_max_checks]

    dropped_transitive = set()
    for e in candidates:
        if e not in edge_support:
            continue
        direct = edge_lengths[e]
        if direct <= 1.0:
            continue
        alt = _shortest_alternative_with_hop_limit(
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

    # Build output node dataframe (WGS84).
    if nodes.empty:
        return {
            "projected_crs": projected_crs,
            "nodes": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "centerlines": pd.DataFrame(),
            "trace_count": len(trace_ranges),
            "sample_point_count": len(x_arr),
        }

    lon, lat = to_wgs.transform(nodes["x"].to_numpy(), nodes["y"].to_numpy())
    nodes = nodes.copy()
    nodes["lon"] = lon
    nodes["lat"] = lat

    # Build directed edge dataframe.
    edge_rows = []
    for (u, v), s in edge_support.items():
        x1, y1 = node_xy[u]
        x2, y2 = node_xy[v]
        line_xy = LineString([(x1, y1), (x2, y2)])
        line_wgs = transform(to_wgs.transform, line_xy)

        support = max(float(s["support"]), 1.0)
        has_rev = (v, u) in edge_support
        dir_travel = "B" if has_rev else "T"

        day_mode = s["day_counter"].most_common(1)[0][0] if s["day_counter"] else None
        hour_mode = (
            s["hour_counter"].most_common(1)[0][0] if s["hour_counter"] else None
        )

        edge_rows.append(
            {
                "u": u,
                "v": v,
                "support": float(s["support"]),
                "weighted_support": float(s["weighted_support"]),
                "vpd_support": float(s["vpd_support"]),
                "hpd_support": float(s["hpd_support"]),
                "mean_step_length_m": float(s["length_sum"] / support),
                "construction_percent_mean": float(s["construction_sum"] / support),
                "traffic_signal_count_mean": float(s["traffic_signal_sum"] / support),
                "altitude_mean": float(s["altitude_sum"] / s["altitude_count"])
                if s["altitude_count"] > 0
                else np.nan,
                "crosswalk_types": sorted(s["crosswalk_types"]),
                "day_mode": day_mode,
                "hour_mode": hour_mode,
                "dir_travel": dir_travel,
                "geometry": line_wgs,
            }
        )

    edges_df = pd.DataFrame(edge_rows)

    # Stitch to continuous centerlines and smooth.
    centerline_paths = _stitch_centerline_paths(edge_support)
    centerline_rows = []

    for path_nodes in centerline_paths:
        if len(path_nodes) < 2:
            continue
        raw_xy = np.asarray([node_xy[n] for n in path_nodes], dtype=np.float64)
        smooth_xy = _chaikin(raw_xy, iterations=config.smooth_iterations)

        line_xy = LineString([(float(x), float(y)) for x, y in smooth_xy])
        if line_xy.length < config.min_centerline_length_m:
            continue
        line_wgs = transform(to_wgs.transform, line_xy)

        fw = 0.0
        rv = 0.0
        support_sum = 0.0
        weighted_support_sum = 0.0
        construction_sum = 0.0
        signal_sum = 0.0
        alt_sum = 0.0
        alt_count = 0
        crosswalk = set()
        day_counter = Counter()
        hour_counter = Counter()

        for i in range(1, len(path_nodes)):
            a = path_nodes[i - 1]
            b = path_nodes[i]
            if (a, b) in edge_support:
                s = edge_support[(a, b)]
                fw += s["support"]
                support_sum += s["support"]
                weighted_support_sum += s["weighted_support"]
                construction_sum += s["construction_sum"]
                signal_sum += s["traffic_signal_sum"]
                alt_sum += s["altitude_sum"]
                alt_count += s["altitude_count"]
                crosswalk.update(s["crosswalk_types"])
                day_counter.update(s["day_counter"])
                hour_counter.update(s["hour_counter"])
            if (b, a) in edge_support:
                s = edge_support[(b, a)]
                rv += s["support"]
                support_sum += s["support"]
                weighted_support_sum += s["weighted_support"]
                construction_sum += s["construction_sum"]
                signal_sum += s["traffic_signal_sum"]
                alt_sum += s["altitude_sum"]
                alt_count += s["altitude_count"]
                crosswalk.update(s["crosswalk_types"])
                day_counter.update(s["day_counter"])
                hour_counter.update(s["hour_counter"])

        if support_sum <= 0:
            continue

        if fw > 0 and rv > 0:
            dir_travel = "B"
        elif fw >= rv:
            dir_travel = "T"
        else:
            dir_travel = "F"

        centerline_rows.append(
            {
                "node_path": path_nodes,
                "support": float(support_sum),
                "weighted_support": float(weighted_support_sum),
                "construction_percent_mean": float(construction_sum / support_sum),
                "traffic_signal_count_mean": float(signal_sum / support_sum),
                "altitude_mean": float(alt_sum / alt_count)
                if alt_count > 0
                else np.nan,
                "crosswalk_types": sorted(crosswalk),
                "day_mode": day_counter.most_common(1)[0][0] if day_counter else None,
                "hour_mode": hour_counter.most_common(1)[0][0]
                if hour_counter
                else None,
                "dir_travel": dir_travel,
                "geometry": line_wgs,
            }
        )

    centerlines_df = pd.DataFrame(centerline_rows)

    return {
        "projected_crs": projected_crs,
        "nodes": nodes,
        "edges": edges_df,
        "centerlines": centerlines_df,
        "trace_count": len(trace_ranges),
        "sample_point_count": len(x_arr),
    }


def save_centerline_outputs(
    result: dict, output_dir: str | Path, stem: str = "generated_centerlines"
) -> dict:
    import geopandas as gpd
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = result["nodes"].copy()
    edges = result["edges"].copy()
    centerlines = result["centerlines"].copy()

    files = {}

    def serialize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            if c == "geometry":
                continue
            if out[c].dtype == "object":
                out[c] = out[c].map(
                    lambda v: json.dumps(sorted(v) if isinstance(v, set) else v)
                    if isinstance(v, (list, dict, set))
                    else v
                )
        return out

    if not nodes.empty:
        nodes["geometry"] = gpd.points_from_xy(
            nodes["lon"], nodes["lat"], crs="EPSG:4326"
        )
        gnodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
        node_path = output_dir / f"{stem}_nodes.gpkg"
        gnodes.to_file(node_path, layer="nodes", driver="GPKG")
        files["nodes"] = str(node_path)

    if not edges.empty:
        edges_write = serialize_object_columns(edges)
        gedges = gpd.GeoDataFrame(edges_write, geometry="geometry", crs="EPSG:4326")
        edge_path = output_dir / f"{stem}_edges.gpkg"
        gedges.to_file(edge_path, layer="edges", driver="GPKG")
        files["edges"] = str(edge_path)

    if not centerlines.empty:
        center_write = serialize_object_columns(centerlines)
        gcenter = gpd.GeoDataFrame(center_write, geometry="geometry", crs="EPSG:4326")
        center_path = output_dir / f"{stem}.gpkg"
        gcenter.to_file(center_path, layer="centerlines", driver="GPKG")
        files["centerlines"] = str(center_path)

        csv_path = output_dir / f"{stem}.csv"
        out_csv = centerlines.copy()
        out_csv["geometry_wkt"] = out_csv["geometry"].astype(str)
        out_csv = out_csv.drop(columns=["geometry"])
        out_csv.to_csv(csv_path, index=False)
        files["centerlines_csv"] = str(csv_path)

    summary_path = output_dir / f"{stem}_summary.json"
    pd.Series(
        {
            "trace_count": int(result.get("trace_count", 0)),
            "sample_point_count": int(result.get("sample_point_count", 0)),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "centerline_count": int(len(centerlines)),
            "projected_crs": str(result.get("projected_crs")),
        }
    ).to_json(summary_path, indent=2)
    files["summary"] = str(summary_path)

    return files
