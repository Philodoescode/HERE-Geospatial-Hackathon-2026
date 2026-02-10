"""Kharita-inspired centerline generation algorithm.

This is the original algorithm extracted from the monolithic
``generation.py`` pipeline.  It performs:

1. Trace sampling (densify to evenly-spaced points with headings)
2. Heading-aware incremental clustering (Kharita-style)
3. Directed co-occurrence graph construction
4. Three-pass edge pruning (support, direction-conflict, transitive)
5. Centerline stitching and Chaikin smoothing
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import transform

from .base import AlgorithmConfig, BaseCenterlineAlgorithm
from ..utils import (
    angle_diff_deg,
    bearing_from_xy,
    chaikin,
    interpolate_altitudes,
    sample_line_projected,
    shortest_alternative_with_hop_limit,
    stitch_centerline_paths,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KharitaConfig(AlgorithmConfig):
    """All tuning knobs for the Kharita-inspired algorithm."""

    # Heading-aware clustering
    cluster_radius_m: float = 10.0
    heading_tolerance_deg: float = 45.0
    heading_distance_weight_m: float = 0.22
    min_cluster_points: int = 1

    # Edge extraction
    max_transition_distance_m: float = 50.0

    # Edge pruning
    min_edge_support: float = 2.0
    reverse_edge_ratio: float = 0.2
    transitive_max_hops: int = 4
    transitive_ratio: float = 1.03
    transitive_max_checks: int = 25000


# ---------------------------------------------------------------------------
# Clustering core
# ---------------------------------------------------------------------------


def _kharita_style_incremental_clustering(
        xs: np.ndarray,
        ys: np.ndarray,
        headings: np.ndarray,
        point_weights: np.ndarray,
        config: KharitaConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Heading-aware radius clustering adapted from Kharita/Kharita*."""
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
    dirty_count = 0

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
            ad = angle_diff_deg(heading, cheading)
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


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------


class KharitaAlgorithm(BaseCenterlineAlgorithm):
    """Kharita-inspired clustering + co-occurrence graph pipeline.

    This is the original algorithm that was previously hard-coded in
    ``centerline.generation.generate_centerlines()``, now refactored
    into the pluggable algorithm system.
    """

    name = "kharita"
    description = (
        "Kharita-inspired heading-aware clustering with co-occurrence graph "
        "construction, three-pass edge pruning, and Chaikin smoothing."
    )

    def __init__(self, config: KharitaConfig | None = None) -> None:
        self.config = config or KharitaConfig()

    # -- CLI integration ----------------------------------------------------

    def add_cli_args(self, parser) -> None:
        g = parser.add_argument_group("Kharita algorithm parameters")
        g.add_argument("--cluster-radius-m", type=float, default=10.0)
        g.add_argument("--heading-tolerance-deg", type=float, default=45.0)
        g.add_argument("--sample-spacing-m", type=float, default=8.0)
        g.add_argument("--max-points-per-trace", type=int, default=120)
        g.add_argument("--max-transition-distance-m", type=float, default=50.0)
        g.add_argument("--min-edge-support", type=float, default=2.0)
        g.add_argument("--min-centerline-length-m", type=float, default=12.0)
        g.add_argument("--smooth-iterations", type=int, default=2)

    def configure(self, args) -> None:
        d: dict = {}
        for f in fields(KharitaConfig):
            cli_name = f.name.replace("_", "-")
            attr_name = f.name
            if hasattr(args, attr_name):
                val = getattr(args, attr_name)
                if val is not None:
                    d[attr_name] = val
            elif hasattr(args, cli_name):
                val = getattr(args, cli_name)
                if val is not None:
                    d[attr_name] = val
        self.config = KharitaConfig.from_dict(d)

    # -- Core algorithm -----------------------------------------------------

    def generate(
            self,
            traces: pd.DataFrame,
            projected_crs: CRS,
            to_proj: Transformer,
            to_wgs: Transformer,
    ) -> dict:
        config = self.config

        # -- Trace sampling -------------------------------------------------
        trace_meta: Dict[int, dict] = {}
        trace_ranges: Dict[int, Tuple[int, int]] = {}

        xs: List[float] = []
        ys: List[float] = []
        headings_list: List[float] = []
        point_weights: List[float] = []
        altitudes_list: List[float] = []
        trace_index_for_point: List[int] = []

        next_trace_id = 0

        for _, row in traces.iterrows():
            line_wgs = row["geometry"]
            line_xy = transform(to_proj.transform, line_wgs)
            if line_xy.is_empty or line_xy.length <= 1.0:
                continue

            sample_dists, sample_xy, sample_headings = sample_line_projected(
                line_xy=line_xy,
                sample_spacing_m=config.sample_spacing_m,
                max_points=config.max_points_per_trace,
            )
            if len(sample_xy) < 2:
                continue

            alts = interpolate_altitudes(
                row.get("altitudes", []), line_xy, sample_dists
            )

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
                headings_list.append(float(sample_headings[i]))
                point_weights.append(base_w)
                altitudes_list.append(float(alts[i]) if i < len(alts) else float("nan"))
                trace_index_for_point.append(next_trace_id)
            end = len(xs)

            trace_ranges[next_trace_id] = (start, end)
            trace_meta[next_trace_id] = {
                "trace_id": row.get("trace_id"),
                "source": source,
                "construction_percent": float(
                    row.get("construction_percent", 0.0) or 0.0
                ),
                "traffic_signal_count": float(
                    row.get("traffic_signal_count", 0.0) or 0.0
                ),
                "crosswalk_types": row.get("crosswalk_types", [])
                if isinstance(row.get("crosswalk_types", []), list)
                else [],
                "day": int(row["day"]) if pd.notnull(row.get("day", np.nan)) else None,
                "hour": int(row["hour"])
                if pd.notnull(row.get("hour", np.nan))
                else None,
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
        heading_arr = np.asarray(headings_list, dtype=np.float32)
        weight_arr = np.asarray(point_weights, dtype=np.float32)
        altitude_arr = np.asarray(altitudes_list, dtype=np.float32)

        # -- Clustering -----------------------------------------------------
        labels, nodes = _kharita_style_incremental_clustering(
            xs=x_arr,
            ys=y_arr,
            headings=heading_arr,
            point_weights=weight_arr,
            config=config,
        )

        # -- Directed co-occurrence graph -----------------------------------
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

        # -- Edge pruning ---------------------------------------------------
        # Pass 1: minimum support
        edge_support = {
            k: v
            for k, v in edge_support.items()
            if v["support"] >= config.min_edge_support
        }

        # Pass 2: direction conflict
        to_drop = set()
        for (u, v), sv in list(edge_support.items()):
            if (v, u) not in edge_support:
                continue
            sr = edge_support[(v, u)]
            if sv["support"] < sr["support"] * config.reverse_edge_ratio:
                to_drop.add((u, v))
        for k in to_drop:
            edge_support.pop(k, None)

        # Pass 3: transitive pruning
        node_xy = {
            int(r.node_id): (float(r.x), float(r.y))
            for r in nodes.itertuples(index=False)
        }
        edge_lengths = {
            (u, v): float(
                math.hypot(
                    node_xy[u][0] - node_xy[v][0],
                    node_xy[u][1] - node_xy[v][1],
                )
            )
            for (u, v) in edge_support
        }

        graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
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

        # -- Output nodes (WGS84) ------------------------------------------
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

        # -- Output edges ---------------------------------------------------
        edge_rows = []
        for (u, v), s in edge_support.items():
            x1, y1 = node_xy[u]
            x2, y2 = node_xy[v]
            line_xy = LineString([(x1, y1), (x2, y2)])
            line_wgs = transform(to_wgs.transform, line_xy)

            support = max(float(s["support"]), 1.0)
            has_rev = (v, u) in edge_support
            dir_travel = "B" if has_rev else "T"

            day_mode = (
                s["day_counter"].most_common(1)[0][0] if s["day_counter"] else None
            )
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
                    "traffic_signal_count_mean": float(
                        s["traffic_signal_sum"] / support
                    ),
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

        # -- Centerline stitching + smoothing --------------------------------
        centerline_paths = stitch_centerline_paths(edge_support)
        centerline_rows = []

        for path_nodes in centerline_paths:
            if len(path_nodes) < 2:
                continue
            raw_xy = np.asarray([node_xy[n] for n in path_nodes], dtype=np.float64)
            smooth_xy = chaikin(raw_xy, iterations=config.smooth_iterations)

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
            crosswalk: set = set()
            day_counter: Counter = Counter()
            hour_counter: Counter = Counter()

            for i in range(1, len(path_nodes)):
                a = path_nodes[i - 1]
                b = path_nodes[i]
                if (a, b) in edge_support:
                    es = edge_support[(a, b)]
                    fw += es["support"]
                    support_sum += es["support"]
                    weighted_support_sum += es["weighted_support"]
                    construction_sum += es["construction_sum"]
                    signal_sum += es["traffic_signal_sum"]
                    alt_sum += es["altitude_sum"]
                    alt_count += es["altitude_count"]
                    crosswalk.update(es["crosswalk_types"])
                    day_counter.update(es["day_counter"])
                    hour_counter.update(es["hour_counter"])
                if (b, a) in edge_support:
                    es = edge_support[(b, a)]
                    rv += es["support"]
                    support_sum += es["support"]
                    weighted_support_sum += es["weighted_support"]
                    construction_sum += es["construction_sum"]
                    signal_sum += es["traffic_signal_sum"]
                    alt_sum += es["altitude_sum"]
                    alt_count += es["altitude_count"]
                    crosswalk.update(es["crosswalk_types"])
                    day_counter.update(es["day_counter"])
                    hour_counter.update(es["hour_counter"])

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
                    "day_mode": day_counter.most_common(1)[0][0]
                    if day_counter
                    else None,
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
