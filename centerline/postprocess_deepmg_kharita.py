from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from shapely.geometry import LineString

from .utils import angle_diff_deg, bearing_from_xy


@dataclass
class DeepMGPostprocessConfig:
    enabled: bool = False
    link_radius_m: float | None = None
    alpha_virtual: float = 1.4
    min_supp_virtual: int | None = None
    pred_min_supp: int = 1
    similar_direction_deg: float = 20.0
    no_new_vertex_offset_m: float = 15.0
    mm_snap_m: float = 15.0
    max_virtual_links_per_node: int = 2
    enable_duplicate_merge: bool = True
    dup_eps_m: float = 3.0
    enable_curve_smoothing: bool = True
    curve_lambda: float = 0.2


def _support_metric(edge_stats: dict) -> float:
    if "effective_support" in edge_stats:
        return float(edge_stats.get("effective_support", 0.0) or 0.0)
    if "weighted_support" in edge_stats:
        return float(edge_stats.get("weighted_support", 0.0) or 0.0)
    return float(edge_stats.get("support", 0.0) or 0.0)


def _undirected_adjacency(edge_support: Dict[Tuple[int, int], dict]) -> dict:
    adj = {}
    for u, v in edge_support:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _edge_length(node_xy: Dict[int, Tuple[float, float]], u: int, v: int) -> float:
    x1, y1 = node_xy[u]
    x2, y2 = node_xy[v]
    return float(np.hypot(x2 - x1, y2 - y1))


def _init_virtual_edge_stats(template: dict | None = None) -> dict:
    s = {
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
        "day_counter": {},
        "hour_counter": {},
        "path_quality_sum": 0.0,
        "path_quality_count": 0.0,
        "sensor_quality_sum": 0.0,
        "sensor_quality_count": 0.0,
        "hpd_speed_sum": 0.0,
        "hpd_speed_sq_sum": 0.0,
        "hpd_speed_count": 0.0,
        "heading_sin_sum": 0.0,
        "heading_cos_sum": 0.0,
        "heading_count": 0.0,
        "is_virtual": True,
        "postprocess_tags": {"virtual_link_added"},
    }
    if template is not None:
        # Keep compatibility with any additional fields.
        for k in template:
            s.setdefault(k, template[k])
    return s


def _is_line_intersecting_existing(
    *,
    u: int,
    v: int,
    node_xy: Dict[int, Tuple[float, float]],
    existing_undirected: set[Tuple[int, int]],
) -> bool:
    uv = LineString([node_xy[u], node_xy[v]])
    for a, b in existing_undirected:
        if len({u, v, a, b}) < 4:
            continue
        ab = LineString([node_xy[a], node_xy[b]])
        inter = uv.intersection(ab)
        if not inter.is_empty:
            return True
    return False


def _autoscale(
    *,
    sample_point_count: int,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    cfg: DeepMGPostprocessConfig,
) -> tuple[float, int]:
    if not edge_support:
        r0 = float(cfg.link_radius_m) if cfg.link_radius_m is not None else 50.0
        s0 = int(cfg.min_supp_virtual) if cfg.min_supp_virtual is not None else 2
        return max(1.0, r0), max(0, s0)

    und = {(min(u, v), max(u, v)) for (u, v) in edge_support}
    total_len = sum(_edge_length(node_xy, u, v) for (u, v) in und)
    density = float(sample_point_count) / max(total_len, 1.0)  # points / meter
    t = float(np.clip((density - 0.05) / 0.45, 0.0, 1.0))
    auto_radius = 100.0 - 50.0 * t
    auto_min_supp = int(round(2.0 + 3.0 * t))
    if cfg.link_radius_m is not None:
        r = max(1.0, float(cfg.link_radius_m))
    else:
        r = max(50.0, min(100.0, float(auto_radius)))

    if cfg.min_supp_virtual is not None:
        s = max(0, int(cfg.min_supp_virtual))
    else:
        s = max(2, min(5, int(auto_min_supp)))
    return r, s


def _add_virtual_links(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    link_radius_m: float,
    similar_direction_deg: float,
) -> None:
    adj = _undirected_adjacency(edge_support)
    und = {(min(u, v), max(u, v)) for (u, v) in edge_support}
    dead_nodes = [n for n, neigh in adj.items() if len(neigh) == 1]
    if not dead_nodes:
        return

    template = next(iter(edge_support.values())) if edge_support else None
    created = []
    for u in dead_nodes:
        prev = next(iter(adj[u]))
        ux, uy = node_xy[u]
        px, py = node_xy[prev]
        ray_h = bearing_from_xy(px, py, ux, uy)

        best = None
        best_dist = float("inf")
        for v in node_xy.keys():
            if v == u or v == prev:
                continue
            if (min(u, v), max(u, v)) in und:
                continue
            dist = _edge_length(node_xy, u, v)
            if dist > link_radius_m or dist <= 1e-6:
                continue
            vx, vy = node_xy[v]
            hv = bearing_from_xy(ux, uy, vx, vy)
            hd = angle_diff_deg(ray_h, hv)
            if hd > 90.0:
                continue
            if _is_line_intersecting_existing(
                u=u, v=v, node_xy=node_xy, existing_undirected=und
            ):
                continue
            if dist < best_dist:
                best = v
                best_dist = dist

        if best is not None:
            created.append((u, int(best), float(best_dist), float(ray_h)))

    # Remove similar outgoing link directions per dead-end.
    by_u = {}
    for u, v, dist, h in created:
        by_u.setdefault(u, []).append((u, v, dist, h))

    final_links = []
    for u, links in by_u.items():
        links = sorted(links, key=lambda x: x[2])
        kept = []
        kept_h = []
        for item in links:
            _, _, _, h = item
            if all(angle_diff_deg(h, hh) >= similar_direction_deg for hh in kept_h):
                kept.append(item)
                kept_h.append(h)
        final_links.extend(kept[:1])  # one best link per dead-end in this pass

    for u, v, dist, _ in final_links:
        for a, b in [(u, v), (v, u)]:
            if (a, b) in edge_support:
                continue
            s = _init_virtual_edge_stats(template)
            s["length_sum"] = dist
            edge_support[(a, b)] = s


def _refine_with_lightweight_support(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    labels: np.ndarray,
    trace_ranges: Dict[int, Tuple[int, int]],
    alpha_virtual: float,
    min_supp_virtual: int,
    pred_min_supp: int,
    max_hops: int = 6,
) -> None:
    if not edge_support:
        return
    virtual_keys = [k for k, s in edge_support.items() if bool(s.get("is_virtual", False))]
    if not virtual_keys:
        # still enforce pred floor
        to_del = [
            k
            for k, s in edge_support.items()
            if not bool(s.get("is_virtual", False)) and float(s.get("support", 0.0) or 0.0) < float(pred_min_supp)
        ]
        for k in to_del:
            edge_support.pop(k, None)
        return

    vset = set(virtual_keys)
    counts = {k: 0.0 for k in virtual_keys}
    for _, (start, end) in trace_ranges.items():
        if end - start < 2:
            continue
        seq = labels[start:end]
        n = len(seq)
        for i in range(n - 1):
            u = int(seq[i])
            hi = min(n, i + max_hops + 1)
            for j in range(i + 1, hi):
                k = (u, int(seq[j]))
                if k in vset:
                    counts[k] += 1.0
                    break

    to_del = []
    for k in virtual_keys:
        s = edge_support.get(k)
        if s is None:
            continue
        c = float(counts.get(k, 0.0))
        adjusted = c / max(float(alpha_virtual), 1e-6)
        s["support"] = c
        s["weighted_support"] = adjusted
        s.setdefault("postprocess_tags", set()).add("support_refined")
        if adjusted < float(min_supp_virtual):
            to_del.append(k)

    # Pred edge pruning floor.
    for k, s in list(edge_support.items()):
        if bool(s.get("is_virtual", False)):
            continue
        if float(s.get("support", 0.0) or 0.0) < float(pred_min_supp):
            to_del.append(k)

    for k in to_del:
        edge_support.pop(k, None)


def _cap_virtual_links_per_node(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    max_virtual_links_per_node: int,
) -> None:
    if max_virtual_links_per_node <= 0:
        return

    incident = {}
    for (u, v), s in edge_support.items():
        if not bool(s.get("is_virtual", False)):
            continue
        und = (min(u, v), max(u, v))
        incident.setdefault(u, set()).add(und)
        incident.setdefault(v, set()).add(und)

    to_drop_und = set()
    for n, und_links in incident.items():
        if len(und_links) <= max_virtual_links_per_node:
            continue
        sorted_links = sorted(
            und_links,
            key=lambda e: (
                _support_metric(edge_support.get(e, {}))
                + _support_metric(edge_support.get((e[1], e[0]), {})),
                -_edge_length(node_xy, e[0], e[1]),
            ),
            reverse=True,
        )
        for e in sorted_links[max_virtual_links_per_node:]:
            to_drop_und.add(e)

    for u, v in to_drop_und:
        edge_support.pop((u, v), None)
        edge_support.pop((v, u), None)


def _deduplicate_near_parallel_edges(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    dup_eps_m: float,
) -> None:
    keys = list(edge_support.keys())
    to_drop = set()
    for i in range(len(keys)):
        if keys[i] in to_drop:
            continue
        u1, v1 = keys[i]
        if u1 == v1:
            continue
        p1u = np.asarray(node_xy[u1], dtype=np.float64)
        p1v = np.asarray(node_xy[v1], dtype=np.float64)
        h1 = bearing_from_xy(p1u[0], p1u[1], p1v[0], p1v[1])
        for j in range(i + 1, len(keys)):
            if keys[j] in to_drop:
                continue
            u2, v2 = keys[j]
            if len({u1, v1, u2, v2}) <= 2:
                continue
            if {u1, v1} == {u2, v2}:
                continue
            p2u = np.asarray(node_xy[u2], dtype=np.float64)
            p2v = np.asarray(node_xy[v2], dtype=np.float64)
            d_direct = max(float(np.hypot(*(p1u - p2u))), float(np.hypot(*(p1v - p2v))))
            d_swap = max(float(np.hypot(*(p1u - p2v))), float(np.hypot(*(p1v - p2u))))
            d = min(d_direct, d_swap)
            if d > dup_eps_m:
                continue
            h2 = bearing_from_xy(p2u[0], p2u[1], p2v[0], p2v[1])
            if angle_diff_deg(h1, h2) > 20.0:
                continue

            s1 = _support_metric(edge_support[keys[i]])
            s2 = _support_metric(edge_support[keys[j]])
            if s1 >= s2:
                to_drop.add(keys[j])
            else:
                to_drop.add(keys[i])
                break

    for k in to_drop:
        edge_support.pop(k, None)


def run_deepmg_kharita_postprocess(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    labels: np.ndarray,
    trace_ranges: Dict[int, Tuple[int, int]],
    sample_point_count: int,
    config: DeepMGPostprocessConfig,
) -> dict:
    if not config.enabled or not edge_support:
        return {
            "post_link_radius_m": config.link_radius_m,
            "post_min_supp_virtual": config.min_supp_virtual,
            "virtual_links_added": 0,
        }

    for s in edge_support.values():
        s.setdefault("is_virtual", False)
        s.setdefault("postprocess_tags", set())

    link_radius_m, min_supp_virtual = _autoscale(
        sample_point_count=sample_point_count,
        edge_support=edge_support,
        node_xy=node_xy,
        cfg=config,
    )

    n0 = sum(1 for s in edge_support.values() if bool(s.get("is_virtual", False)))
    _add_virtual_links(
        edge_support=edge_support,
        node_xy=node_xy,
        link_radius_m=link_radius_m,
        similar_direction_deg=float(config.similar_direction_deg),
    )
    n1 = sum(1 for s in edge_support.values() if bool(s.get("is_virtual", False)))

    _refine_with_lightweight_support(
        edge_support=edge_support,
        labels=labels,
        trace_ranges=trace_ranges,
        alpha_virtual=float(config.alpha_virtual),
        min_supp_virtual=int(min_supp_virtual),
        pred_min_supp=int(config.pred_min_supp),
    )

    _cap_virtual_links_per_node(
        edge_support=edge_support,
        node_xy=node_xy,
        max_virtual_links_per_node=int(config.max_virtual_links_per_node),
    )
    if config.enable_duplicate_merge:
        _deduplicate_near_parallel_edges(
            edge_support=edge_support,
            node_xy=node_xy,
            dup_eps_m=float(config.dup_eps_m),
        )

    return {
        "post_link_radius_m": float(link_radius_m),
        "post_min_supp_virtual": int(min_supp_virtual),
        "virtual_links_added": int(max(0, n1 - n0)),
    }
