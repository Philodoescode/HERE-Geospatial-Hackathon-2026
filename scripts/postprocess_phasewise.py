"""Phasewise DeepMG postprocessing for Kharita outputs.

Run one phase at a time so you can inspect intermediate results before
continuing.  State is persisted to a pickle file between phases.

Phases
------
1. load              Load nodes/edges, build edge_support, compute autoscale.
2. virtual_links     Add virtual links to dead-end nodes.
3. support_refine    Prune low-support edges (virtual + existing).
4. cap_virtual       Limit virtual links per node.
5. deduplicate       Remove near-parallel duplicate edges.
6. rebuild           Rebuild edges/centerlines DataFrames, save final output.

Usage
-----
    # Run phase 1 only (creates state pickle + snapshot GPKG)
    python scripts/postprocess_phasewise.py --phase load

    # Continue with phase 2
    python scripts/postprocess_phasewise.py --phase virtual_links

    # ... inspect snapshots in outputs/kharita_full_tuned_no_deepmg_phasewise/ ...

    # Or run all phases in sequence
    python scripts/postprocess_phasewise.py --phase all
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.postprocess_deepmg_kharita import (
    DeepMGPostprocessConfig,
    _autoscale,
    _cap_virtual_links_per_node,
    _deduplicate_near_parallel_edges,
    _refine_with_lightweight_support,
    _support_metric,
    _undirected_adjacency,
    _edge_length,
    _init_virtual_edge_stats,
    angle_diff_deg,
    bearing_from_xy,
)
from centerline.utils import smooth_polyline_preserve_turns, stitch_centerline_paths


# ---------------------------------------------------------------------------
# Optimized Virtual Link Algorithm
# ---------------------------------------------------------------------------


def _fast_add_virtual_links(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    link_radius_m: float,
    similar_direction_deg: float,
) -> None:
    """Optimized version of _add_virtual_links using KDTree and STRtree."""
    adj = _undirected_adjacency(edge_support)
    und = {(min(u, v), max(u, v)) for (u, v) in edge_support}
    dead_nodes = [n for n, neigh in adj.items() if len(neigh) == 1]

    if not dead_nodes:
        return

    # Build spatial index for nodes
    node_ids = sorted(list(node_xy.keys()))
    node_idx_map = {uid: i for i, uid in enumerate(node_ids)}
    coords = np.array([node_xy[uid] for uid in node_ids])
    tree = KDTree(coords)

    # Build spatial index for existing edges to check intersections
    existing_geoms = []
    existing_geom_indices = []  # map back to (u, v) if needed, or just used for intersection count
    for u, v in und:
        if u not in node_xy or v not in node_xy:
            continue
        line = LineString([node_xy[u], node_xy[v]])
        existing_geoms.append(line)

    str_tree = STRtree(existing_geoms) if existing_geoms else None

    template = next(iter(edge_support.values())) if edge_support else None
    created = []

    # Process dead-end nodes
    for u in dead_nodes:
        if u not in adj or not adj[u]:
            continue
        prev = next(iter(adj[u]))
        ux, uy = node_xy[u]
        px, py = node_xy[prev]
        ray_h = bearing_from_xy(px, py, ux, uy)

        # Query KDTree for candidates within radius
        # k=100 limit to avoid checking too many points in dense areas
        dists, indices = tree.query([ux, uy], k=50, distance_upper_bound=link_radius_m)

        # Unpack if single neighbor
        if not isinstance(indices, (list, np.ndarray)):
            indices = [indices]
            dists = [dists]

        best = None
        best_dist = float("inf")

        for idx, dist in zip(indices, dists):
            if idx == len(node_ids):  # KDTree returns N for no neighbor
                continue

            v = node_ids[idx]
            if v == u or v == prev:
                continue

            # Check if edge already exists
            if (min(u, v), max(u, v)) in und:
                continue

            # Distance check (already done by KDTree but double check)
            if dist > link_radius_m or dist <= 1e-6:
                continue

            vx, vy = node_xy[v]
            hv = bearing_from_xy(ux, uy, vx, vy)
            hd = angle_diff_deg(ray_h, hv)

            # Angle check: must be roughly forward (within 90 deg of ray)
            if abs(hd) > 90.0:
                continue

            # Intersection check
            uv_line = LineString([(ux, uy), (vx, vy)])

            # Fast intersection check using STRtree
            intersecting = False
            if str_tree:
                # Query potential intersections
                candidates = str_tree.query(uv_line)
                for cand_idx in candidates:
                    geom = existing_geoms[cand_idx]
                    # Valid intersection if they touch not just at endpoints
                    # But endpoints u and v are allowed to touch existing graph (obviously)
                    # We want to avoid crossing *middle* of other edges
                    if uv_line.intersects(geom):
                        # intersection is allowed if it's just at endpoints
                        inter = uv_line.intersection(geom)
                        # If intersection is just a point and that point is u or v, it's fine
                        # But wait, we are checking if the *segment* uv cuts through another segment
                        # standard check:
                        if not inter.is_empty:
                            # If intersection is a point equal to one of our endpoints, ignore
                            if isinstance(inter, Point):
                                if (
                                    inter.distance(Point(ux, uy)) < 1e-3
                                    or inter.distance(Point(vx, vy)) < 1e-3
                                ):
                                    continue  # Touching at endpoint is fine
                            # Otherwise it's a real crossing
                            intersecting = True
                            break

            if intersecting:
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


# ---------------------------------------------------------------------------
# Defaults -- edit these or override via CLI
# ---------------------------------------------------------------------------
DEFAULT_NODES = (
    ROOT
    / "outputs/kharita_full_tuned_no_deepmg/kharita_full_tuned_no_deepmg_nodes.parquet"
)
DEFAULT_EDGES = (
    ROOT
    / "outputs/kharita_full_tuned_no_deepmg/kharita_full_tuned_no_deepmg_edges.parquet"
)
DEFAULT_OUT_DIR = ROOT / "outputs/kharita_full_tuned_no_deepmg_phasewise"

PHASES = [
    "load",
    "virtual_links",
    "support_refine",
    "cap_virtual",
    "deduplicate",
    "simplify",
    "rebuild",
]


# ---------------------------------------------------------------------------
# Loaders (reused from postprocess_kharita_outputs.py)
# ---------------------------------------------------------------------------


def _parse_list_like(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    txt = str(value).strip()
    if not txt:
        return []
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _load_nodes(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".gpkg":
        df = gpd.read_file(path, layer="nodes")
    else:
        raise ValueError(f"Unsupported: {path}")
    required = {"node_id", "x", "y", "lon", "lat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Nodes missing columns: {missing}")
    for c in ["node_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["x", "y", "lon", "lat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["node_id", "x", "y", "lon", "lat"]).copy()
    df["node_id"] = df["node_id"].astype(int)
    return df


def _load_edges(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".gpkg":
        df = gpd.read_file(path, layer="edges")
    else:
        raise ValueError(f"Unsupported: {path}")
    required = {"u", "v", "support", "weighted_support"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Edges missing columns: {missing}")
    for c in ["u", "v"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    num_cols = [
        "support",
        "weighted_support",
        "effective_support",
        "vpd_support",
        "hpd_support",
        "mean_step_length_m",
        "construction_percent_mean",
        "traffic_signal_count_mean",
        "altitude_mean",
        "dyn_w_probe",
        "dyn_w_vpd",
        "road_likeness_score",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "crosswalk_types" not in df.columns:
        df["crosswalk_types"] = [[] for _ in range(len(df))]
    else:
        df["crosswalk_types"] = df["crosswalk_types"].map(_parse_list_like)
    if "postprocess_tags" not in df.columns:
        df["postprocess_tags"] = [[] for _ in range(len(df))]
    else:
        df["postprocess_tags"] = df["postprocess_tags"].map(_parse_list_like)
    if "is_virtual_link" not in df.columns:
        df["is_virtual_link"] = False
    else:
        df["is_virtual_link"] = df["is_virtual_link"].fillna(False).astype(bool)
    df = df.dropna(subset=["u", "v"]).copy()
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    return df


def _build_edge_support(edges: pd.DataFrame) -> dict[tuple[int, int], dict]:
    edge_support: dict[tuple[int, int], dict] = {}
    for r in edges.itertuples(index=False):
        u, v = int(r.u), int(r.v)
        support = float(getattr(r, "support", 0.0) or 0.0)
        weighted = float(getattr(r, "weighted_support", support) or 0.0)
        mean_step = float(getattr(r, "mean_step_length_m", 0.0) or 0.0)
        length_sum = max(0.0, mean_step) * max(support, 0.0)
        altitude_mean = getattr(r, "altitude_mean", np.nan)
        s = {
            "support": support,
            "weighted_support": weighted,
            "effective_support": float(
                getattr(r, "effective_support", weighted) or weighted
            ),
            "vpd_support": float(getattr(r, "vpd_support", 0.0) or 0.0),
            "hpd_support": float(getattr(r, "hpd_support", 0.0) or 0.0),
            "length_sum": length_sum,
            "construction_sum": float(
                getattr(r, "construction_percent_mean", 0.0) or 0.0
            )
            * max(support, 0.0),
            "traffic_signal_sum": float(
                getattr(r, "traffic_signal_count_mean", 0.0) or 0.0
            )
            * max(support, 0.0),
            "altitude_sum": float(altitude_mean) * max(support, 0.0)
            if np.isfinite(altitude_mean)
            else 0.0,
            "altitude_count": int(max(round(support), 0))
            if np.isfinite(altitude_mean)
            else 0,
            "crosswalk_types": set(_parse_list_like(getattr(r, "crosswalk_types", []))),
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
            "is_virtual": bool(getattr(r, "is_virtual_link", False)),
            "postprocess_tags": set(
                _parse_list_like(getattr(r, "postprocess_tags", []))
            ),
            "dyn_w_probe": float(getattr(r, "dyn_w_probe", 0.0) or 0.0),
            "dyn_w_vpd": float(getattr(r, "dyn_w_vpd", 1.0) or 1.0),
            "road_likeness_score": float(getattr(r, "road_likeness_score", 0.5) or 0.5),
        }
        edge_support[(u, v)] = s
    return edge_support


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _edge_stats(edge_support: dict) -> dict:
    """Compute summary statistics for the current edge_support state."""
    total = len(edge_support)
    und = {(min(u, v), max(u, v)) for (u, v) in edge_support}
    virtual = sum(1 for s in edge_support.values() if s.get("is_virtual", False))
    supports = [_support_metric(s) for s in edge_support.values()]
    # Count dead-end nodes
    adj: dict[int, set] = {}
    for u, v in edge_support:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    dead_ends = sum(1 for n, neigh in adj.items() if len(neigh) == 1)

    stats = {
        "directed_edge_count": total,
        "undirected_edge_count": len(und),
        "virtual_edge_count": virtual,
        "non_virtual_edge_count": total - virtual,
        "node_count": len(adj),
        "dead_end_count": dead_ends,
    }
    if supports:
        arr = np.array(supports)
        stats.update(
            {
                "support_mean": float(np.mean(arr)),
                "support_median": float(np.median(arr)),
                "support_min": float(np.min(arr)),
                "support_max": float(np.max(arr)),
                "support_p10": float(np.percentile(arr, 10)),
                "support_p90": float(np.percentile(arr, 90)),
            }
        )
    return stats


def _save_edge_snapshot(
    edge_support: dict,
    node_xy: dict,
    node_lonlat: dict,
    out_dir: Path,
    phase_name: str,
) -> Path:
    """Save a GeoPackage snapshot of edges for visual inspection."""
    rows = []
    for (u, v), s in edge_support.items():
        if u not in node_lonlat or v not in node_lonlat:
            continue
        lon1, lat1 = node_lonlat[u]
        lon2, lat2 = node_lonlat[v]
        rows.append(
            {
                "u": u,
                "v": v,
                "support": float(s.get("support", 0.0) or 0.0),
                "weighted_support": float(s.get("weighted_support", 0.0) or 0.0),
                "effective_support": float(s.get("effective_support", 0.0) or 0.0),
                "is_virtual": bool(s.get("is_virtual", False)),
                "geometry": LineString([(lon1, lat1), (lon2, lat2)]),
            }
        )
    if not rows:
        print(f"  [snapshot] No edges to save for phase '{phase_name}'")
        return out_dir / f"phase_{phase_name}_edges.gpkg"

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    path = out_dir / f"phase_{phase_name}_edges.gpkg"
    gdf.to_file(path, driver="GPKG", layer="edges")
    print(f"  [snapshot] Saved {len(rows)} edges -> {path}")
    return path


def _save_state(state: dict, out_dir: Path) -> Path:
    """Persist the pipeline state to a pickle file."""
    path = out_dir / "pipeline_state.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)
    return path


def _load_state(out_dir: Path) -> dict:
    """Load persisted pipeline state."""
    path = out_dir / "pipeline_state.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No pipeline state found at {path}. Run phase 'load' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def phase_load(args: argparse.Namespace) -> dict:
    """Phase 1: Load data, build edge_support, compute autoscale parameters."""
    print("=" * 60)
    print("PHASE: load")
    print("=" * 60)

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    print(f"  Loading nodes from {nodes_path}")
    nodes_df = _load_nodes(nodes_path)
    print(f"  Loaded {len(nodes_df)} nodes")

    print(f"  Loading edges from {edges_path}")
    edges_df = _load_edges(edges_path)
    print(f"  Loaded {len(edges_df)} edges")

    node_xy = {
        int(r.node_id): (float(r.x), float(r.y))
        for r in nodes_df.itertuples(index=False)
    }
    node_lonlat = {
        int(r.node_id): (float(r.lon), float(r.lat))
        for r in nodes_df.itertuples(index=False)
    }

    print("  Building edge_support dict...")
    edge_support = _build_edge_support(edges_df)
    print(f"  Built edge_support with {len(edge_support)} directed edges")

    # Init virtual/postprocess tags
    for s in edge_support.values():
        s.setdefault("is_virtual", False)
        s.setdefault("postprocess_tags", set())

    # Estimate sample_point_count
    estimated_spc = int(
        max(
            1.0, sum(float(s.get("support", 0.0) or 0.0) for s in edge_support.values())
        )
    )
    sample_point_count = (
        args.assume_sample_point_count
        if args.assume_sample_point_count
        else estimated_spc
    )
    print(f"  Estimated sample_point_count = {estimated_spc}")
    print(f"  Using sample_point_count = {sample_point_count}")

    # Build config
    cfg = DeepMGPostprocessConfig(
        enabled=True,
        link_radius_m=args.post_link_radius_m,
        alpha_virtual=args.post_alpha_virtual,
        min_supp_virtual=args.post_min_supp_virtual,
        pred_min_supp=args.post_pred_min_supp,
        similar_direction_deg=args.post_similar_direction_deg,
        max_virtual_links_per_node=args.post_max_virtual_links_per_node,
        enable_duplicate_merge=args.post_enable_duplicate_merge,
        dup_eps_m=args.post_dup_eps_m,
        enable_curve_smoothing=args.post_enable_curve_smoothing,
        curve_lambda=args.post_curve_lambda,
    )

    # Autoscale
    link_radius_m, min_supp_virtual = _autoscale(
        sample_point_count=sample_point_count,
        edge_support=edge_support,
        node_xy=node_xy,
        cfg=cfg,
    )
    print(f"  Autoscale results:")
    print(f"    link_radius_m    = {link_radius_m:.1f}")
    print(f"    min_supp_virtual = {min_supp_virtual}")

    stats = _edge_stats(edge_support)
    print(f"\n  Edge stats after load:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    state = {
        "nodes_df": nodes_df,
        "node_xy": node_xy,
        "node_lonlat": node_lonlat,
        "edge_support": edge_support,
        "sample_point_count": sample_point_count,
        "link_radius_m": link_radius_m,
        "min_supp_virtual": min_supp_virtual,
        "cfg": cfg,
        "completed_phases": ["load"],
        "phase_stats": {"load": stats},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_edge_snapshot(edge_support, node_xy, node_lonlat, out_dir, "1_load")
    _save_state(state, out_dir)
    print(f"\n  State saved to {out_dir / 'pipeline_state.pkl'}")
    return state


def _fast_deduplicate_near_parallel_edges(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    dup_eps_m: float,
) -> None:
    """Optimized deduplication using KDTree."""
    # Build list of undirected edges to check
    # We only care about (u, v) where u < v to avoid double processing
    # But we need to handle (u, v) and (v, u) supports together?
    # No, original code iterates keys() which are directed edges.
    # But it handles them pairwise?
    # Original: u1, v1 = keys[i]; u2, v2 = keys[j].
    # If {u1, v1} == {u2, v2}, continue.
    # It compares DIFFERENT edges (geometrically distinct but close).

    keys = sorted(list(edge_support.keys()))  # Sort for determinism
    to_drop = set()

    # Precompute edge geometries and headings to save time
    # Index edges by start node for fast lookup
    edges_by_start = {}
    edges_by_end = {}

    # Also need node KDTree
    node_ids = sorted(list(node_xy.keys()))
    node_idx_map = {uid: i for i, uid in enumerate(node_ids)}
    coords = np.array([node_xy[uid] for uid in node_ids])
    tree = KDTree(coords)

    for k in keys:
        u, v = k
        edges_by_start.setdefault(u, []).append(k)
        edges_by_end.setdefault(v, []).append(k)

    processed = set()

    for k1 in keys:
        if k1 in to_drop:
            continue

        u1, v1 = k1

        # We want to find k2=(u2, v2) such that u2 near u1 and v2 near v1 (or swapped)
        # 1. Direct match: u2 near u1 AND v2 near v1
        # 2. Swap match: u2 near v1 AND v2 near u1

        # Get neighbors of u1
        # Use a slightly larger radius to be safe? No, eps is strict.
        u1_xy = node_xy[u1]
        v1_xy = node_xy[v1]

        near_u1_indices = tree.query_ball_point(u1_xy, r=dup_eps_m)
        near_v1_indices = tree.query_ball_point(v1_xy, r=dup_eps_m)

        near_u1 = {node_ids[i] for i in near_u1_indices}
        near_v1 = {node_ids[i] for i in near_v1_indices}

        candidates = set()

        # Case 1: Direct (u2 near u1)
        for u2 in near_u1:
            # Edges starting at u2
            for k2 in edges_by_start.get(u2, []):
                if k2 == k1:
                    continue
                if k2 in to_drop:
                    continue
                _, v2 = k2
                if v2 in near_v1:
                    candidates.add(k2)

            # Edges ending at u2 (reverse direction in storage but geometric match)
            # If k2=(v2, u2) where u2 is near u1...
            # Then v2 must be near v1 for direct match?
            # No, if k2=(v2, u2) then starts at v2.
            # If u2 near u1, then end of k2 is near start of k1.
            # If v2 near v1, then start of k2 is near end of k1.
            # This is "swap" logic relative to directed graph, but "direct" geometrically?
            pass

        # Simplified approach:
        # Just check all edges connected to any node near u1 or v1?
        # That might be too many.

        # Better: Iterate candidates found via node proximity

        # Check direct candidates: starts near u1, ends near v1
        for u2 in near_u1:
            for k2 in edges_by_start.get(u2, []):
                if k2 == k1 or k2 in to_drop:
                    continue
                if k2[1] in near_v1:
                    candidates.add(k2)

        # Check swap candidates: starts near v1, ends near u1
        for u2 in near_v1:
            for k2 in edges_by_start.get(u2, []):
                if k2 == k1 or k2 in to_drop:
                    continue
                if k2[1] in near_u1:
                    candidates.add(k2)

        # Also check reverse stored edges?
        # The edge_support contains directed edges.
        # If we have (u1, v1) and (u2, v2) where u1~u2 and v1~v2, we compare them.
        # If we have (u1, v1) and (v2, u2) where u1~v2 and v1~u2, we compare them.
        # My candidate search above covers both cases because:
        # If k2=(v2, u2) is swap-match to k1=(u1, v1), then start(k2)=v2 is near end(k1)=v1.
        # And end(k2)=u2 is near start(k1)=u1.
        # My "swap candidates" loop checks starts near v1. So it finds k2=(v2, ...) where v2 near v1.
        # Then checks if end k2[1] is near u1.
        # So yes, it covers it.

        for k2 in candidates:
            if k2 in to_drop:
                continue

            u2, v2 = k2

            # Additional checks from original code
            # "if len({u1, v1, u2, v2}) <= 2: continue" -> shared endpoints
            if len({u1, v1, u2, v2}) <= 2:
                # If they share endpoints but are different edges (e.g. parallel curves),
                # dist check handles it.
                # But original code skips if <= 2 unique nodes?
                # "if {u1, v1} == {u2, v2}: continue" -> same edge (maybe reversed)
                # But if u1=u2 and v1=v2, it's same directed edge (already filtered k1!=k2)
                # If u1=v2 and v1=u2, it's reverse edge. Original code skips this?
                # Yes: "if {u1, v1} == {u2, v2}: continue"
                if {u1, v1} == {u2, v2}:
                    continue
                # If they share 1 node, e.g. u1=u2, then d_direct = max(0, dist(v1, v2)).
                # That is valid to merge if v1 is very close to v2.
                pass

            p1u = np.asarray(node_xy[u1])
            p1v = np.asarray(node_xy[v1])
            p2u = np.asarray(node_xy[u2])
            p2v = np.asarray(node_xy[v2])

            # Recalculate dists to be sure
            d_direct = max(np.linalg.norm(p1u - p2u), np.linalg.norm(p1v - p2v))
            d_swap = max(np.linalg.norm(p1u - p2v), np.linalg.norm(p1v - p2u))
            d = min(d_direct, d_swap)

            if d > dup_eps_m:
                continue

            h1 = bearing_from_xy(p1u[0], p1u[1], p1v[0], p1v[1])
            h2 = bearing_from_xy(p2u[0], p2u[1], p2v[0], p2v[1])

            # If swapped, h2 should be roughly h1 + 180?
            # Original code: "if angle_diff_deg(h1, h2) > 20.0: continue"
            # This implies it only merges if they point in SAME direction.
            # So if k2 is reverse of k1 (geometrically), h2 ~ h1 + 180.
            # angle_diff(h1, h2) would be ~180 > 20. So it won't merge opposing edges.
            # This is correct for directed graph.
            # What if k2 is physically same but directed v2->u2?
            # Then d_swap is small. But h2 is opposite.
            # So we only merge edges flowing in same direction.

            if angle_diff_deg(h1, h2) > 20.0:
                continue

            s1 = _support_metric(edge_support[k1])
            s2 = _support_metric(edge_support[k2])

            if s1 >= s2:
                to_drop.add(k2)
            else:
                to_drop.add(k1)
                break  # k1 is dropped, stop checking k1 against others

    for k in to_drop:
        edge_support.pop(k, None)


def phase_virtual_links(args: argparse.Namespace) -> dict:
    """Phase 2: Add virtual links to dead-end nodes."""
    print("=" * 60)
    print("PHASE: virtual_links")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    state = _load_state(out_dir)
    if "load" not in state["completed_phases"]:
        raise RuntimeError("Phase 'load' must be completed first.")

    edge_support = state["edge_support"]
    node_xy = state["node_xy"]
    link_radius_m = state["link_radius_m"]
    cfg = state["cfg"]

    n_before = len(edge_support)
    n_virtual_before = sum(
        1 for s in edge_support.values() if s.get("is_virtual", False)
    )

    print(f"  Edges before: {n_before} (virtual: {n_virtual_before})")
    print(
        f"  Adding virtual links (radius={link_radius_m:.1f}m, "
        f"similar_dir={cfg.similar_direction_deg}deg)..."
    )

    t0 = time.time()
    _fast_add_virtual_links(
        edge_support=edge_support,
        node_xy=node_xy,
        link_radius_m=link_radius_m,
        similar_direction_deg=float(cfg.similar_direction_deg),
    )
    elapsed = time.time() - t0

    n_after = len(edge_support)
    n_virtual_after = sum(
        1 for s in edge_support.values() if s.get("is_virtual", False)
    )
    print(f"  Edges after:  {n_after} (virtual: {n_virtual_after})")
    print(f"  New virtual links added: {n_virtual_after - n_virtual_before}")
    print(f"  Time: {elapsed:.1f}s")

    stats = _edge_stats(edge_support)
    print(f"\n  Edge stats after virtual_links:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    state["completed_phases"].append("virtual_links")
    state["phase_stats"]["virtual_links"] = stats

    _save_edge_snapshot(
        edge_support, node_xy, state["node_lonlat"], out_dir, "2_virtual_links"
    )
    _save_state(state, out_dir)
    print(f"\n  State saved.")
    return state


def phase_support_refine(args: argparse.Namespace) -> dict:
    """Phase 3: Prune low-support edges."""
    print("=" * 60)
    print("PHASE: support_refine")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    state = _load_state(out_dir)
    if "virtual_links" not in state["completed_phases"]:
        raise RuntimeError("Phase 'virtual_links' must be completed first.")

    edge_support = state["edge_support"]
    cfg = state["cfg"]
    min_supp_virtual = state["min_supp_virtual"]

    n_before = len(edge_support)
    print(f"  Edges before: {n_before}")
    print(
        f"  Refining support (alpha_virtual={cfg.alpha_virtual}, "
        f"min_supp_virtual={min_supp_virtual}, pred_min_supp={cfg.pred_min_supp})..."
    )
    print(f"  NOTE: Standalone mode -- no trace labels available.")
    print(f"        Support refinement uses empty traces (virtual edges keep their")
    print(f"        initial support=0, so only min_supp_virtual threshold matters).")

    t0 = time.time()
    _refine_with_lightweight_support(
        edge_support=edge_support,
        labels=np.asarray([], dtype=np.int32),
        trace_ranges={},
        alpha_virtual=float(cfg.alpha_virtual),
        min_supp_virtual=int(min_supp_virtual),
        pred_min_supp=int(cfg.pred_min_supp),
    )
    elapsed = time.time() - t0

    n_after = len(edge_support)
    n_removed = n_before - n_after
    print(f"  Edges after:  {n_after}")
    print(f"  Edges pruned: {n_removed}")
    print(f"  Time: {elapsed:.1f}s")

    stats = _edge_stats(edge_support)
    print(f"\n  Edge stats after support_refine:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    state["completed_phases"].append("support_refine")
    state["phase_stats"]["support_refine"] = stats

    _save_edge_snapshot(
        edge_support,
        state["node_xy"],
        state["node_lonlat"],
        out_dir,
        "3_support_refine",
    )
    _save_state(state, out_dir)
    print(f"\n  State saved.")
    return state


def phase_cap_virtual(args: argparse.Namespace) -> dict:
    """Phase 4: Cap virtual links per node."""
    print("=" * 60)
    print("PHASE: cap_virtual")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    state = _load_state(out_dir)
    if "support_refine" not in state["completed_phases"]:
        raise RuntimeError("Phase 'support_refine' must be completed first.")

    edge_support = state["edge_support"]
    node_xy = state["node_xy"]
    cfg = state["cfg"]

    n_before = len(edge_support)
    n_virtual_before = sum(
        1 for s in edge_support.values() if s.get("is_virtual", False)
    )
    print(f"  Edges before: {n_before} (virtual: {n_virtual_before})")
    print(f"  Capping virtual links per node to {cfg.max_virtual_links_per_node}...")

    t0 = time.time()
    _cap_virtual_links_per_node(
        edge_support=edge_support,
        node_xy=node_xy,
        max_virtual_links_per_node=int(cfg.max_virtual_links_per_node),
    )
    elapsed = time.time() - t0

    n_after = len(edge_support)
    n_virtual_after = sum(
        1 for s in edge_support.values() if s.get("is_virtual", False)
    )
    print(f"  Edges after:  {n_after} (virtual: {n_virtual_after})")
    print(f"  Virtual links removed: {n_virtual_before - n_virtual_after}")
    print(f"  Time: {elapsed:.1f}s")

    stats = _edge_stats(edge_support)
    print(f"\n  Edge stats after cap_virtual:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    state["completed_phases"].append("cap_virtual")
    state["phase_stats"]["cap_virtual"] = stats

    _save_edge_snapshot(
        edge_support, node_xy, state["node_lonlat"], out_dir, "4_cap_virtual"
    )
    _save_state(state, out_dir)
    print(f"\n  State saved.")
    return state


def phase_deduplicate(args: argparse.Namespace) -> dict:
    """Phase 5: Remove near-parallel duplicate edges."""
    print("=" * 60)
    print("PHASE: deduplicate")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    state = _load_state(out_dir)
    if "cap_virtual" not in state["completed_phases"]:
        raise RuntimeError("Phase 'cap_virtual' must be completed first.")

    edge_support = state["edge_support"]
    node_xy = state["node_xy"]
    cfg = state["cfg"]

    n_before = len(edge_support)
    print(f"  Edges before: {n_before}")

    if not cfg.enable_duplicate_merge:
        print(f"  Duplicate merge DISABLED in config. Skipping.")
    else:
        print(f"  Deduplicating near-parallel edges (eps={cfg.dup_eps_m}m)...")
        t0 = time.time()
        # Use optimized fast version
        _fast_deduplicate_near_parallel_edges(
            edge_support=edge_support,
            node_xy=node_xy,
            dup_eps_m=float(cfg.dup_eps_m),
        )
        elapsed = time.time() - t0
        n_after = len(edge_support)
        print(f"  Edges after:  {n_after}")
        print(f"  Edges removed: {n_before - n_after}")
        print(f"  Time: {elapsed:.1f}s")

    stats = _edge_stats(edge_support)
    print(f"\n  Edge stats after deduplicate:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    state["completed_phases"].append("deduplicate")
    state["phase_stats"]["deduplicate"] = stats

    _save_edge_snapshot(
        edge_support, node_xy, state["node_lonlat"], out_dir, "5_deduplicate"
    )
    _save_state(state, out_dir)
    print(f"\n  State saved.")
    return state


def _merge_close_nodes(
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    node_lonlat: Dict[int, Tuple[float, float]],
    snap_dist_m: float,
) -> int:
    """Merge nodes that are within snap_dist_m of each other."""
    if snap_dist_m <= 0:
        return 0

    node_ids = sorted(list(node_xy.keys()))
    coords = np.array([node_xy[uid] for uid in node_ids])
    tree = KDTree(coords)

    # Find clusters
    # query_ball_tree finds all pairs within r
    # This can be slow if r is large, but for 4m it should be fast
    # But we want connected components of close nodes

    # Simple greedy approach:
    # Iterate nodes, if not remapped, find neighbors, merge to first.

    remap = {}  # old_id -> new_id
    merged_count = 0

    # Use visited set to avoid reprocessing
    visited = set()

    for i, u in enumerate(node_ids):
        if u in visited:
            continue

        u_xy = node_xy[u]
        # Find neighbors within snap_dist
        indices = tree.query_ball_point(u_xy, r=snap_dist_m)

        cluster = []
        for idx in indices:
            v = node_ids[idx]
            if v not in visited:
                cluster.append(v)
                visited.add(v)

        if len(cluster) <= 1:
            continue

        # Merge cluster to u (or centroid?)
        # Centroid is better geometry
        # But we need to keep one ID as representative. Let's use u.

        # Calculate centroid
        c_x = np.mean([node_xy[v][0] for v in cluster])
        c_y = np.mean([node_xy[v][1] for v in cluster])
        c_lon = np.mean([node_lonlat[v][0] for v in cluster])
        c_lat = np.mean([node_lonlat[v][1] for v in cluster])

        # Update u's position
        node_xy[u] = (c_x, c_y)
        node_lonlat[u] = (c_lon, c_lat)

        for v in cluster:
            if v != u:
                remap[v] = u
                merged_count += 1

    if not remap:
        return 0

    # Remap edges
    new_edges = {}
    for (u, v), s in edge_support.items():
        new_u = remap.get(u, u)
        new_v = remap.get(v, v)

        if new_u == new_v:
            # Loop created, drop it
            continue

        # If multiple edges merge into same (new_u, new_v), keep best support
        if (new_u, new_v) in new_edges:
            s_old = new_edges[(new_u, new_v)]
            if _support_metric(s) > _support_metric(s_old):
                new_edges[(new_u, new_v)] = s
        else:
            new_edges[(new_u, new_v)] = s

    # Replace edge_support in place
    edge_support.clear()
    edge_support.update(new_edges)

    # Remove merged nodes from node dicts?
    # Not strictly necessary if we don't iterate them again, but cleaner
    for v in remap:
        if v in node_xy:
            del node_xy[v]
        if v in node_lonlat:
            del node_lonlat[v]

    return merged_count


def phase_simplify(args: argparse.Namespace) -> dict:
    """Phase 5.5: Simplify graph by merging close nodes."""
    print("=" * 60)
    print("PHASE: simplify")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    state = _load_state(out_dir)
    # Can run after deduplicate or cap_virtual
    prev_phases = set(state["completed_phases"])
    if "deduplicate" not in prev_phases and "cap_virtual" not in prev_phases:
        raise RuntimeError("Must run after 'cap_virtual' or 'deduplicate'.")

    edge_support = state["edge_support"]
    node_xy = state["node_xy"]
    node_lonlat = state["node_lonlat"]

    # Config for snap distance? Add to args or hardcode decent default
    snap_dist = getattr(args, "simplify_snap_dist_m", 4.0)

    print(f"  Merging nodes within {snap_dist}m...")
    n_before = len(node_xy)
    e_before = len(edge_support)

    merged = _merge_close_nodes(edge_support, node_xy, node_lonlat, snap_dist)

    n_after = len(node_xy)
    e_after = len(edge_support)

    print(f"  Nodes merged: {merged}")
    print(f"  Nodes remaining: {n_after} (was {n_before})")
    print(f"  Edges remaining: {e_after} (was {e_before})")

    stats = _edge_stats(edge_support)
    state["completed_phases"].append("simplify")
    state["phase_stats"]["simplify"] = stats

    _save_edge_snapshot(edge_support, node_xy, node_lonlat, out_dir, "5b_simplify")
    _save_state(state, out_dir)
    print(f"\n  State saved.")
    return state


def phase_rebuild(args: argparse.Namespace) -> dict:
    """Phase 6: Rebuild edges/centerlines DataFrames and save final output."""
    print("=" * 60)
    print("PHASE: rebuild")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    state = _load_state(out_dir)
    if "deduplicate" not in state["completed_phases"]:
        raise RuntimeError("Phase 'deduplicate' must be completed first.")

    edge_support = state["edge_support"]
    node_xy = state["node_xy"]
    node_lonlat = state["node_lonlat"]
    nodes_df = state["nodes_df"]
    cfg = state["cfg"]

    # Rebuild edges DataFrame
    print("  Rebuilding edges DataFrame...")
    edge_rows = []
    for (u, v), s in edge_support.items():
        if u not in node_xy or v not in node_xy:
            continue
        x1, y1 = node_xy[u]
        x2, y2 = node_xy[v]
        lon1, lat1 = node_lonlat[u]
        lon2, lat2 = node_lonlat[v]
        support = max(float(s.get("support", 0.0) or 0.0), 1e-6)
        has_rev = (v, u) in edge_support
        edge_rows.append(
            {
                "u": int(u),
                "v": int(v),
                "support": float(s.get("support", 0.0) or 0.0),
                "weighted_support": float(s.get("weighted_support", 0.0) or 0.0),
                "effective_support": float(
                    s.get("effective_support", s.get("weighted_support", 0.0)) or 0.0
                ),
                "dyn_w_probe": float(s.get("dyn_w_probe", 0.0) or 0.0),
                "dyn_w_vpd": float(s.get("dyn_w_vpd", 1.0) or 1.0),
                "road_likeness_score": float(s.get("road_likeness_score", 0.5) or 0.5),
                "vpd_support": float(s.get("vpd_support", 0.0) or 0.0),
                "hpd_support": float(s.get("hpd_support", 0.0) or 0.0),
                "mean_step_length_m": float(s.get("length_sum", 0.0) or 0.0) / support,
                "construction_percent_mean": float(
                    s.get("construction_sum", 0.0) or 0.0
                )
                / support,
                "traffic_signal_count_mean": float(
                    s.get("traffic_signal_sum", 0.0) or 0.0
                )
                / support,
                "altitude_mean": (
                    float(s.get("altitude_sum", 0.0) or 0.0)
                    / max(float(s.get("altitude_count", 0) or 0), 1.0)
                    if float(s.get("altitude_count", 0) or 0) > 0
                    else np.nan
                ),
                "crosswalk_types": sorted(list(s.get("crosswalk_types", set()))),
                "dir_travel": "B" if has_rev else "T",
                "is_virtual_link": bool(s.get("is_virtual", False)),
                "postprocess_tags": sorted(list(s.get("postprocess_tags", set()))),
                "geometry": LineString([(lon1, lat1), (lon2, lat2)]),
            }
        )
    edges_df = (
        gpd.GeoDataFrame(edge_rows, crs="EPSG:4326")
        if edge_rows
        else pd.DataFrame(edge_rows)
    )
    print(f"  Rebuilt {len(edges_df)} edges")

    # Rebuild centerlines
    print("  Stitching centerline paths...")
    paths = stitch_centerline_paths(edge_support)
    print(f"  Found {len(paths)} centerline paths")

    min_cl_len = args.min_centerline_length_m
    smooth_iter = args.smooth_iterations
    turn_deg = args.turn_smoothing_deg
    turn_nw = args.turn_smoothing_neighbor_weight

    cl_rows = []
    skipped_short = 0
    for path_nodes in paths:
        if len(path_nodes) < 2:
            continue
        if any(n not in node_xy or n not in node_lonlat for n in path_nodes):
            continue
        raw_xy = np.asarray([node_xy[n] for n in path_nodes], dtype=np.float64)
        raw_lonlat = np.asarray([node_lonlat[n] for n in path_nodes], dtype=np.float64)

        # 1. Smooth projected XY for length calculation
        smooth_xy = smooth_polyline_preserve_turns(
            raw_xy,
            passes=max(int(smooth_iter), 0),
            turn_deg=float(turn_deg),
            neighbor_weight=float(turn_nw),
        )
        if cfg.enable_curve_smoothing:
            smooth_xy = smooth_polyline_preserve_turns(
                smooth_xy,
                passes=1,
                turn_deg=float(turn_deg),
                neighbor_weight=float(np.clip(cfg.curve_lambda * 0.5, 0.05, 0.35)),
            )

        length_m = float(
            np.sum(np.hypot(np.diff(smooth_xy[:, 0]), np.diff(smooth_xy[:, 1])))
        )
        if length_m < float(min_cl_len):
            skipped_short += 1
            continue

        # 2. Smooth Lat/Lon for output geometry
        if getattr(args, "use_spline_smoothing", False):
            # Spline smoothing (B-spline)
            # Filter duplicates to avoid splprep errors
            valid_mask = np.concatenate(
                ([True], np.any(np.diff(raw_lonlat, axis=0) != 0, axis=1))
            )
            pts = raw_lonlat[valid_mask]

            # Need at least k+1 points for spline order k
            k = 3
            if len(pts) > k:
                try:
                    # s is smoothness factor.
                    # args.spline_smooth_factor defaults to 5.0
                    # For lat/lon degrees, we need roughly 1e-9 to 1e-8 range.
                    # 1 deg ~ 111km. 1m ~ 1e-5 deg. 1m error variance ~ 1e-10.
                    # s ~ m * variance. m points.
                    # Let's use a heuristic based on point count and user factor.
                    s_val = len(pts) * (
                        getattr(args, "spline_smooth_factor", 5.0) * 1e-10
                    )

                    tck, u = splprep([pts[:, 0], pts[:, 1]], u=None, s=s_val, k=k)
                    # Upsample for smoothness
                    u_new = np.linspace(u.min(), u.max(), max(len(pts) * 3, 10))
                    result = splev(u_new, tck)
                    smooth_lonlat = np.column_stack(result)
                    line_wgs = LineString(smooth_lonlat)
                except Exception as e:
                    # print(f"    Spline failed for path {path_nodes[0]}->{path_nodes[-1]}: {e}")
                    # Fallback to simple smoothing
                    simple_smooth = smooth_polyline_preserve_turns(
                        raw_lonlat,
                        passes=2,
                        turn_deg=float(turn_deg),
                        neighbor_weight=float(turn_nw),
                    )
                    line_wgs = LineString(simple_smooth)
            else:
                line_wgs = LineString(raw_lonlat)
        else:
            # Apply same iterative smoothing to lonlat
            smooth_lonlat = smooth_polyline_preserve_turns(
                raw_lonlat,
                passes=max(int(smooth_iter), 0),
                turn_deg=float(turn_deg),
                neighbor_weight=float(turn_nw),
            )
            if cfg.enable_curve_smoothing:
                smooth_lonlat = smooth_polyline_preserve_turns(
                    smooth_lonlat,
                    passes=1,
                    turn_deg=float(turn_deg),
                    neighbor_weight=float(np.clip(cfg.curve_lambda * 0.5, 0.05, 0.35)),
                )
            line_wgs = LineString(smooth_lonlat)

        # endpoint_dist_m calculation (unchanged)

        # Aggregate edge stats along this path
        support_sum = 0.0
        weighted_sum = 0.0
        has_virtual = False
        post_tags: set[str] = set()
        fw, rv = 0.0, 0.0
        for i in range(1, len(path_nodes)):
            a, b = path_nodes[i - 1], path_nodes[i]
            if (a, b) in edge_support:
                es = edge_support[(a, b)]
                fw += float(es.get("support", 0.0) or 0.0)
                support_sum += float(es.get("support", 0.0) or 0.0)
                weighted_sum += float(es.get("weighted_support", 0.0) or 0.0)
                has_virtual = has_virtual or bool(es.get("is_virtual", False))
                post_tags.update(es.get("postprocess_tags", set()))
            if (b, a) in edge_support:
                es = edge_support[(b, a)]
                rv += float(es.get("support", 0.0) or 0.0)
                support_sum += float(es.get("support", 0.0) or 0.0)
                weighted_sum += float(es.get("weighted_support", 0.0) or 0.0)
                has_virtual = has_virtual or bool(es.get("is_virtual", False))
                post_tags.update(es.get("postprocess_tags", set()))

        if support_sum <= 0.0:
            continue
        if fw > 0 and rv > 0:
            dt = "B"
        elif fw >= rv:
            dt = "T"
        else:
            dt = "F"

        cl_rows.append(
            {
                "node_path": path_nodes,
                "support": support_sum,
                "weighted_support": weighted_sum,
                "dir_travel": dt,
                "u": int(path_nodes[0]),
                "v": int(path_nodes[-1]),
                "length_m": length_m,
                "is_virtual_link": has_virtual,
                "postprocess_tags": sorted(list(post_tags)),
                "geometry": line_wgs,
            }
        )

    cl_df = (
        gpd.GeoDataFrame(cl_rows, crs="EPSG:4326") if cl_rows else pd.DataFrame(cl_rows)
    )
    print(f"  Built {len(cl_df)} centerlines (skipped {skipped_short} short)")

    # Save outputs
    stem = args.stem
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=[Point(row.lon, row.lat) for row in nodes_df.itertuples(index=False)],
        crs="EPSG:4326",
    )

    nodes_path = out_dir / f"{stem}_nodes.gpkg"
    nodes_gdf.to_file(nodes_path, driver="GPKG", layer="nodes")
    print(f"  Saved nodes -> {nodes_path}")

    edges_path = out_dir / f"{stem}_edges.gpkg"
    if isinstance(edges_df, gpd.GeoDataFrame) and len(edges_df) > 0:
        edges_df.to_file(edges_path, driver="GPKG", layer="edges")
    print(f"  Saved edges -> {edges_path}")

    cl_path = out_dir / f"{stem}.gpkg"
    if isinstance(cl_df, gpd.GeoDataFrame) and len(cl_df) > 0:
        # node_path is a list -- convert for GPKG compatibility
        cl_save = cl_df.copy()
        cl_save["node_path"] = cl_save["node_path"].apply(json.dumps)
        cl_save["postprocess_tags"] = cl_save["postprocess_tags"].apply(json.dumps)
        cl_save.to_file(cl_path, driver="GPKG", layer="centerlines")
    print(f"  Saved centerlines -> {cl_path}")

    # Also save parquet versions
    for df, name in [(edges_df, "edges"), (cl_df, "centerlines")]:
        if len(df) == 0:
            continue
        pq_df = df.copy()
        if "geometry" in pq_df.columns:
            from shapely import wkt

            pq_df["geometry_wkt"] = pq_df["geometry"].apply(
                lambda g: g.wkt if g else ""
            )
            pq_df = pq_df.drop(columns=["geometry"])
        # Convert list/set columns to JSON strings for parquet
        for col in pq_df.columns:
            if pq_df[col].apply(lambda x: isinstance(x, (list, set))).any():
                pq_df[col] = pq_df[col].apply(
                    lambda x: json.dumps(sorted(list(x)) if isinstance(x, set) else x)
                    if isinstance(x, (list, set))
                    else x
                )
        pq_path = (
            out_dir / f"{stem}_{name}.parquet"
            if name != "centerlines"
            else out_dir / f"{stem}.parquet"
        )
        pq_df.to_parquet(pq_path, index=False)
        print(f"  Saved {name} parquet -> {pq_path}")

    # Save phase comparison summary
    summary = {
        "phases_completed": state["completed_phases"] + ["rebuild"],
        "phase_stats": {},
    }
    for phase_name, pstats in state["phase_stats"].items():
        summary["phase_stats"][phase_name] = pstats
    summary["phase_stats"]["rebuild"] = {
        "final_edge_count": len(edges_df),
        "centerline_count": len(cl_df),
        "skipped_short_centerlines": skipped_short,
    }

    summary_path = out_dir / f"{stem}_phase_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Saved phase summary -> {summary_path}")

    state["completed_phases"].append("rebuild")
    state["phase_stats"]["rebuild"] = summary["phase_stats"]["rebuild"]
    _save_state(state, out_dir)
    print(f"\n  Rebuild complete.")
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASE_FUNCS = {
    "load": phase_load,
    "virtual_links": phase_virtual_links,
    "support_refine": phase_support_refine,
    "cap_virtual": phase_cap_virtual,
    "deduplicate": phase_deduplicate,
    "simplify": phase_simplify,
    "rebuild": phase_rebuild,
}


def main():
    parser = argparse.ArgumentParser(
        description="Phasewise DeepMG postprocessing for Kharita outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available phases: {', '.join(PHASES)}\nUse --phase all to run all phases sequentially.",
    )
    parser.add_argument(
        "--phase",
        required=True,
        help=f"Which phase to run. One of: {', '.join(PHASES)}, or 'all'.",
    )
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="kharita_postprocessed")

    # DeepMG config
    parser.add_argument("--post-link-radius-m", type=float, default=None)
    parser.add_argument("--post-alpha-virtual", type=float, default=1.4)
    parser.add_argument("--post-min-supp-virtual", type=int, default=0)
    parser.add_argument("--post-pred-min-supp", type=int, default=0)
    parser.add_argument("--post-similar-direction-deg", type=float, default=20.0)
    parser.add_argument("--post-max-virtual-links-per-node", type=int, default=2)
    parser.add_argument(
        "--post-enable-duplicate-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--post-dup-eps-m", type=float, default=3.0)
    parser.add_argument(
        "--post-enable-curve-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--post-curve-lambda", type=float, default=0.2)
    parser.add_argument("--assume-sample-point-count", type=int, default=None)
    parser.add_argument("--simplify-snap-dist-m", type=float, default=4.0)

    # Centerline rebuild config
    parser.add_argument("--min-centerline-length-m", type=float, default=12.0)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--turn-smoothing-deg", type=float, default=30.0)
    parser.add_argument("--turn-smoothing-neighbor-weight", type=float, default=0.25)
    parser.add_argument(
        "--use-spline-smoothing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--spline-smooth-factor", type=float, default=5.0)

    args = parser.parse_args()

    phase = args.phase.lower()
    if phase == "all":
        for p in PHASES:
            PHASE_FUNCS[p](args)
            print()
    elif phase in PHASE_FUNCS:
        PHASE_FUNCS[phase](args)
    else:
        parser.error(f"Unknown phase '{phase}'. Choose from: {', '.join(PHASES)}, all")


if __name__ == "__main__":
    main()
