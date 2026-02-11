from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.generation import save_centerline_outputs
from centerline.postprocess_deepmg_kharita import (
    DeepMGPostprocessConfig,
    run_deepmg_kharita_postprocess,
)
from centerline.utils import smooth_polyline_preserve_turns, stitch_centerline_paths


def _load_layer(path: Path, layer: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".gpkg":
        return gpd.read_file(path, layer=layer)
    raise ValueError(f"Unsupported input format for {path}. Use .parquet or .gpkg")


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


def _load_nodes(nodes_path: Path) -> pd.DataFrame:
    df = _load_layer(nodes_path, layer="nodes").copy()
    required = {"node_id", "x", "y", "lon", "lat"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Nodes input missing required columns: {missing}")
    for c in ["node_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["x", "y", "lon", "lat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["node_id", "x", "y", "lon", "lat"]).copy()
    df["node_id"] = df["node_id"].astype(int)
    return df


def _load_edges(edges_path: Path) -> pd.DataFrame:
    df = _load_layer(edges_path, layer="edges").copy()
    required = {"u", "v", "support", "weighted_support"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Edges input missing required columns: {missing}")

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
        u = int(r.u)
        v = int(r.v)
        support = float(getattr(r, "support", 0.0) or 0.0)
        weighted = float(getattr(r, "weighted_support", support) or 0.0)
        mean_step = float(getattr(r, "mean_step_length_m", 0.0) or 0.0)
        length_sum = max(0.0, mean_step) * max(support, 0.0)
        altitude_mean = getattr(r, "altitude_mean", np.nan)

        s = {
            "support": support,
            "weighted_support": weighted,
            "effective_support": float(getattr(r, "effective_support", weighted) or weighted),
            "vpd_support": float(getattr(r, "vpd_support", 0.0) or 0.0),
            "hpd_support": float(getattr(r, "hpd_support", 0.0) or 0.0),
            "length_sum": length_sum,
            "construction_sum": float(getattr(r, "construction_percent_mean", 0.0) or 0.0) * max(support, 0.0),
            "traffic_signal_sum": float(getattr(r, "traffic_signal_count_mean", 0.0) or 0.0) * max(support, 0.0),
            "altitude_sum": float(altitude_mean) * max(support, 0.0) if np.isfinite(altitude_mean) else 0.0,
            "altitude_count": int(max(round(support), 0)) if np.isfinite(altitude_mean) else 0,
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
            "postprocess_tags": set(_parse_list_like(getattr(r, "postprocess_tags", []))),
            "dyn_w_probe": float(getattr(r, "dyn_w_probe", 0.0) or 0.0),
            "dyn_w_vpd": float(getattr(r, "dyn_w_vpd", 1.0) or 1.0),
            "road_likeness_score": float(getattr(r, "road_likeness_score", 0.5) or 0.5),
        }
        edge_support[(u, v)] = s
    return edge_support


def _rebuild_edges_df(
    *,
    edge_support: dict[tuple[int, int], dict],
    node_xy: dict[int, tuple[float, float]],
    node_lonlat: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for (u, v), s in edge_support.items():
        if u not in node_xy or v not in node_xy:
            continue
        x1, y1 = node_xy[u]
        x2, y2 = node_xy[v]
        lon1, lat1 = node_lonlat[u]
        lon2, lat2 = node_lonlat[v]

        support = max(float(s.get("support", 0.0) or 0.0), 1e-6)
        has_rev = (v, u) in edge_support
        rows.append(
            {
                "u": int(u),
                "v": int(v),
                "support": float(s.get("support", 0.0) or 0.0),
                "weighted_support": float(s.get("weighted_support", 0.0) or 0.0),
                "effective_support": float(s.get("effective_support", s.get("weighted_support", 0.0)) or 0.0),
                "dyn_w_probe": float(s.get("dyn_w_probe", 0.0) or 0.0),
                "dyn_w_vpd": float(s.get("dyn_w_vpd", 1.0) or 1.0),
                "road_likeness_score": float(s.get("road_likeness_score", 0.5) or 0.5),
                "vpd_support": float(s.get("vpd_support", 0.0) or 0.0),
                "hpd_support": float(s.get("hpd_support", 0.0) or 0.0),
                "mean_step_length_m": float(s.get("length_sum", 0.0) or 0.0) / support,
                "construction_percent_mean": float(s.get("construction_sum", 0.0) or 0.0) / support,
                "traffic_signal_count_mean": float(s.get("traffic_signal_sum", 0.0) or 0.0) / support,
                "altitude_mean": (
                    float(s.get("altitude_sum", 0.0) or 0.0) / max(float(s.get("altitude_count", 0) or 0), 1.0)
                    if float(s.get("altitude_count", 0) or 0) > 0
                    else np.nan
                ),
                "crosswalk_types": sorted(list(s.get("crosswalk_types", set()))),
                "day_mode": None,
                "hour_mode": None,
                "dir_travel": "B" if has_rev else "T",
                "is_virtual_link": bool(s.get("is_virtual", False)),
                "postprocess_tags": sorted(list(s.get("postprocess_tags", set()))),
                "geometry": LineString([(lon1, lat1), (lon2, lat2)]),
                "_length_m_xy": float(np.hypot(x2 - x1, y2 - y1)),
            }
        )
    return pd.DataFrame(rows)


def _rebuild_centerlines_df(
    *,
    edge_support: dict[tuple[int, int], dict],
    node_xy: dict[int, tuple[float, float]],
    node_lonlat: dict[int, tuple[float, float]],
    min_centerline_length_m: float,
    smooth_iterations: int,
    turn_smoothing_deg: float,
    turn_smoothing_neighbor_weight: float,
    enable_curve_smoothing: bool,
    curve_lambda: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    paths = stitch_centerline_paths(edge_support)

    for path_nodes in paths:
        if len(path_nodes) < 2:
            continue
        if any((n not in node_xy or n not in node_lonlat) for n in path_nodes):
            continue

        raw_xy = np.asarray([node_xy[n] for n in path_nodes], dtype=np.float64)
        smooth_xy = smooth_polyline_preserve_turns(
            raw_xy,
            passes=max(int(smooth_iterations), 0),
            turn_deg=float(turn_smoothing_deg),
            neighbor_weight=float(turn_smoothing_neighbor_weight),
        )
        if enable_curve_smoothing:
            smooth_xy = smooth_polyline_preserve_turns(
                smooth_xy,
                passes=1,
                turn_deg=float(turn_smoothing_deg),
                neighbor_weight=float(np.clip(curve_lambda * 0.5, 0.05, 0.35)),
            )

        length_m = float(
            np.sum(np.hypot(np.diff(smooth_xy[:, 0]), np.diff(smooth_xy[:, 1])))
        )
        if length_m < float(min_centerline_length_m):
            continue

        lonlat = [node_lonlat[n] for n in path_nodes]
        line_wgs = LineString(lonlat)
        endpoint_dist_m = float(
            np.hypot(smooth_xy[-1, 0] - smooth_xy[0, 0], smooth_xy[-1, 1] - smooth_xy[0, 1])
        )

        support_sum = 0.0
        weighted_support_sum = 0.0
        effective_support_sum = 0.0
        has_virtual = False
        post_tags: set[str] = set()
        dyn_probe_wsum = 0.0
        dyn_vpd_wsum = 0.0
        road_like_wsum = 0.0
        dyn_w = 0.0

        fw = 0.0
        rv = 0.0
        for i in range(1, len(path_nodes)):
            a = path_nodes[i - 1]
            b = path_nodes[i]
            if (a, b) in edge_support:
                es = edge_support[(a, b)]
                fw += float(es.get("support", 0.0) or 0.0)
                support_sum += float(es.get("support", 0.0) or 0.0)
                weighted_support_sum += float(es.get("weighted_support", 0.0) or 0.0)
                eff = float(es.get("effective_support", es.get("weighted_support", 0.0)) or 0.0)
                effective_support_sum += eff
                w_eff = max(eff, 1e-6)
                dyn_probe_wsum += float(es.get("dyn_w_probe", 0.0) or 0.0) * w_eff
                dyn_vpd_wsum += float(es.get("dyn_w_vpd", 1.0) or 1.0) * w_eff
                road_like_wsum += float(es.get("road_likeness_score", 0.5) or 0.5) * w_eff
                dyn_w += w_eff
                has_virtual = has_virtual or bool(es.get("is_virtual", False))
                post_tags.update(es.get("postprocess_tags", set()))
            if (b, a) in edge_support:
                es = edge_support[(b, a)]
                rv += float(es.get("support", 0.0) or 0.0)
                support_sum += float(es.get("support", 0.0) or 0.0)
                weighted_support_sum += float(es.get("weighted_support", 0.0) or 0.0)
                eff = float(es.get("effective_support", es.get("weighted_support", 0.0)) or 0.0)
                effective_support_sum += eff
                w_eff = max(eff, 1e-6)
                dyn_probe_wsum += float(es.get("dyn_w_probe", 0.0) or 0.0) * w_eff
                dyn_vpd_wsum += float(es.get("dyn_w_vpd", 1.0) or 1.0) * w_eff
                road_like_wsum += float(es.get("road_likeness_score", 0.5) or 0.5) * w_eff
                dyn_w += w_eff
                has_virtual = has_virtual or bool(es.get("is_virtual", False))
                post_tags.update(es.get("postprocess_tags", set()))

        if support_sum <= 0.0:
            continue

        if fw > 0 and rv > 0:
            dir_travel = "B"
        elif fw >= rv:
            dir_travel = "T"
        else:
            dir_travel = "F"

        rows.append(
            {
                "node_path": path_nodes,
                "support": support_sum,
                "weighted_support": weighted_support_sum,
                "effective_support": effective_support_sum,
                "dyn_w_probe": float(dyn_probe_wsum / max(dyn_w, 1e-6)),
                "dyn_w_vpd": float(dyn_vpd_wsum / max(dyn_w, 1e-6)),
                "road_likeness_score": float(road_like_wsum / max(dyn_w, 1e-6)),
                "dir_travel": dir_travel,
                "u": int(path_nodes[0]),
                "v": int(path_nodes[-1]),
                "length_m": length_m,
                "endpoint_dist_m": endpoint_dist_m,
                "is_virtual_link": bool(has_virtual),
                "postprocess_tags": sorted(list(post_tags)),
                "geometry": line_wgs,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply DeepMG-style topology postprocess to existing Kharita nodes/edges "
            "outputs and rebuild centerlines without regenerating traces."
        )
    )
    parser.add_argument("--nodes", type=Path, required=True, help="Nodes output (.gpkg or .parquet).")
    parser.add_argument("--edges", type=Path, required=True, help="Edges output (.gpkg or .parquet).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="kharita_postprocessed")

    parser.add_argument("--post-link-radius-m", type=float, default=None)
    parser.add_argument("--post-alpha-virtual", type=float, default=1.4)
    parser.add_argument(
        "--post-min-supp-virtual",
        type=int,
        default=0,
        help=(
            "Default is 0 for standalone postprocess (no trace-level support refinement "
            "data is available after generation)."
        ),
    )
    parser.add_argument(
        "--post-pred-min-supp",
        type=int,
        default=0,
        help="Minimum support for existing edges during standalone postprocess.",
    )
    parser.add_argument("--post-similar-direction-deg", type=float, default=20.0)
    parser.add_argument("--post-max-virtual-links-per-node", type=int, default=2)
    parser.add_argument(
        "--post-enable-duplicate-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--post-dup-eps-m", type=float, default=3.0)
    parser.add_argument(
        "--assume-sample-point-count",
        type=int,
        default=None,
        help="Optional override used by autoscaling logic; defaults to estimated value from edge supports.",
    )

    parser.add_argument("--min-centerline-length-m", type=float, default=12.0)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--turn-smoothing-deg", type=float, default=30.0)
    parser.add_argument("--turn-smoothing-neighbor-weight", type=float, default=0.25)
    parser.add_argument(
        "--post-enable-curve-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--post-curve-lambda", type=float, default=0.2)
    args = parser.parse_args()

    nodes = _load_nodes(args.nodes)
    edges = _load_edges(args.edges)

    node_xy = {
        int(r.node_id): (float(r.x), float(r.y))
        for r in nodes.itertuples(index=False)
    }
    node_lonlat = {
        int(r.node_id): (float(r.lon), float(r.lat))
        for r in nodes.itertuples(index=False)
    }

    edge_support = _build_edge_support(edges)
    estimated_sample_point_count = int(
        max(1.0, sum(float(s.get("support", 0.0) or 0.0) for s in edge_support.values()))
    )
    sample_point_count = (
        int(args.assume_sample_point_count)
        if args.assume_sample_point_count is not None
        else estimated_sample_point_count
    )

    cfg = DeepMGPostprocessConfig(
        enabled=True,
        link_radius_m=args.post_link_radius_m,
        alpha_virtual=float(args.post_alpha_virtual),
        min_supp_virtual=int(args.post_min_supp_virtual),
        pred_min_supp=int(args.post_pred_min_supp),
        similar_direction_deg=float(args.post_similar_direction_deg),
        max_virtual_links_per_node=int(args.post_max_virtual_links_per_node),
        enable_duplicate_merge=bool(args.post_enable_duplicate_merge),
        dup_eps_m=float(args.post_dup_eps_m),
        enable_curve_smoothing=bool(args.post_enable_curve_smoothing),
        curve_lambda=float(args.post_curve_lambda),
    )

    # Standalone mode: we do not have trace labels/ranges post-generation.
    post_meta = run_deepmg_kharita_postprocess(
        edge_support=edge_support,
        node_xy=node_xy,
        labels=np.asarray([], dtype=np.int32),
        trace_ranges={},
        sample_point_count=sample_point_count,
        config=cfg,
    )

    edges_df = _rebuild_edges_df(
        edge_support=edge_support,
        node_xy=node_xy,
        node_lonlat=node_lonlat,
    )
    if "_length_m_xy" in edges_df.columns:
        edges_df = edges_df.drop(columns=["_length_m_xy"])

    centerlines_df = _rebuild_centerlines_df(
        edge_support=edge_support,
        node_xy=node_xy,
        node_lonlat=node_lonlat,
        min_centerline_length_m=float(args.min_centerline_length_m),
        smooth_iterations=int(args.smooth_iterations),
        turn_smoothing_deg=float(args.turn_smoothing_deg),
        turn_smoothing_neighbor_weight=float(args.turn_smoothing_neighbor_weight),
        enable_curve_smoothing=bool(args.post_enable_curve_smoothing),
        curve_lambda=float(args.post_curve_lambda),
    )

    result = {
        "projected_crs": "standalone_postprocess",
        "nodes": nodes.copy(),
        "edges": edges_df,
        "centerlines": centerlines_df,
        "trace_count": 0,
        "sample_point_count": sample_point_count,
    }

    files = save_centerline_outputs(
        result=result,
        output_dir=args.out_dir,
        stem=args.stem,
    )

    meta_path = Path(args.out_dir) / f"{args.stem}_postprocess_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "mode": "standalone_deepmg_postprocess",
                "note": (
                    "Trace-level labels/ranges are unavailable post-generation. "
                    "Support refinement uses empty traces."
                ),
                "input_nodes": str(args.nodes),
                "input_edges": str(args.edges),
                "estimated_sample_point_count": estimated_sample_point_count,
                "sample_point_count_used": sample_point_count,
                "postprocess_meta": post_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Postprocess completed.")
    for k, v in files.items():
        print(f"{k}: {v}")
    print(f"postprocess_meta: {meta_path}")


if __name__ == "__main__":
    main()
