from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkt
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from .io_utils import (
    clip_line_geometries_to_bbox,
    infer_local_projected_crs,
    load_bbox_from_txt,
    load_navstreet_csv,
)


def _table_columns(path: str | Path) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p).columns.tolist()
    return pd.read_csv(p, nrows=0).columns.tolist()


def _load_line_geodata(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".gpkg", ".geojson", ".shp"}:
        gdf = gpd.read_file(path, layer=layer)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf[gdf.geometry.notnull()].copy()

    if suffix in {".csv", ".parquet"}:
        df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
        if "geometry_wkt" in df.columns:
            geom_col = "geometry_wkt"
        elif "geom" in df.columns:
            geom_col = "geom"
        elif "geometry" in df.columns:
            geom_col = "geometry"
        else:
            raise ValueError(f"Could not find WKT geometry column in {path}")

        geoms = []
        for s in df[geom_col].astype(str):
            try:
                geoms.append(wkt.loads(s))
            except Exception:
                geoms.append(None)
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
        return gdf[gdf.geometry.notnull()].copy()

    raise ValueError(f"Unsupported file type: {path}")


def _sample_points(geoms: Iterable[LineString], step_m: float, max_points: int = 500000) -> np.ndarray:
    pts = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        length = float(g.length)
        if length <= 0:
            continue
        n = max(int(length // max(step_m, 0.5)), 1)
        for d in np.linspace(0.0, length, n + 1):
            p = g.interpolate(float(d))
            pts.append((float(p.x), float(p.y)))
            if len(pts) >= max_points:
                return np.asarray(pts, dtype=np.float32)
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _nn_stats(src_pts: np.ndarray, dst_pts: np.ndarray) -> Dict[str, float]:
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    tree = cKDTree(dst_pts)
    d, _ = tree.query(src_pts, k=1)
    return {
        "mean": float(np.nanmean(d)),
        "median": float(np.nanmedian(d)),
        "p90": float(np.nanpercentile(d, 90)),
        "p95": float(np.nanpercentile(d, 95)),
    }


def _infer_graph(gdf_proj: gpd.GeoDataFrame, snap_m: float = 8.0) -> tuple[Dict[int, tuple[float, float]], list[tuple[int, int]]]:
    """Build a graph from line features, using explicit ids if available."""
    if gdf_proj.empty:
        return {}, []

    use_uv = "u" in gdf_proj.columns and "v" in gdf_proj.columns
    use_ref = "ref_node_pvid" in gdf_proj.columns and "nref_node_pvid" in gdf_proj.columns

    nodes_acc: Dict[int, list[tuple[float, float]]] = {}
    edges: list[tuple[int, int]] = []

    if use_uv or use_ref:
        for r in gdf_proj.itertuples(index=False):
            line = r.geometry
            if line is None or line.is_empty:
                continue
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            if use_uv:
                u = int(getattr(r, "u"))
                v = int(getattr(r, "v"))
            else:
                u = int(getattr(r, "ref_node_pvid"))
                v = int(getattr(r, "nref_node_pvid"))
            edges.append((u, v))
            nodes_acc.setdefault(u, []).append((float(coords[0][0]), float(coords[0][1])))
            nodes_acc.setdefault(v, []).append((float(coords[-1][0]), float(coords[-1][1])))

        nodes = {
            nid: (
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])),
            )
            for nid, pts in nodes_acc.items()
        }
        return nodes, edges

    # Endpoint snapping fallback.
    node_coords: list[tuple[float, float]] = []
    edges_out: list[tuple[int, int]] = []
    tree: cKDTree | None = None

    def get_or_create_node(x: float, y: float) -> int:
        nonlocal tree
        if tree is not None and node_coords:
            cand = tree.query_ball_point([x, y], r=snap_m)
            if cand:
                return int(cand[0])
        node_coords.append((x, y))
        tree = cKDTree(np.asarray(node_coords, dtype=np.float64))
        return len(node_coords) - 1

    for r in gdf_proj.itertuples(index=False):
        line = r.geometry
        if line is None or line.is_empty:
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        a = get_or_create_node(float(coords[0][0]), float(coords[0][1]))
        b = get_or_create_node(float(coords[-1][0]), float(coords[-1][1]))
        if a != b:
            edges_out.append((a, b))

    nodes = {i: xy for i, xy in enumerate(node_coords)}
    return nodes, edges_out


def _degree_hist_similarity(gen_edges: list[tuple[int, int]], gt_edges: list[tuple[int, int]]) -> dict:
    def hist(edges: list[tuple[int, int]]) -> np.ndarray:
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        if not deg:
            return np.zeros(1, dtype=np.float64)
        max_d = max(deg.values())
        h = np.zeros(max_d + 1, dtype=np.float64)
        for d in deg.values():
            h[d] += 1.0
        if h.sum() > 0:
            h /= h.sum()
        return h

    p = hist(gen_edges)
    q = hist(gt_edges)
    mlen = max(len(p), len(q))
    p = np.pad(p, (0, mlen - len(p)))
    q = np.pad(q, (0, mlen - len(q)))
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    js_norm = js / np.log(2.0) if js > 0 else 0.0
    similarity = float(max(0.0, 1.0 - js_norm))
    return {
        "degree_js_divergence": float(js_norm),
        "degree_similarity": similarity,
    }


def _intersection_error(
        gen_nodes: Dict[int, tuple[float, float]],
        gen_edges: list[tuple[int, int]],
        gt_nodes: Dict[int, tuple[float, float]],
        gt_edges: list[tuple[int, int]],
) -> dict:
    def intersection_pts(nodes: Dict[int, tuple[float, float]], edges: list[tuple[int, int]]) -> np.ndarray:
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        pts = [nodes[n] for n, d in deg.items() if d >= 3 and n in nodes]
        if not pts:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    g = intersection_pts(gen_nodes, gen_edges)
    t = intersection_pts(gt_nodes, gt_edges)
    if len(g) == 0 or len(t) == 0:
        return {
            "generated_intersection_count": int(len(g)),
            "ground_truth_intersection_count": int(len(t)),
            "generated_to_gt_m": {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")},
            "gt_to_generated_m": {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")},
        }
    return {
        "generated_intersection_count": int(len(g)),
        "ground_truth_intersection_count": int(len(t)),
        "generated_to_gt_m": _nn_stats(g, t),
        "gt_to_generated_m": _nn_stats(t, g),
    }


def _hausdorff_summary(gen_proj: gpd.GeoDataFrame, gt_proj: gpd.GeoDataFrame, max_pairs: int = 2000) -> dict:
    if gen_proj.empty or gt_proj.empty:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "count": 0}

    gt_index = gt_proj.sindex
    dvals: list[float] = []
    checked = 0
    for r in gen_proj.itertuples(index=False):
        if checked >= max_pairs:
            break
        g = r.geometry
        if g is None or g.is_empty:
            continue
        bounds = box(*g.bounds).buffer(30.0).bounds
        cand_idx = list(gt_index.intersection(bounds))
        if not cand_idx:
            continue
        best = float("inf")
        for j in cand_idx:
            d = float(g.hausdorff_distance(gt_proj.iloc[j].geometry))
            if d < best:
                best = d
        if np.isfinite(best):
            dvals.append(best)
            checked += 1

    if not dvals:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "count": 0}
    arr = np.asarray(dvals, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "p90": float(np.nanpercentile(arr, 90)),
        "count": int(len(arr)),
    }


def evaluate_centerline_geodataframes(
        generated_gdf: gpd.GeoDataFrame,
        ground_truth_gdf: gpd.GeoDataFrame,
        buffer_m: float = 15.0,
        sample_step_m: float = 10.0,
        bbox_file: str | Path | None = None,
        apply_bbox: bool = False,
        clip_generated_to_ground_truth: bool = False,
        clip_buffer_m: float = 0.0,
        compute_topology_metrics: bool = True,
        compute_hausdorff: bool = False,
) -> dict:
    gen = generated_gdf.copy()
    gt = ground_truth_gdf.copy()

    if gen.crs is None:
        gen = gen.set_crs("EPSG:4326")
    if gt.crs is None:
        gt = gt.set_crs("EPSG:4326")

    gen = gen[gen.geometry.notnull()].copy()
    gt = gt[gt.geometry.notnull()].copy()

    if apply_bbox and bbox_file is not None:
        bbox = load_bbox_from_txt(bbox_file)
        gen = clip_line_geometries_to_bbox(gen, bbox_wgs84=bbox, geometry_col="geometry")
        gt = clip_line_geometries_to_bbox(gt, bbox_wgs84=bbox, geometry_col="geometry")

    if gen.empty or gt.empty:
        return {
            "generated_count": int(len(gen)),
            "ground_truth_count": int(len(gt)),
            "error": "One or both datasets are empty.",
        }

    if str(gen.crs).upper() != "EPSG:4326":
        gen = gen.to_crs("EPSG:4326")
    if str(gt.crs).upper() != "EPSG:4326":
        gt = gt.to_crs("EPSG:4326")

    proj_crs = infer_local_projected_crs(list(gen.geometry) + list(gt.geometry))
    gen_proj = gen.to_crs(proj_crs)
    gt_proj = gt.to_crs(proj_crs)

    gen_proj = gen_proj[gen_proj.geometry.notnull()].copy()
    gt_proj = gt_proj[gt_proj.geometry.notnull()].copy()

    if clip_generated_to_ground_truth and not gt_proj.empty:
        minx, miny, maxx, maxy = gt_proj.total_bounds
        clip_poly = box(minx, miny, maxx, maxy).buffer(float(clip_buffer_m))
        gen_proj["geometry"] = gen_proj.geometry.map(lambda g: g.intersection(clip_poly) if g is not None else None)
        gen_proj = gen_proj[gen_proj.geometry.notnull()].copy()
        gen_proj = gen_proj[~gen_proj.geometry.is_empty].copy()

    gen_total_len = float(gen_proj.length.sum())
    gt_total_len = float(gt_proj.length.sum())

    gt_buffer_union = unary_union(gt_proj.buffer(buffer_m).to_list())
    gen_buffer_union = unary_union(gen_proj.buffer(buffer_m).to_list())

    matched_gen_len = float(gen_proj.geometry.intersection(gt_buffer_union).length.sum())
    matched_gt_len = float(gt_proj.geometry.intersection(gen_buffer_union).length.sum())

    precision = (matched_gen_len / gen_total_len) if gen_total_len > 0 else float("nan")
    recall = (matched_gt_len / gt_total_len) if gt_total_len > 0 else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0.0

    gen_pts = _sample_points(gen_proj.geometry, step_m=sample_step_m)
    gt_pts = _sample_points(gt_proj.geometry, step_m=sample_step_m)
    gen_to_gt = _nn_stats(gen_pts, gt_pts)
    gt_to_gen = _nn_stats(gt_pts, gen_pts)

    metrics = {
        "projected_crs": str(proj_crs),
        "generated_count": int(len(gen_proj)),
        "ground_truth_count": int(len(gt_proj)),
        "generated_total_length_m": gen_total_len,
        "ground_truth_total_length_m": gt_total_len,
        "matched_generated_length_m": matched_gen_len,
        "matched_ground_truth_length_m": matched_gt_len,
        "length_precision": precision,
        "length_recall": recall,
        "length_f1": f1,
        "buffer_m": float(buffer_m),
        "sample_step_m": float(sample_step_m),
        "apply_bbox": bool(apply_bbox),
        "bbox_file": str(bbox_file) if bbox_file is not None else None,
        "clip_generated_to_ground_truth": bool(clip_generated_to_ground_truth),
        "clip_buffer_m": float(clip_buffer_m),
        "generated_to_groundtruth_distance_m": gen_to_gt,
        "groundtruth_to_generated_distance_m": gt_to_gen,
    }

    if compute_topology_metrics:
        gen_nodes, gen_edges = _infer_graph(gen_proj)
        gt_nodes, gt_edges = _infer_graph(gt_proj)
        metrics["topology"] = {
            **_degree_hist_similarity(gen_edges, gt_edges),
            "intersection_location_error_m": _intersection_error(gen_nodes, gen_edges, gt_nodes, gt_edges),
            "generated_node_count": int(len(gen_nodes)),
            "ground_truth_node_count": int(len(gt_nodes)),
            "generated_edge_count": int(len(gen_edges)),
            "ground_truth_edge_count": int(len(gt_edges)),
        }

    if compute_hausdorff:
        metrics["hausdorff_m"] = _hausdorff_summary(gen_proj, gt_proj)

    return metrics


def evaluate_centerlines(
        generated_centerlines: str | Path,
        ground_truth_navstreet: str | Path,
        generated_layer: str | None = None,
        ground_truth_layer: str | None = None,
        buffer_m: float = 15.0,
        sample_step_m: float = 10.0,
        bbox_file: str | Path | None = None,
        apply_bbox: bool = False,
        clip_generated_to_ground_truth: bool = False,
        clip_buffer_m: float = 0.0,
        compute_topology_metrics: bool = True,
        compute_hausdorff: bool = False,
) -> dict:
    gen = _load_line_geodata(generated_centerlines, layer=generated_layer)

    gt_path = Path(ground_truth_navstreet)
    gt_cols = _table_columns(gt_path) if gt_path.suffix.lower() in {".csv", ".parquet"} else []
    if gt_path.suffix.lower() in {".csv", ".parquet"} and "nav" in gt_path.name.lower() and "geom" in gt_cols:
        gt_df = load_navstreet_csv(gt_path)
        gt = gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")
    else:
        gt = _load_line_geodata(ground_truth_navstreet, layer=ground_truth_layer)

    return evaluate_centerline_geodataframes(
        generated_gdf=gen,
        ground_truth_gdf=gt,
        buffer_m=buffer_m,
        sample_step_m=sample_step_m,
        bbox_file=bbox_file,
        apply_bbox=apply_bbox,
        clip_generated_to_ground_truth=clip_generated_to_ground_truth,
        clip_buffer_m=clip_buffer_m,
        compute_topology_metrics=compute_topology_metrics,
        compute_hausdorff=compute_hausdorff,
    )


def generate_evaluation_plots(
        generated_centerlines: str | Path,
        ground_truth_navstreet: str | Path,
        out_dir: str | Path,
        generated_layer: str | None = None,
        ground_truth_layer: str | None = None,
        buffer_m: float = 15.0,
        stem: str = "centerline_eval",
) -> dict:
    import matplotlib.pyplot as plt

    gen = _load_line_geodata(generated_centerlines, layer=generated_layer)
    gt_path = Path(ground_truth_navstreet)
    gt_cols = _table_columns(gt_path) if gt_path.suffix.lower() in {".csv", ".parquet"} else []
    if gt_path.suffix.lower() in {".csv", ".parquet"} and "geom" in gt_cols:
        gt_df = load_navstreet_csv(gt_path)
        gt = gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")
    else:
        gt = _load_line_geodata(ground_truth_navstreet, layer=ground_truth_layer)

    if gen.crs is None:
        gen = gen.set_crs("EPSG:4326")
    if gt.crs is None:
        gt = gt.set_crs("EPSG:4326")
    gen = gen.to_crs("EPSG:4326")
    gt = gt.to_crs("EPSG:4326")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overlay map.
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    gt.plot(ax=ax, linewidth=1.0, color="#1f77b4", alpha=0.6, label="Ground Truth")
    gen.plot(ax=ax, linewidth=1.0, color="#d62728", alpha=0.6, label="Generated")
    ax.set_title("Centerline Overlay")
    ax.set_axis_off()
    ax.legend(loc="lower left")
    overlay_path = out_dir / f"{stem}_overlay.png"
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)

    # Unmatched segments.
    proj = infer_local_projected_crs(list(gen.geometry) + list(gt.geometry))
    gen_p = gen.to_crs(proj)
    gt_p = gt.to_crs(proj)
    gt_buffer = unary_union(gt_p.buffer(buffer_m).to_list())
    gen_buffer = unary_union(gen_p.buffer(buffer_m).to_list())

    gen_unmatched = gen_p.copy()
    gen_unmatched["geometry"] = gen_unmatched.geometry.map(lambda g: g.difference(gt_buffer))
    gen_unmatched = gen_unmatched[~gen_unmatched.geometry.is_empty].copy()
    gen_unmatched = gen_unmatched[gen_unmatched.geometry.notnull()].copy()

    gt_unmatched = gt_p.copy()
    gt_unmatched["geometry"] = gt_unmatched.geometry.map(lambda g: g.difference(gen_buffer))
    gt_unmatched = gt_unmatched[~gt_unmatched.geometry.is_empty].copy()
    gt_unmatched = gt_unmatched[gt_unmatched.geometry.notnull()].copy()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    if not gt_unmatched.empty:
        gt_unmatched.to_crs("EPSG:4326").plot(ax=ax, linewidth=1.2, color="#1f77b4", alpha=0.8, label="GT Unmatched")
    if not gen_unmatched.empty:
        gen_unmatched.to_crs("EPSG:4326").plot(ax=ax, linewidth=1.2, color="#d62728", alpha=0.8, label="Generated Unmatched")
    ax.set_title("Unmatched Segments")
    ax.set_axis_off()
    ax.legend(loc="lower left")
    mismatch_path = out_dir / f"{stem}_mismatch.png"
    fig.savefig(mismatch_path, bbox_inches="tight")
    plt.close(fig)

    return {"overlay_plot": str(overlay_path), "mismatch_plot": str(mismatch_path)}


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    flat = pd.json_normalize(metrics, sep=".")
    flat.to_parquet(output_path.with_suffix(".parquet"), index=False)
