from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import transform, unary_union

from .io_utils import infer_local_projected_crs, load_navstreet_csv


def _load_line_geodata(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".gpkg", ".geojson", ".shp"}:
        gdf = gpd.read_file(path, layer=layer)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf[gdf.geometry.notnull()].copy()

    if suffix == ".csv":
        df = pd.read_csv(path)

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
        gdf = gdf[gdf.geometry.notnull()].copy()
        return gdf

    raise ValueError(f"Unsupported file type: {path}")


def _sample_points(geoms: Iterable[LineString], step_m: float, max_points: int = 500000) -> np.ndarray:
    pts = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        length = float(g.length)
        if length <= 0:
            continue
        n = max(int(length // step_m), 1)
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
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    tree = cKDTree(dst_pts)
    d, _ = tree.query(src_pts, k=1)
    return {
        "mean": float(np.nanmean(d)),
        "median": float(np.nanmedian(d)),
        "p90": float(np.nanpercentile(d, 90)),
        "p95": float(np.nanpercentile(d, 95)),
    }


def evaluate_centerlines(
    generated_centerlines: str | Path,
    ground_truth_navstreet: str | Path,
    generated_layer: str | None = None,
    ground_truth_layer: str | None = None,
    buffer_m: float = 15.0,
    sample_step_m: float = 10.0,
) -> dict:
    gen = _load_line_geodata(generated_centerlines, layer=generated_layer)

    gt_path = Path(ground_truth_navstreet)
    if gt_path.suffix.lower() == ".csv" and "nav" in gt_path.name.lower() and "geom" in pd.read_csv(gt_path, nrows=0).columns:
        gt_df = load_navstreet_csv(gt_path)
        gt = gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")
    else:
        gt = _load_line_geodata(ground_truth_navstreet, layer=ground_truth_layer)

    if gen.empty or gt.empty:
        return {
            "generated_count": int(len(gen)),
            "ground_truth_count": int(len(gt)),
            "error": "One or both datasets are empty.",
        }

    proj_crs = infer_local_projected_crs(list(gen.geometry) + list(gt.geometry))
    to_proj = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)

    gen_proj = gen.copy()
    gen_proj["geometry"] = gen_proj.geometry.map(lambda g: transform(to_proj.transform, g))
    gen_proj = gen_proj.set_crs(proj_crs, allow_override=True)

    gt_proj = gt.copy()
    gt_proj["geometry"] = gt_proj.geometry.map(lambda g: transform(to_proj.transform, g))
    gt_proj = gt_proj.set_crs(proj_crs, allow_override=True)

    gen_proj = gen_proj[gen_proj.geometry.notnull()].copy()
    gt_proj = gt_proj[gt_proj.geometry.notnull()].copy()

    gen_total_len = float(gen_proj.length.sum())
    gt_total_len = float(gt_proj.length.sum())

    gt_buffer_union = unary_union(gt_proj.buffer(buffer_m).to_list())
    gen_buffer_union = unary_union(gen_proj.buffer(buffer_m).to_list())

    matched_gen_len = float(gen_proj.geometry.intersection(gt_buffer_union).length.sum())
    matched_gt_len = float(gt_proj.geometry.intersection(gen_buffer_union).length.sum())

    precision = (matched_gen_len / gen_total_len) if gen_total_len > 0 else float("nan")
    recall = (matched_gt_len / gt_total_len) if gt_total_len > 0 else float("nan")
    if precision > 0 and recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

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
        "generated_to_groundtruth_distance_m": gen_to_gt,
        "groundtruth_to_generated_distance_m": gt_to_gen,
    }

    return metrics


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
