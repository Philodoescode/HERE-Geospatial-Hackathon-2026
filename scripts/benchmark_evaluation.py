from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.ops import unary_union

from centerline.evaluation import (
    build_evaluation_context,
    evaluate_centerline_geodataframes,
)
from centerline.io_utils import clip_line_geometries_to_bbox, infer_local_projected_crs, load_bbox_from_txt, load_navstreet_csv


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
        if "geom" in df.columns and "nav" in path.name.lower():
            gt_df = load_navstreet_csv(path)
            return gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")

        geoms = []
        for s in df[geom_col].astype(str):
            try:
                geoms.append(wkt.loads(s))
            except Exception:
                geoms.append(None)
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
        return gdf[gdf.geometry.notnull()].copy()

    raise ValueError(f"Unsupported file type: {path}")


def _legacy_sample_points(geoms, step_m: float) -> np.ndarray:
    pts = []
    step = max(float(step_m), 0.5)
    for g in geoms:
        if g is None or g.is_empty:
            continue
        length = float(g.length)
        if length <= 0:
            continue
        n = max(int(length // step), 1)
        for d in np.linspace(0.0, length, n + 1):
            p = g.interpolate(float(d))
            pts.append((float(p.x), float(p.y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _legacy_eval(
    generated_gdf: gpd.GeoDataFrame,
    ground_truth_gdf: gpd.GeoDataFrame,
    *,
    buffer_m: float,
    sample_step_m: float,
    bbox_file: str | Path | None,
    apply_bbox: bool,
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

    if str(gen.crs).upper() != "EPSG:4326":
        gen = gen.to_crs("EPSG:4326")
    if str(gt.crs).upper() != "EPSG:4326":
        gt = gt.to_crs("EPSG:4326")

    proj_crs = infer_local_projected_crs(list(gen.geometry) + list(gt.geometry))
    gen_proj = gen.to_crs(proj_crs)
    gt_proj = gt.to_crs(proj_crs)

    gen_total_len = float(gen_proj.length.sum())
    gt_total_len = float(gt_proj.length.sum())

    gt_buffer_union = unary_union(gt_proj.buffer(buffer_m).to_list())
    gen_buffer_union = unary_union(gen_proj.buffer(buffer_m).to_list())

    matched_gen_len = float(gen_proj.geometry.intersection(gt_buffer_union).length.sum())
    matched_gt_len = float(gt_proj.geometry.intersection(gen_buffer_union).length.sum())

    precision = (matched_gen_len / gen_total_len) if gen_total_len > 0 else float("nan")
    recall = (matched_gt_len / gt_total_len) if gt_total_len > 0 else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0.0

    gen_pts = _legacy_sample_points(gen_proj.geometry, sample_step_m)
    gt_pts = _legacy_sample_points(gt_proj.geometry, sample_step_m)

    return {
        "generated_count": int(len(gen_proj)),
        "ground_truth_count": int(len(gt_proj)),
        "length_precision": float(precision),
        "length_recall": float(recall),
        "length_f1": float(f1),
        "generated_sample_count": int(len(gen_pts)),
        "ground_truth_sample_count": int(len(gt_pts)),
    }


def _timed(fn, repeat: int) -> tuple[float, list[dict]]:
    out = []
    t0 = time.perf_counter()
    for _ in range(repeat):
        out.append(fn())
    elapsed = time.perf_counter() - t0
    return elapsed / max(repeat, 1), out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark old-style vs new evaluation runtime.")
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--generated-layer", default="centerlines")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--ground-truth-layer", default="navstreet")
    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--bbox-file", type=Path, default=Path("data/Kosovo_bounding_box.txt"))
    parser.add_argument("--apply-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    gen = _load_line_geodata(args.generated, layer=args.generated_layer)
    gt = _load_line_geodata(args.ground_truth, layer=args.ground_truth_layer)

    context = build_evaluation_context(
        gt,
        bbox_file=args.bbox_file,
        apply_bbox=args.apply_bbox,
        sample_step_m=args.sample_step_m,
        topology_radii_m=(8.0, 15.0),
    )

    legacy_t, legacy_out = _timed(
        lambda: _legacy_eval(
            gen,
            gt,
            buffer_m=args.buffer_m,
            sample_step_m=args.sample_step_m,
            bbox_file=args.bbox_file,
            apply_bbox=args.apply_bbox,
        ),
        repeat=args.repeat,
    )
    new_no_context_t, new_no_context_out = _timed(
        lambda: evaluate_centerline_geodataframes(
            generated_gdf=gen,
            ground_truth_gdf=gt,
            buffer_m=args.buffer_m,
            sample_step_m=args.sample_step_m,
            bbox_file=args.bbox_file,
            apply_bbox=args.apply_bbox,
            compute_topology_metrics=True,
            compute_itopo=True,
            topology_radii_m=(8.0, 15.0),
            compute_hausdorff=False,
        ),
        repeat=args.repeat,
    )
    new_context_t, new_context_out = _timed(
        lambda: evaluate_centerline_geodataframes(
            generated_gdf=gen,
            ground_truth_gdf=gt,
            buffer_m=args.buffer_m,
            sample_step_m=args.sample_step_m,
            bbox_file=args.bbox_file,
            apply_bbox=args.apply_bbox,
            context=context,
            compute_topology_metrics=True,
            compute_itopo=True,
            topology_radii_m=(8.0, 15.0),
            compute_hausdorff=False,
        ),
        repeat=args.repeat,
    )

    report = {
        "repeat": int(args.repeat),
        "timing_sec": {
            "legacy_global_union_avg": legacy_t,
            "new_no_context_avg": new_no_context_t,
            "new_with_context_avg": new_context_t,
        },
        "speedup": {
            "new_vs_legacy": (legacy_t / new_no_context_t) if new_no_context_t > 0 else float("inf"),
            "new_context_vs_legacy": (legacy_t / new_context_t) if new_context_t > 0 else float("inf"),
            "new_context_vs_new_no_context": (new_no_context_t / new_context_t) if new_context_t > 0 else float("inf"),
        },
        "legacy_last": legacy_out[-1] if legacy_out else {},
        "new_no_context_last": new_no_context_out[-1] if new_no_context_out else {},
        "new_with_context_last": new_context_out[-1] if new_context_out else {},
    }

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
