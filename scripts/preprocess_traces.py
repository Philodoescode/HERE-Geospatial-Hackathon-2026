from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.io_utils import (
    clip_line_geometries_to_bbox,
    infer_local_projected_crs,
    load_bbox_from_txt,
    load_hpd_traces,
    load_vpd_traces,
)
from centerline.preprocessing import (
    SourcePreprocessConfig,
    default_source_preprocess_config,
    preprocess_traces_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess VPD + Probe(HPD) traces for Roadster-style centerline generation."
    )
    parser.add_argument("--vpd-csv", type=Path, default=Path("data/Kosovo_VPD/Kosovo_VPD.csv"))
    parser.add_argument(
        "--hpd-csvs",
        type=Path,
        nargs="+",
        default=[
            Path("data/Kosovo_HPD/XKO_HPD_week_1.csv"),
            Path("data/Kosovo_HPD/XKO_HPD_week_2.csv"),
        ],
    )
    parser.add_argument("--fused-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-vpd-rows", type=int, default=None)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=None)
    parser.add_argument("--bbox-file", type=Path, default=Path("data/Kosovo_bounding_box.txt"))
    parser.add_argument("--apply-bbox", action=argparse.BooleanOptionalAction, default=True)

    # VPD preprocessing knobs
    parser.add_argument("--vpd-spacing-m", type=float, default=4.0)
    parser.add_argument("--vpd-simplify-m", type=float, default=1.5)
    parser.add_argument("--vpd-gap-m", type=float, default=60.0)

    # HPD preprocessing knobs
    parser.add_argument("--hpd-spacing-m", type=float, default=8.0)
    parser.add_argument("--hpd-simplify-m", type=float, default=3.0)
    parser.add_argument("--hpd-gap-m", type=float, default=90.0)
    parser.add_argument("--hpd-time-gap-s", type=float, default=300.0)

    parser.add_argument("--out-dir", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument("--stem", default="kosovo_preprocessed_traces")
    args = parser.parse_args()

    vpd = load_vpd_traces(args.vpd_csv, fused_only=args.fused_only, max_rows=args.max_vpd_rows)
    hpd = load_hpd_traces(args.hpd_csvs, max_rows_per_file=args.max_hpd_rows_per_file)
    traces = pd.concat([vpd, hpd], ignore_index=True)
    traces = traces[traces["geometry"].notnull()].reset_index(drop=True)

    if args.apply_bbox:
        bbox = load_bbox_from_txt(args.bbox_file)
        traces = clip_line_geometries_to_bbox(traces, bbox_wgs84=bbox, geometry_col="geometry")

    if traces.empty:
        raise RuntimeError("No valid traces found after loading/filtering.")

    proj = infer_local_projected_crs(list(traces["geometry"]))
    to_proj = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    to_wgs = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    vpd_cfg = default_source_preprocess_config("VPD")
    hpd_cfg = default_source_preprocess_config("HPD")
    vpd_cfg = SourcePreprocessConfig(
        resample_spacing_m=args.vpd_spacing_m,
        simplify_tolerance_m=args.vpd_simplify_m,
        max_gap_m=args.vpd_gap_m,
        max_time_gap_s=vpd_cfg.max_time_gap_s,
        min_segment_length_m=vpd_cfg.min_segment_length_m,
        spike_deflection_deg=vpd_cfg.spike_deflection_deg,
        spike_edge_max_m=vpd_cfg.spike_edge_max_m,
        split_turn_deflection_deg=vpd_cfg.split_turn_deflection_deg,
        split_turn_min_progress_m=vpd_cfg.split_turn_min_progress_m,
    )
    hpd_cfg = SourcePreprocessConfig(
        resample_spacing_m=args.hpd_spacing_m,
        simplify_tolerance_m=args.hpd_simplify_m,
        max_gap_m=args.hpd_gap_m,
        max_time_gap_s=args.hpd_time_gap_s,
        min_segment_length_m=hpd_cfg.min_segment_length_m,
        spike_deflection_deg=hpd_cfg.spike_deflection_deg,
        spike_edge_max_m=hpd_cfg.spike_edge_max_m,
        split_turn_deflection_deg=hpd_cfg.split_turn_deflection_deg,
        split_turn_min_progress_m=hpd_cfg.split_turn_min_progress_m,
    )

    pre = preprocess_traces_dataframe(
        traces=traces,
        to_proj=to_proj,
        to_wgs=to_wgs,
        source_cfgs={"VPD": vpd_cfg, "HPD": hpd_cfg},
    )
    if pre.empty:
        raise RuntimeError("Preprocessing produced no segments.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save geospatial output.
    gpkg_path = args.out_dir / f"{args.stem}.gpkg"
    write_cols = [c for c in pre.columns if c not in {"geometry_xy", "coords_xy", "altitudes", "point_times", "headings"}]
    gdf = gpd.GeoDataFrame(pre[write_cols].copy(), geometry="geometry", crs="EPSG:4326")
    gdf.to_file(gpkg_path, layer="preprocessed_traces", driver="GPKG")

    # Save CSV for debugging.
    csv_path = args.out_dir / f"{args.stem}.csv"
    out_csv = pre.copy()
    out_csv["geometry_wkt"] = out_csv["geometry"].astype(str)
    out_csv["geometry_xy_wkt"] = out_csv["geometry_xy"].astype(str)
    out_csv["coords_xy"] = out_csv["coords_xy"].map(lambda a: json.dumps(np.asarray(a).tolist()))
    out_csv["altitudes"] = out_csv["altitudes"].map(lambda a: json.dumps(np.asarray(a).tolist()))
    out_csv["point_times"] = out_csv["point_times"].map(lambda a: json.dumps(np.asarray(a).tolist()))
    out_csv["headings"] = out_csv["headings"].map(lambda a: json.dumps(np.asarray(a).tolist()))
    out_csv = out_csv.drop(columns=["geometry", "geometry_xy"])
    out_csv.to_csv(csv_path, index=False)

    summary = {
        "projected_crs": str(proj),
        "input_trace_count": int(len(traces)),
        "preprocessed_segment_count": int(len(pre)),
        "source_counts": {k: int(v) for k, v in pre["source"].value_counts().to_dict().items()},
        "settings": {
            "vpd": vars(vpd_cfg),
            "hpd": vars(hpd_cfg),
        },
    }
    summary_path = args.out_dir / f"{args.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved preprocessed traces: {gpkg_path}")
    print(f"Saved preprocessed CSV   : {csv_path}")
    print(f"Saved summary            : {summary_path}")
    print(f"Segments                 : {len(pre)}")


if __name__ == "__main__":
    main()
