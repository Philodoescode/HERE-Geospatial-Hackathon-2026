from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.algorithms.roadster import RoadsterAlgorithm  # noqa: E402
from centerline.evaluation import evaluate_centerlines, save_metrics  # noqa: E402
from centerline.generation import generate_centerlines_with_algorithm  # noqa: E402


def _serialize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "geometry":
            continue
        if out[col].dtype == "object":
            out[col] = out[col].map(
                lambda v: json.dumps(sorted(v) if isinstance(v, set) else v)
                if isinstance(v, (list, dict, set))
                else v
            )
    return out


def _write_csv_from_parquet(parquet_path: str, csv_path: str) -> None:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df.to_csv(csv_path, index=False)


def _attempt_csv_write_from_parquet(
    parquet_path: Path,
    csv_path: Path,
    timeout_sec: float,
) -> tuple[bool, bool]:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_write_csv_from_parquet,
        args=(str(parquet_path), str(csv_path)),
    )
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        if csv_path.exists():
            try:
                csv_path.unlink()
            except Exception:
                pass
        return False, True
    if proc.exitcode != 0:
        if csv_path.exists():
            try:
                csv_path.unlink()
            except Exception:
                pass
        return False, False
    return csv_path.exists(), False


def _save_outputs(
    result: dict,
    *,
    out_dir: Path,
    stem: str,
    write_csv: bool,
    csv_timeout_sec: float,
) -> dict[str, str | bool | float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str | bool | float] = {}

    nodes = result["nodes"].copy()
    edges = result["edges"].copy()
    centerlines = result["centerlines"].copy()

    if not nodes.empty:
        nodes["geometry"] = gpd.points_from_xy(nodes["lon"], nodes["lat"], crs="EPSG:4326")
        gnodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
        nodes_gpkg = out_dir / f"{stem}_nodes.gpkg"
        gnodes.to_file(nodes_gpkg, layer="nodes", driver="GPKG")
        files["nodes_gpkg"] = str(nodes_gpkg)

        nodes_parquet = out_dir / f"{stem}_nodes.parquet"
        nodes_write = nodes.copy()
        nodes_write["geometry_wkt"] = nodes_write["geometry"].astype(str)
        nodes_write = nodes_write.drop(columns=["geometry"])
        nodes_write.to_parquet(nodes_parquet, index=False, engine="pyarrow")
        files["nodes_parquet"] = str(nodes_parquet)

    if not edges.empty:
        edges_write = _serialize_object_columns(edges)
        gedges = gpd.GeoDataFrame(edges_write, geometry="geometry", crs="EPSG:4326")
        edges_gpkg = out_dir / f"{stem}_edges.gpkg"
        gedges.to_file(edges_gpkg, layer="edges", driver="GPKG")
        files["edges_gpkg"] = str(edges_gpkg)

        edges_parquet = out_dir / f"{stem}_edges.parquet"
        edges_wkt = edges_write.copy()
        edges_wkt["geometry_wkt"] = edges_wkt["geometry"].astype(str)
        edges_wkt = edges_wkt.drop(columns=["geometry"])
        edges_wkt.to_parquet(edges_parquet, index=False, engine="pyarrow")
        files["edges_parquet"] = str(edges_parquet)

    if not centerlines.empty:
        center_write = _serialize_object_columns(centerlines)
        gcenter = gpd.GeoDataFrame(center_write, geometry="geometry", crs="EPSG:4326")
        center_gpkg = out_dir / f"{stem}.gpkg"
        gcenter.to_file(center_gpkg, layer="centerlines", driver="GPKG")
        files["centerlines_gpkg"] = str(center_gpkg)

        center_parquet = out_dir / f"{stem}.parquet"
        center_wkt = center_write.copy()
        center_wkt["geometry_wkt"] = center_wkt["geometry"].astype(str)
        center_wkt = center_wkt.drop(columns=["geometry"])
        center_wkt.to_parquet(center_parquet, index=False, engine="pyarrow")
        files["centerlines_parquet"] = str(center_parquet)

        if write_csv:
            center_csv = out_dir / f"{stem}.csv"
            csv_written, csv_timed_out = _attempt_csv_write_from_parquet(
                center_parquet, center_csv, timeout_sec=csv_timeout_sec
            )
            files["centerlines_csv"] = str(center_csv) if csv_written else ""
            files["centerlines_csv_written"] = bool(csv_written)
            files["centerlines_csv_timeout"] = bool(csv_timed_out)
            files["centerlines_csv_timeout_sec"] = float(csv_timeout_sec)

    summary = {
        "trace_count": int(result.get("trace_count", 0)),
        "sample_point_count": int(result.get("sample_point_count", 0)),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "centerline_count": int(len(centerlines)),
        "projected_crs": str(result.get("projected_crs")),
    }
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files["summary_json"] = str(summary_path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Roadster on full parquet dataset with start/end timestamps, elapsed time, "
            "and centerline evaluation metrics."
        )
    )
    parser.add_argument("--vpd-parquet", type=Path, default=Path("../data/Kosovo_VPD/Kosovo_VPD.parquet"))
    parser.add_argument(
        "--hpd-parquets",
        type=Path,
        nargs="+",
        default=[
            Path("../data/Kosovo_HPD/XKO_HPD_week_1.parquet"),
            Path("../data/Kosovo_HPD/XKO_HPD_week_2.parquet"),
        ],
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("../data/Kosovo's nav streets/nav_kosovo.parquet"),
    )
    parser.add_argument("--bbox-file", type=Path, default=Path("../data/Kosovo_bounding_box.txt"))
    parser.add_argument("--apply-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-vpd-rows", type=int, default=None)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=None)
    parser.add_argument("--out-root", type=Path, default=Path("../outputs/roadster_full_parquet_timed"))
    parser.add_argument("--stem", default="roadster_full_parquet")
    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--compute-topology-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-hausdorff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv-timeout-sec", type=float, default=8.0)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.smoke:
        if args.max_vpd_rows is None:
            args.max_vpd_rows = 1200
        if args.max_hpd_rows_per_file is None:
            args.max_hpd_rows_per_file = 18000

    out_root = args.out_root
    centerline_out = out_root / "centerlines"
    eval_out = out_root / "evaluation"
    centerline_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.now(timezone.utc).astimezone()
    start_epoch = time.time()

    print(f"[roadster-run] pid={os.getpid()}")
    print(f"[roadster-run] start_local={start_dt.isoformat()}")
    print(f"[roadster-run] smoke={args.smoke}")
    print(f"[roadster-run] vpd_parquet={args.vpd_parquet}")
    print(f"[roadster-run] hpd_parquets={','.join(str(p) for p in args.hpd_parquets)}")
    print(f"[roadster-run] max_vpd_rows={args.max_vpd_rows}")
    print(f"[roadster-run] max_hpd_rows_per_file={args.max_hpd_rows_per_file}")

    algo = RoadsterAlgorithm()
    result = generate_centerlines_with_algorithm(
        vpd_csv=args.vpd_parquet,
        hpd_csvs=args.hpd_parquets,
        algorithm=algo,
        fused_only=args.fused_only,
        max_vpd_rows=args.max_vpd_rows,
        max_hpd_rows_per_file=args.max_hpd_rows_per_file,
        bbox_file=args.bbox_file,
        apply_bbox=args.apply_bbox,
    )
    files = _save_outputs(
        result,
        out_dir=centerline_out,
        stem=args.stem,
        write_csv=args.write_csv,
        csv_timeout_sec=float(args.csv_timeout_sec),
    )

    centerline_gpkg = Path(str(files["centerlines_gpkg"]))
    metrics_json = eval_out / f"{args.stem}_metrics.json"
    gt_layer = "navstreet" if args.ground_truth.suffix.lower() == ".gpkg" else None
    metrics = evaluate_centerlines(
        generated_centerlines=centerline_gpkg,
        generated_layer="centerlines",
        ground_truth_navstreet=args.ground_truth,
        ground_truth_layer=gt_layer,
        buffer_m=float(args.buffer_m),
        sample_step_m=float(args.sample_step_m),
        bbox_file=args.bbox_file,
        apply_bbox=bool(args.apply_bbox),
        clip_generated_to_ground_truth=False,
        clip_buffer_m=0.0,
        compute_topology_metrics=bool(args.compute_topology_metrics),
        compute_hausdorff=bool(args.compute_hausdorff),
    )
    save_metrics(metrics, metrics_json)

    end_epoch = time.time()
    end_dt = datetime.now(timezone.utc).astimezone()
    elapsed = end_epoch - start_epoch

    run_summary = {
        "start_local": start_dt.isoformat(),
        "end_local": end_dt.isoformat(),
        "elapsed_seconds": float(elapsed),
        "inputs": {
            "vpd_parquet": str(args.vpd_parquet),
            "hpd_parquets": [str(p) for p in args.hpd_parquets],
            "ground_truth": str(args.ground_truth),
            "bbox_file": str(args.bbox_file),
        },
        "outputs": {
            **{k: v for k, v in files.items()},
            "metrics_json": str(metrics_json),
            "metrics_parquet": str(metrics_json.with_suffix(".parquet")),
        },
        "evaluation": metrics,
    }
    run_summary_path = out_root / f"{args.stem}_run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"[roadster-run] end_local={end_dt.isoformat()}")
    print(f"[roadster-run] elapsed_seconds={elapsed:.2f}")
    print(f"[roadster-run] centerlines_gpkg={files.get('centerlines_gpkg', '')}")
    print(f"[roadster-run] centerlines_parquet={files.get('centerlines_parquet', '')}")
    print(f"[roadster-run] metrics_json={metrics_json}")
    print(f"[roadster-run] run_summary_json={run_summary_path}")
    for key in (
        "generated_count",
        "ground_truth_count",
        "length_precision",
        "length_recall",
        "length_f1",
    ):
        if key in metrics:
            print(f"[roadster-run] {key}={metrics[key]}")

    csv_written = files.get("centerlines_csv_written")
    csv_timeout = files.get("centerlines_csv_timeout")
    if csv_written is not None:
        print(f"[roadster-run] centerlines_csv_written={csv_written}")
        print(f"[roadster-run] centerlines_csv_timeout={csv_timeout}")


if __name__ == "__main__":
    main()
