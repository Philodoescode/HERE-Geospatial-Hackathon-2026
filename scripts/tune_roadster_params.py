from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.algorithms.roadster import RoadsterAlgorithm, RoadsterConfig
from centerline.evaluation import evaluate_centerline_geodataframes
from centerline.generation import generate_centerlines_with_algorithm, save_centerline_outputs
from centerline.io_utils import load_navstreet_csv


@dataclass
class ProgressEvent:
    pipeline: str
    stage: str
    step: int
    total_steps: int
    percent: float
    message: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], None]


def default_console_progress(event: ProgressEvent) -> None:
    pct = f"{event.percent:.1f}%"
    prefix = f"[{event.pipeline}] {event.stage} {event.step}/{event.total_steps} ({pct})"
    if event.metrics:
        parts = ", ".join(f"{k}={v}" for k, v in event.metrics.items())
        print(f"{prefix} {event.message} [{parts}]")
    else:
        print(f"{prefix} {event.message}")


def _parse_float_list(text: str) -> list[float]:
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty numeric list, got: {text}")
    return vals


def _build_trials(space: dict[str, list], search: str, n_trials: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    keys = list(space.keys())
    if search == "grid":
        return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    return [{k: rng.choice(space[k]) for k in keys} for _ in range(n_trials)]


def _emit_trial_progress(
    progress: ProgressCallback | None,
    *,
    trial_idx: int,
    total_trials: int,
    message: str,
    metrics: dict[str, float | int | str] | None = None,
) -> None:
    if progress is None:
        return
    percent = 100.0 * float(trial_idx) / max(1, total_trials)
    progress(
        ProgressEvent(
            pipeline="roadster_tune",
            stage="trial",
            step=trial_idx,
            total_steps=total_trials,
            percent=percent,
            message=message,
            metrics=metrics or {},
        )
    )


def tune_roadster_params(
    *,
    vpd_csv: str | Path = Path("data/Kosovo_VPD/Kosovo_VPD.csv"),
    hpd_csvs: list[str | Path] | None = None,
    ground_truth: str | Path = Path("outputs/ground_truth/kosovo_navstreet_ground_truth.gpkg"),
    ground_truth_layer: str = "navstreet",
    bbox_file: str | Path = Path("data/Kosovo_bounding_box.txt"),
    apply_bbox: bool = True,
    max_vpd_rows: int | None = 1500,
    max_hpd_rows_per_file: int | None = 50000,
    search: str = "random",
    trials: int = 24,
    seed: int = 42,
    cluster_center_radius_values: str = "25,35,45",
    cluster_frechet_eps_values: str = "12,16,20",
    cluster_heading_tolerance_values: str = "20,30,40",
    min_edge_support_values: str = "3,4.5,6,8",
    min_centerline_length_values: str = "12,20,30,40",
    buffer_m: float = 15.0,
    sample_step_m: float = 10.0,
    complexity_penalty: float = 0.000002,
    out_dir: str | Path = Path("outputs/tuning_roadster"),
    save_best_output: bool = True,
    progress: ProgressCallback | None = None,
) -> dict:
    hpd_csvs = hpd_csvs or [
        Path("data/Kosovo_HPD/XKO_HPD_week_1.csv"),
        Path("data/Kosovo_HPD/XKO_HPD_week_2.csv"),
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(ground_truth)
    if gt_path.suffix.lower() == ".csv":
        gt_df = load_navstreet_csv(gt_path)
        gt_gdf = gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")
    else:
        gt_gdf = gpd.read_file(gt_path, layer=ground_truth_layer)
        if gt_gdf.crs is None:
            gt_gdf = gt_gdf.set_crs("EPSG:4326")

    space = {
        "cluster_center_radius_m": _parse_float_list(cluster_center_radius_values),
        "cluster_frechet_eps_m": _parse_float_list(cluster_frechet_eps_values),
        "cluster_heading_tolerance_deg": _parse_float_list(cluster_heading_tolerance_values),
        "min_edge_support": _parse_float_list(min_edge_support_values),
        "min_centerline_length_m": _parse_float_list(min_centerline_length_values),
    }
    trial_params = _build_trials(space=space, search=search, n_trials=trials, seed=seed)

    rows = []
    best = None
    total_trials = len(trial_params)
    for idx, p in enumerate(trial_params, start=1):
        t0 = time.time()
        try:
            cfg = RoadsterConfig(
                cluster_center_radius_m=float(p["cluster_center_radius_m"]),
                cluster_frechet_eps_m=float(p["cluster_frechet_eps_m"]),
                cluster_heading_tolerance_deg=float(p["cluster_heading_tolerance_deg"]),
                min_edge_support=float(p["min_edge_support"]),
                min_centerline_length_m=float(p["min_centerline_length_m"]),
            )
            algo = RoadsterAlgorithm(config=cfg)
            result = generate_centerlines_with_algorithm(
                vpd_csv=vpd_csv,
                hpd_csvs=hpd_csvs,
                algorithm=algo,
                fused_only=True,
                max_vpd_rows=max_vpd_rows,
                max_hpd_rows_per_file=max_hpd_rows_per_file,
                bbox_file=bbox_file,
                apply_bbox=apply_bbox,
            )

            center = result.get("centerlines", pd.DataFrame())
            center_count = int(len(center))
            if center_count == 0:
                metrics = {"length_f1": 0.0, "length_precision": 0.0, "length_recall": 0.0}
            else:
                center_gdf = gpd.GeoDataFrame(center.copy(), geometry="geometry", crs="EPSG:4326")
                metrics = evaluate_centerline_geodataframes(
                    generated_gdf=center_gdf,
                    ground_truth_gdf=gt_gdf,
                    buffer_m=buffer_m,
                    sample_step_m=sample_step_m,
                    bbox_file=bbox_file,
                    apply_bbox=apply_bbox,
                )
            f1 = float(metrics.get("length_f1", 0.0) or 0.0)
            precision = float(metrics.get("length_precision", 0.0) or 0.0)
            recall = float(metrics.get("length_recall", 0.0) or 0.0)
            score = f1 - complexity_penalty * center_count
            elapsed = time.time() - t0

            row = {
                "trial": idx,
                **p,
                "centerline_count": center_count,
                "trace_count": int(result.get("trace_count", 0)),
                "sample_point_count": int(result.get("sample_point_count", 0)),
                "length_precision": precision,
                "length_recall": recall,
                "length_f1": f1,
                "objective_score": score,
                "elapsed_sec": elapsed,
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            if best is None or row["objective_score"] > best["objective_score"]:
                best = row
            _emit_trial_progress(
                progress,
                trial_idx=idx,
                total_trials=total_trials,
                message="Trial complete.",
                metrics={"score": round(score, 6), "f1": round(f1, 6), "centerlines": center_count},
            )
        except Exception as ex:
            elapsed = time.time() - t0
            row = {
                "trial": idx,
                **p,
                "centerline_count": -1,
                "trace_count": 0,
                "sample_point_count": 0,
                "length_precision": 0.0,
                "length_recall": 0.0,
                "length_f1": 0.0,
                "objective_score": -1e9,
                "elapsed_sec": elapsed,
                "status": "failed",
                "error": str(ex),
            }
            rows.append(row)
            _emit_trial_progress(
                progress,
                trial_idx=idx,
                total_trials=total_trials,
                message="Trial failed.",
                metrics={"error": str(ex)},
            )

    df = pd.DataFrame(rows).sort_values(["objective_score", "length_f1"], ascending=False)
    csv_path = out_dir / "tuning_results.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "search": search,
        "trials_requested": int(trials),
        "trials_executed": int(total_trials),
        "best": best,
    }
    summary_path = out_dir / "tuning_best.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    files = None
    if save_best_output and best is not None:
        cfg = RoadsterConfig(
            cluster_center_radius_m=float(best["cluster_center_radius_m"]),
            cluster_frechet_eps_m=float(best["cluster_frechet_eps_m"]),
            cluster_heading_tolerance_deg=float(best["cluster_heading_tolerance_deg"]),
            min_edge_support=float(best["min_edge_support"]),
            min_centerline_length_m=float(best["min_centerline_length_m"]),
        )
        algo = RoadsterAlgorithm(config=cfg)
        best_result = generate_centerlines_with_algorithm(
            vpd_csv=vpd_csv,
            hpd_csvs=hpd_csvs,
            algorithm=algo,
            fused_only=True,
            max_vpd_rows=max_vpd_rows,
            max_hpd_rows_per_file=max_hpd_rows_per_file,
            bbox_file=bbox_file,
            apply_bbox=apply_bbox,
        )
        files = save_centerline_outputs(
            result=best_result,
            output_dir=out_dir,
            stem="best_tuned_roadster_centerlines",
        )

    return {
        "tuning_results_csv": str(csv_path),
        "tuning_best_json": str(summary_path),
        "best": best,
        "saved_best_output_files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Roadster centerline parameters on smoke subset.")
    parser.add_argument("--vpd-csv", type=Path, default=Path("data/Kosovo_VPD/Kosovo_VPD.csv"))
    parser.add_argument(
        "--hpd-csvs",
        type=Path,
        nargs="+",
        default=[Path("data/Kosovo_HPD/XKO_HPD_week_1.csv"), Path("data/Kosovo_HPD/XKO_HPD_week_2.csv")],
    )
    parser.add_argument("--ground-truth", type=Path, default=Path("outputs/ground_truth/kosovo_navstreet_ground_truth.gpkg"))
    parser.add_argument("--ground-truth-layer", default="navstreet")
    parser.add_argument("--bbox-file", type=Path, default=Path("data/Kosovo_bounding_box.txt"))
    parser.add_argument("--apply-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-vpd-rows", type=int, default=1500)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=50000)
    parser.add_argument("--search", choices=["random", "grid"], default="random")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-center-radius-values", default="25,35,45")
    parser.add_argument("--cluster-frechet-eps-values", default="12,16,20")
    parser.add_argument("--cluster-heading-tolerance-values", default="20,30,40")
    parser.add_argument("--min-edge-support-values", default="3,4.5,6,8")
    parser.add_argument("--min-centerline-length-values", default="12,20,30,40")
    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--complexity-penalty", type=float, default=0.000002)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tuning_roadster"))
    parser.add_argument("--save-best-output", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    result = tune_roadster_params(
        vpd_csv=args.vpd_csv,
        hpd_csvs=args.hpd_csvs,
        ground_truth=args.ground_truth,
        ground_truth_layer=args.ground_truth_layer,
        bbox_file=args.bbox_file,
        apply_bbox=args.apply_bbox,
        max_vpd_rows=args.max_vpd_rows,
        max_hpd_rows_per_file=args.max_hpd_rows_per_file,
        search=args.search,
        trials=args.trials,
        seed=args.seed,
        cluster_center_radius_values=args.cluster_center_radius_values,
        cluster_frechet_eps_values=args.cluster_frechet_eps_values,
        cluster_heading_tolerance_values=args.cluster_heading_tolerance_values,
        min_edge_support_values=args.min_edge_support_values,
        min_centerline_length_values=args.min_centerline_length_values,
        buffer_m=args.buffer_m,
        sample_step_m=args.sample_step_m,
        complexity_penalty=args.complexity_penalty,
        out_dir=args.out_dir,
        save_best_output=args.save_best_output,
        progress=default_console_progress,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
