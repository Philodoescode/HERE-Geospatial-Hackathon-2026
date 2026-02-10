from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.algorithms.kharita import KharitaAlgorithm, KharitaConfig
from centerline.evaluation import evaluate_centerline_geodataframes
from centerline.generation import generate_centerlines_with_algorithm, save_centerline_outputs
from centerline.io_utils import load_navstreet_csv


def _parse_float_list(text: str) -> list[float]:
    vals = []
    for t in str(text).split(","):
        tt = t.strip()
        if not tt:
            continue
        vals.append(float(tt))
    if not vals:
        raise ValueError(f"Expected non-empty numeric list, got: {text}")
    return vals


def _parse_int_list(text: str) -> list[int]:
    vals = []
    for t in str(text).split(","):
        tt = t.strip()
        if not tt:
            continue
        vals.append(int(float(tt)))
    if not vals:
        raise ValueError(f"Expected non-empty integer list, got: {text}")
    return vals


def _build_trials(space: dict[str, list], search: str, n_trials: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    keys = list(space.keys())

    if search == "grid":
        prod = itertools.product(*(space[k] for k in keys))
        trials = [dict(zip(keys, vals)) for vals in prod]
        return trials

    trials = []
    for _ in range(n_trials):
        row = {k: rng.choice(space[k]) for k in keys}
        trials.append(row)
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune Kharita centerline parameters on smoke subset and rank by objective score."
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
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("outputs/ground_truth/kosovo_navstreet_ground_truth.gpkg"),
    )
    parser.add_argument("--ground-truth-layer", default="navstreet")

    parser.add_argument("--bbox-file", type=Path, default=Path("data/Kosovo_bounding_box.txt"))
    parser.add_argument("--apply-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clip-generated-to-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--clip-buffer-m", type=float, default=200.0)

    parser.add_argument("--max-vpd-rows", type=int, default=1500)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=50000)

    parser.add_argument("--search", choices=["random", "grid"], default="random")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)

    # Search space
    parser.add_argument("--cluster-radius-values", default="6,8,10,12")
    parser.add_argument("--heading-tolerance-values", default="25,35,45")
    parser.add_argument("--min-edge-support-values", default="2,4,6,8")
    parser.add_argument("--sample-spacing-values", default="8,10,12,15")
    parser.add_argument("--max-transition-values", default="25,35,50")
    parser.add_argument("--min-centerline-length-values", default="12,20,30,40")
    parser.add_argument("--smooth-iterations-values", default="1,2,3")

    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)

    parser.add_argument(
        "--complexity-penalty",
        type=float,
        default=0.000002,
        help="Objective = length_f1 - complexity_penalty * centerline_count",
    )

    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tuning"))
    parser.add_argument("--save-best-output", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = args.ground_truth
    if gt_path.suffix.lower() == ".csv":
        gt_df = load_navstreet_csv(gt_path)
        gt_gdf = gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")
    else:
        gt_gdf = gpd.read_file(gt_path, layer=args.ground_truth_layer)
        if gt_gdf.crs is None:
            gt_gdf = gt_gdf.set_crs("EPSG:4326")

    space = {
        "cluster_radius_m": _parse_float_list(args.cluster_radius_values),
        "heading_tolerance_deg": _parse_float_list(args.heading_tolerance_values),
        "min_edge_support": _parse_float_list(args.min_edge_support_values),
        "sample_spacing_m": _parse_float_list(args.sample_spacing_values),
        "max_transition_distance_m": _parse_float_list(args.max_transition_values),
        "min_centerline_length_m": _parse_float_list(args.min_centerline_length_values),
        "smooth_iterations": _parse_int_list(args.smooth_iterations_values),
    }

    trials = _build_trials(space=space, search=args.search, n_trials=args.trials, seed=args.seed)

    rows = []
    best = None

    print(f"Running {len(trials)} tuning trials ({args.search}).")
    for idx, p in enumerate(trials, start=1):
        t0 = time.time()
        try:
            cfg = KharitaConfig(
                cluster_radius_m=float(p["cluster_radius_m"]),
                heading_tolerance_deg=float(p["heading_tolerance_deg"]),
                min_edge_support=float(p["min_edge_support"]),
                sample_spacing_m=float(p["sample_spacing_m"]),
                max_transition_distance_m=float(p["max_transition_distance_m"]),
                min_centerline_length_m=float(p["min_centerline_length_m"]),
                smooth_iterations=int(p["smooth_iterations"]),
            )
            algo = KharitaAlgorithm(config=cfg)

            result = generate_centerlines_with_algorithm(
                vpd_csv=args.vpd_csv,
                hpd_csvs=args.hpd_csvs,
                algorithm=algo,
                fused_only=True,
                max_vpd_rows=args.max_vpd_rows,
                max_hpd_rows_per_file=args.max_hpd_rows_per_file,
                bbox_file=args.bbox_file,
                apply_bbox=args.apply_bbox,
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
                    buffer_m=args.buffer_m,
                    sample_step_m=args.sample_step_m,
                    bbox_file=args.bbox_file,
                    apply_bbox=args.apply_bbox,
                    clip_generated_to_ground_truth=args.clip_generated_to_ground_truth,
                    clip_buffer_m=args.clip_buffer_m,
                )

            f1 = float(metrics.get("length_f1", 0.0) or 0.0)
            precision = float(metrics.get("length_precision", 0.0) or 0.0)
            recall = float(metrics.get("length_recall", 0.0) or 0.0)
            score = f1 - args.complexity_penalty * center_count
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

            print(
                f"[{idx}/{len(trials)}] score={score:.5f} f1={f1:.4f} "
                f"n={center_count} params={p}"
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
            print(f"[{idx}/{len(trials)}] failed: {ex}")

    df = pd.DataFrame(rows).sort_values(["objective_score", "length_f1"], ascending=False)
    csv_path = args.out_dir / "tuning_results.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "search": args.search,
        "trials_requested": int(args.trials),
        "trials_executed": int(len(trials)),
        "best": best,
    }
    summary_path = args.out_dir / "tuning_best.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved trial table: {csv_path}")
    print(f"Saved best summary: {summary_path}")

    if best is None:
        print("No successful trial found.")
        return

    print("\\nBest trial:")
    print(json.dumps(best, indent=2))

    cmd = (
        "scripts/generate_centerlines.py "
        f"--cluster-radius-m {best['cluster_radius_m']} "
        f"--heading-tolerance-deg {best['heading_tolerance_deg']} "
        f"--min-edge-support {best['min_edge_support']} "
        f"--sample-spacing-m {best['sample_spacing_m']} "
        f"--max-transition-distance-m {best['max_transition_distance_m']} "
        f"--min-centerline-length-m {best['min_centerline_length_m']} "
        f"--smooth-iterations {int(best['smooth_iterations'])}"
    )
    print("\\nBest full-data command:")
    print(cmd)

    if args.save_best_output:
        cfg = KharitaConfig(
            cluster_radius_m=float(best["cluster_radius_m"]),
            heading_tolerance_deg=float(best["heading_tolerance_deg"]),
            min_edge_support=float(best["min_edge_support"]),
            sample_spacing_m=float(best["sample_spacing_m"]),
            max_transition_distance_m=float(best["max_transition_distance_m"]),
            min_centerline_length_m=float(best["min_centerline_length_m"]),
            smooth_iterations=int(best["smooth_iterations"]),
        )
        algo = KharitaAlgorithm(config=cfg)
        best_result = generate_centerlines_with_algorithm(
            vpd_csv=args.vpd_csv,
            hpd_csvs=args.hpd_csvs,
            algorithm=algo,
            fused_only=True,
            max_vpd_rows=args.max_vpd_rows,
            max_hpd_rows_per_file=args.max_hpd_rows_per_file,
            bbox_file=args.bbox_file,
            apply_bbox=args.apply_bbox,
        )
        files = save_centerline_outputs(
            result=best_result,
            output_dir=args.out_dir,
            stem="best_tuned_smoke_centerlines",
        )
        print("\\nSaved best smoke output:")
        for k, v in files.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
