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
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty numeric list, got: {text}")
    return vals


def _parse_int_list(text: str) -> list[int]:
    vals = [int(float(t.strip())) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty integer list, got: {text}")
    return vals


def _parse_bool_list(text: str) -> list[bool]:
    true_vals = {"1", "true", "yes", "y", "on"}
    false_vals = {"0", "false", "no", "n", "off"}
    vals: list[bool] = []
    for token in str(text).split(","):
        tt = token.strip().lower()
        if not tt:
            continue
        if tt in true_vals:
            vals.append(True)
        elif tt in false_vals:
            vals.append(False)
        else:
            raise ValueError(f"Invalid boolean token '{token}' in list: {text}")
    if not vals:
        raise ValueError(f"Expected non-empty boolean list, got: {text}")
    return vals


def _build_trials(space: dict[str, list], search: str, n_trials: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    keys = list(space.keys())
    if search == "grid":
        return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    return [{k: rng.choice(space[k]) for k in keys} for _ in range(n_trials)]


def _score_trial(
    metrics: dict,
    *,
    precision_floor: float,
    max_length_ratio: float,
    precision_penalty: float,
    length_penalty: float,
) -> tuple[float, float, float, float]:
    precision = float(metrics.get("length_precision", 0.0) or 0.0)
    recall = float(metrics.get("length_recall", 0.0) or 0.0)
    generated_len = float(metrics.get("generated_total_length_m", 0.0) or 0.0)
    gt_len = float(metrics.get("ground_truth_total_length_m", 0.0) or 0.0)
    length_ratio = (generated_len / gt_len) if gt_len > 0 else float("inf")

    precision_gap = max(0.0, precision_floor - precision)
    ratio_gap = max(0.0, length_ratio - max_length_ratio)
    score = recall - precision_penalty * (precision_gap ** 2) - length_penalty * (ratio_gap ** 2)
    return score, precision, recall, length_ratio


def _load_ground_truth(path: Path, layer: str) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".csv":
        gt_df = load_navstreet_csv(path)
        return gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")
    gt = gpd.read_file(path, layer=layer)
    if gt.crs is None:
        gt = gt.set_crs("EPSG:4326")
    return gt


def _evaluate_result(
    result: dict,
    gt_gdf: gpd.GeoDataFrame,
    *,
    bbox_file: Path,
    apply_bbox: bool,
    clip_generated_to_ground_truth: bool,
    clip_buffer_m: float,
    buffer_m: float,
    sample_step_m: float,
) -> dict:
    center = result.get("centerlines", pd.DataFrame())
    if len(center) == 0:
        return {
            "generated_count": 0,
            "ground_truth_count": int(len(gt_gdf)),
            "generated_total_length_m": 0.0,
            "ground_truth_total_length_m": float(gt_gdf.to_crs(result.get("projected_crs")).length.sum())
            if result.get("projected_crs") is not None and not gt_gdf.empty
            else 0.0,
            "length_precision": 0.0,
            "length_recall": 0.0,
            "length_f1": 0.0,
            "clip_generated_to_ground_truth": clip_generated_to_ground_truth,
        }

    center_gdf = gpd.GeoDataFrame(center.copy(), geometry="geometry", crs="EPSG:4326")
    return evaluate_centerline_geodataframes(
        generated_gdf=center_gdf,
        ground_truth_gdf=gt_gdf,
        buffer_m=buffer_m,
        sample_step_m=sample_step_m,
        bbox_file=bbox_file,
        apply_bbox=apply_bbox,
        clip_generated_to_ground_truth=clip_generated_to_ground_truth,
        clip_buffer_m=clip_buffer_m,
        compute_topology_metrics=False,
        compute_hausdorff=False,
    )


def _build_config(p: dict) -> KharitaConfig:
    return KharitaConfig(
        cluster_radius_m=float(p["cluster_radius_m"]),
        heading_tolerance_deg=float(p["heading_tolerance_deg"]),
        heading_distance_weight_m=float(p["heading_distance_weight_m"]),
        min_cluster_points=int(p["min_cluster_points"]),
        sample_spacing_m=float(p["sample_spacing_m"]),
        max_points_per_trace=int(p["max_points_per_trace"]),
        vpd_base_weight=float(p["vpd_base_weight"]),
        hpd_base_weight=float(p["hpd_base_weight"]),
        max_transition_distance_m=float(p["max_transition_distance_m"]),
        min_edge_support=float(p["min_edge_support"]),
        reverse_edge_ratio=float(p["reverse_edge_ratio"]),
        enable_transitive_pruning=bool(p["enable_transitive_pruning"]),
        transitive_max_hops=int(p["transitive_max_hops"]),
        transitive_ratio=float(p["transitive_ratio"]),
        transitive_max_checks=int(p["transitive_max_checks"]),
        use_turn_preserving_smoothing=bool(p["use_turn_preserving_smoothing"]),
        turn_smoothing_deg=float(p["turn_smoothing_deg"]),
        turn_smoothing_neighbor_weight=float(p["turn_smoothing_neighbor_weight"]),
        apply_candidate_selection=bool(p["apply_candidate_selection"]),
        candidate_selection_threshold=float(p["candidate_selection_threshold"]),
        candidate_length_scale_m=float(p["candidate_length_scale_m"]),
        candidate_density_scale=float(p["candidate_density_scale"]),
        candidate_force_keep_weighted_support=float(p["candidate_force_keep_weighted_support"]),
        candidate_short_length_m=float(p["candidate_short_length_m"]),
        candidate_low_weighted_support=float(p["candidate_low_weighted_support"]),
        candidate_dangling_max_length_m=float(p["candidate_dangling_max_length_m"]),
        candidate_dangling_min_weighted_support=float(p["candidate_dangling_min_weighted_support"]),
        min_centerline_length_m=float(p["min_centerline_length_m"]),
        smooth_iterations=int(p["smooth_iterations"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recall-first tuning for Kharita in full-bbox/no-clip mode with "
            "precision and generated-length guardrails."
        )
    )
    parser.add_argument("--vpd-csv", type=Path, default=Path("data/Kosovo_VPD/Kosvo_VPD.parquet"))
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
    parser.add_argument("--clip-generated-to-ground-truth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-buffer-m", type=float, default=0.0)
    parser.add_argument("--max-vpd-rows", type=int, default=1500)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=50000)
    parser.add_argument("--search", choices=["random", "grid"], default="random")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cluster-radius-values", default="10,14,18")
    parser.add_argument("--heading-tolerance-values", default="30,40,50")
    parser.add_argument("--heading-distance-weight-values", default="0.10,0.22,0.35")
    parser.add_argument("--min-cluster-points-values", default="1,2")
    parser.add_argument("--sample-spacing-values", default="5,8,10")
    parser.add_argument("--max-points-per-trace-values", default="120,240,360")
    parser.add_argument("--vpd-base-weight-values", default="1.1,1.2,1.4")
    parser.add_argument("--hpd-base-weight-values", default="0.8,1.0,1.1")
    parser.add_argument("--max-transition-values", default="35,50,65")
    parser.add_argument("--min-edge-support-values", default="1,2,3")
    parser.add_argument("--reverse-edge-ratio-values", default="0.05,0.10,0.20")
    parser.add_argument("--enable-transitive-pruning-values", default="true,false")
    parser.add_argument("--transitive-max-hops-values", default="3,4,5")
    parser.add_argument("--transitive-ratio-values", default="1.02,1.03,1.05")
    parser.add_argument("--transitive-max-checks-values", default="0,10000,25000")
    parser.add_argument("--use-turn-smoothing-values", default="true,false")
    parser.add_argument("--turn-smoothing-deg-values", default="24,30,36")
    parser.add_argument("--turn-smoothing-neighbor-weight-values", default="0.20,0.25,0.30")
    parser.add_argument("--apply-candidate-selection-values", default="true,false")
    parser.add_argument("--candidate-selection-threshold-values", default="0.45,0.52,0.60")
    parser.add_argument("--candidate-length-scale-values", default="50,70,90")
    parser.add_argument("--candidate-density-scale-values", default="0.20,0.25,0.35")
    parser.add_argument("--candidate-force-keep-support-values", default="12,18,24")
    parser.add_argument("--candidate-short-length-values", default="8,10,12")
    parser.add_argument("--candidate-low-weighted-support-values", default="4,5,6")
    parser.add_argument("--candidate-dangling-max-length-values", default="25,35,45")
    parser.add_argument("--candidate-dangling-min-weighted-support-values", default="6,8,10")
    parser.add_argument("--min-centerline-length-values", default="6,10,14")
    parser.add_argument("--smooth-iterations-values", default="1,2,3")

    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--precision-floor", type=float, default=0.90)
    parser.add_argument("--max-length-ratio", type=float, default=2.0)
    parser.add_argument("--precision-penalty", type=float, default=2.0)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tuning_kharita_recall"))
    parser.add_argument("--save-best-output", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_gdf = _load_ground_truth(args.ground_truth, args.ground_truth_layer)

    space = {
        "cluster_radius_m": _parse_float_list(args.cluster_radius_values),
        "heading_tolerance_deg": _parse_float_list(args.heading_tolerance_values),
        "heading_distance_weight_m": _parse_float_list(args.heading_distance_weight_values),
        "min_cluster_points": _parse_int_list(args.min_cluster_points_values),
        "sample_spacing_m": _parse_float_list(args.sample_spacing_values),
        "max_points_per_trace": _parse_int_list(args.max_points_per_trace_values),
        "vpd_base_weight": _parse_float_list(args.vpd_base_weight_values),
        "hpd_base_weight": _parse_float_list(args.hpd_base_weight_values),
        "max_transition_distance_m": _parse_float_list(args.max_transition_values),
        "min_edge_support": _parse_float_list(args.min_edge_support_values),
        "reverse_edge_ratio": _parse_float_list(args.reverse_edge_ratio_values),
        "enable_transitive_pruning": _parse_bool_list(args.enable_transitive_pruning_values),
        "transitive_max_hops": _parse_int_list(args.transitive_max_hops_values),
        "transitive_ratio": _parse_float_list(args.transitive_ratio_values),
        "transitive_max_checks": _parse_int_list(args.transitive_max_checks_values),
        "use_turn_preserving_smoothing": _parse_bool_list(args.use_turn_smoothing_values),
        "turn_smoothing_deg": _parse_float_list(args.turn_smoothing_deg_values),
        "turn_smoothing_neighbor_weight": _parse_float_list(args.turn_smoothing_neighbor_weight_values),
        "apply_candidate_selection": _parse_bool_list(args.apply_candidate_selection_values),
        "candidate_selection_threshold": _parse_float_list(args.candidate_selection_threshold_values),
        "candidate_length_scale_m": _parse_float_list(args.candidate_length_scale_values),
        "candidate_density_scale": _parse_float_list(args.candidate_density_scale_values),
        "candidate_force_keep_weighted_support": _parse_float_list(args.candidate_force_keep_support_values),
        "candidate_short_length_m": _parse_float_list(args.candidate_short_length_values),
        "candidate_low_weighted_support": _parse_float_list(args.candidate_low_weighted_support_values),
        "candidate_dangling_max_length_m": _parse_float_list(args.candidate_dangling_max_length_values),
        "candidate_dangling_min_weighted_support": _parse_float_list(args.candidate_dangling_min_weighted_support_values),
        "min_centerline_length_m": _parse_float_list(args.min_centerline_length_values),
        "smooth_iterations": _parse_int_list(args.smooth_iterations_values),
    }

    trials = _build_trials(space=space, search=args.search, n_trials=args.trials, seed=args.seed)
    rows = []
    best_row: dict | None = None
    best_params: dict | None = None

    print(f"Running {len(trials)} recall tuning trials ({args.search}).")
    for idx, p in enumerate(trials, start=1):
        t0 = time.time()
        try:
            cfg = _build_config(p)
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
            metrics = _evaluate_result(
                result,
                gt_gdf,
                bbox_file=args.bbox_file,
                apply_bbox=args.apply_bbox,
                clip_generated_to_ground_truth=args.clip_generated_to_ground_truth,
                clip_buffer_m=args.clip_buffer_m,
                buffer_m=args.buffer_m,
                sample_step_m=args.sample_step_m,
            )
            score, precision, recall, length_ratio = _score_trial(
                metrics,
                precision_floor=args.precision_floor,
                max_length_ratio=args.max_length_ratio,
                precision_penalty=args.precision_penalty,
                length_penalty=args.length_penalty,
            )
            row = {
                "trial": idx,
                **p,
                "centerline_count": int(len(result.get("centerlines", []))),
                "trace_count": int(result.get("trace_count", 0)),
                "sample_point_count": int(result.get("sample_point_count", 0)),
                "length_precision": precision,
                "length_recall": recall,
                "length_f1": float(metrics.get("length_f1", 0.0) or 0.0),
                "length_ratio": length_ratio,
                "objective_score": score,
                "elapsed_sec": float(time.time() - t0),
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            if best_row is None or row["objective_score"] > best_row["objective_score"]:
                best_row = row
                best_params = p.copy()
            print(
                f"[{idx}/{len(trials)}] score={score:.5f} recall={recall:.4f} "
                f"precision={precision:.4f} ratio={length_ratio:.3f}"
            )
        except Exception as ex:
            rows.append(
                {
                    "trial": idx,
                    **p,
                    "centerline_count": -1,
                    "trace_count": 0,
                    "sample_point_count": 0,
                    "length_precision": 0.0,
                    "length_recall": 0.0,
                    "length_f1": 0.0,
                    "length_ratio": float("inf"),
                    "objective_score": -1e9,
                    "elapsed_sec": float(time.time() - t0),
                    "status": "failed",
                    "error": str(ex),
                }
            )
            print(f"[{idx}/{len(trials)}] failed: {ex}")

    df = pd.DataFrame(rows).sort_values(["objective_score", "length_recall"], ascending=False)
    results_csv = args.out_dir / "tuning_results.csv"
    df.to_csv(results_csv, index=False)

    summary = {
        "search": args.search,
        "trials_requested": int(args.trials),
        "trials_executed": int(len(trials)),
        "precision_floor": float(args.precision_floor),
        "max_length_ratio": float(args.max_length_ratio),
        "best": best_row,
    }
    summary_json = args.out_dir / "tuning_best.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved trial table: {results_csv}")
    print(f"Saved best summary: {summary_json}")
    if best_params is None:
        return

    comparison_rows: list[dict] = []
    variants = [
        ("baseline_raw", {"apply_candidate_selection": False, "use_turn_preserving_smoothing": False}),
        ("selected", {"apply_candidate_selection": True, "use_turn_preserving_smoothing": False}),
        ("selected_turn_smoothing", {"apply_candidate_selection": True, "use_turn_preserving_smoothing": True}),
    ]
    for name, override in variants:
        pv = {**best_params, **override}
        cfg = _build_config(pv)
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
        metrics = _evaluate_result(
            result,
            gt_gdf,
            bbox_file=args.bbox_file,
            apply_bbox=args.apply_bbox,
            clip_generated_to_ground_truth=args.clip_generated_to_ground_truth,
            clip_buffer_m=args.clip_buffer_m,
            buffer_m=args.buffer_m,
            sample_step_m=args.sample_step_m,
        )
        length_ratio = (
            float(metrics.get("generated_total_length_m", 0.0) or 0.0)
            / max(float(metrics.get("ground_truth_total_length_m", 0.0) or 0.0), 1e-6)
        )
        comparison_rows.append(
            {
                "variant": name,
                "centerline_count": int(len(result.get("centerlines", []))),
                "length_precision": float(metrics.get("length_precision", 0.0) or 0.0),
                "length_recall": float(metrics.get("length_recall", 0.0) or 0.0),
                "length_f1": float(metrics.get("length_f1", 0.0) or 0.0),
                "length_ratio": float(length_ratio),
                "generated_total_length_m": float(metrics.get("generated_total_length_m", 0.0) or 0.0),
                "ground_truth_total_length_m": float(metrics.get("ground_truth_total_length_m", 0.0) or 0.0),
            }
        )
        if args.save_best_output:
            stem = f"best_{name}_centerlines"
            save_centerline_outputs(result=result, output_dir=args.out_dir, stem=stem)

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = args.out_dir / "comparison_metrics.csv"
    comp_df.to_csv(comp_csv, index=False)
    comp_json = args.out_dir / "comparison_metrics.json"
    comp_json.write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
    print(f"Saved comparison table: {comp_csv}")
    print(f"Saved comparison json: {comp_json}")


if __name__ == "__main__":
    main()
