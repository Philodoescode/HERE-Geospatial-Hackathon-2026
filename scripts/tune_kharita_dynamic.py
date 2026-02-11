from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.algorithms.kharita import KharitaAlgorithm, KharitaConfig
from centerline.evaluation import (
    EvaluationContext,
    build_evaluation_context,
    evaluate_centerline_geodataframes,
)
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
    vals = []
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


def _load_ground_truth(path: Path, layer: str) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".csv":
        gt_df = load_navstreet_csv(path)
        return gpd.GeoDataFrame(gt_df, geometry="geometry", crs="EPSG:4326")
    gt = gpd.read_file(path, layer=layer)
    if gt.crs is None:
        gt = gt.set_crs("EPSG:4326")
    return gt


def _get_nested_float(payload: dict, path: list[str], default: float = 0.0) -> float:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return float(default)
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return float(default)


def _evaluate_result(
    result: dict,
    gt_gdf: gpd.GeoDataFrame,
    *,
    eval_context: EvaluationContext | None,
    bbox_file: Path | None,
    apply_bbox: bool,
    buffer_m: float,
    sample_step_m: float,
    topology_radii_m: tuple[float, ...],
) -> dict:
    center = result.get("centerlines", pd.DataFrame())
    if len(center) == 0:
        return {
            "generated_count": 0,
            "ground_truth_count": int(len(gt_gdf)),
            "generated_total_length_m": 0.0,
            "ground_truth_total_length_m": 0.0,
            "length_precision": 0.0,
            "length_recall": 0.0,
            "length_f1": 0.0,
            "topology": {},
        }
    center_gdf = gpd.GeoDataFrame(center.copy(), geometry="geometry", crs="EPSG:4326")
    return evaluate_centerline_geodataframes(
        generated_gdf=center_gdf,
        ground_truth_gdf=gt_gdf,
        buffer_m=buffer_m,
        sample_step_m=sample_step_m,
        bbox_file=bbox_file,
        apply_bbox=apply_bbox,
        clip_generated_to_ground_truth=False,
        clip_buffer_m=0.0,
        context=eval_context,
        compute_topology_metrics=True,
        compute_itopo=True,
        topology_radii_m=topology_radii_m,
        compute_hausdorff=False,
    )


def _trial_score(
    metrics: dict,
    *,
    precision_floor: float,
    max_length_ratio: float,
    precision_penalty: float,
    length_penalty: float,
) -> tuple[float, float, float, float]:
    precision = float(metrics.get("length_precision", 0.0) or 0.0)
    recall = float(metrics.get("length_recall", 0.0) or 0.0)
    f1 = float(metrics.get("length_f1", 0.0) or 0.0)
    generated_len = float(metrics.get("generated_total_length_m", 0.0) or 0.0)
    gt_len = float(metrics.get("ground_truth_total_length_m", 0.0) or 0.0)
    length_ratio = (generated_len / gt_len) if gt_len > 0 else float("inf")
    p_gap = max(0.0, precision_floor - precision)
    lr_gap = max(0.0, length_ratio - max_length_ratio)
    score = f1 - precision_penalty * (p_gap ** 2) - length_penalty * (lr_gap ** 2)
    return score, precision, recall, length_ratio


def _build_config(p: dict) -> KharitaConfig:
    return KharitaConfig(
        cluster_radius_m=float(p["cluster_radius_m"]),
        heading_tolerance_deg=float(p["heading_tolerance_deg"]),
        heading_distance_weight_m=float(p["heading_distance_weight_m"]),
        min_edge_support=float(p["min_edge_support"]),
        reverse_edge_ratio=float(p["reverse_edge_ratio"]),
        sample_spacing_m=float(p["sample_spacing_m"]),
        max_points_per_trace=int(p["max_points_per_trace"]),
        candidate_selection_threshold=float(p["candidate_selection_threshold"]),
        candidate_density_scale=float(p["candidate_density_scale"]),
        enable_dynamic_weighting=bool(p["enable_dynamic_weighting"]),
        dyn_lambda_vpd=float(p["dyn_lambda_vpd"]),
        dyn_road_likeness_beta=float(p["dyn_road_likeness_beta"]),
        dyn_road_likeness_tau=float(p["dyn_road_likeness_tau"]),
        enable_deepmg_topology_postprocess=bool(p["enable_deepmg_topology_postprocess"]),
        post_link_radius_m=float(p["post_link_radius_m"]) if p["post_link_radius_m"] >= 0 else None,
        post_alpha_virtual=float(p["post_alpha_virtual"]),
        post_min_supp_virtual=int(p["post_min_supp_virtual"]) if int(p["post_min_supp_virtual"]) >= 0 else None,
        post_mm_snap_m=float(p["post_mm_snap_m"]),
        post_dup_eps_m=float(p["post_dup_eps_m"]),
    )


def _run_variant(
    *,
    cfg: KharitaConfig,
    vpd_csv: Path,
    hpd_csvs: list[Path],
    bbox_file: Path | None,
    apply_bbox: bool,
    max_vpd_rows: int | None,
    max_hpd_rows_per_file: int | None,
    gt_gdf: gpd.GeoDataFrame,
    eval_context: EvaluationContext | None,
    buffer_m: float,
    sample_step_m: float,
    topology_radii_m: tuple[float, ...],
) -> tuple[dict, dict]:
    algo = KharitaAlgorithm(config=cfg)
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
    metrics = _evaluate_result(
        result,
        gt_gdf,
        eval_context=eval_context,
        bbox_file=bbox_file,
        apply_bbox=apply_bbox,
        buffer_m=buffer_m,
        sample_step_m=sample_step_m,
        topology_radii_m=topology_radii_m,
    )
    return result, metrics


def main() -> None:
    run_start_epoch = time.time()
    run_start_dt = datetime.now().astimezone()

    def _print_run_timing_footer() -> None:
        run_end_epoch = time.time()
        run_end_dt = datetime.now().astimezone()
        total_elapsed_sec = run_end_epoch - run_start_epoch
        print(f"run_finished_at: {run_end_dt.isoformat()}")
        print(f"total_elapsed_sec: {total_elapsed_sec:.3f}")
        print(f"total_elapsed_min: {total_elapsed_sec / 60.0:.3f}")

    parser = argparse.ArgumentParser(
        description="Tune Kharita dynamic weighting + DeepMG-inspired topology postprocess."
    )
    parser.add_argument("--vpd-csv", type=Path, default=Path("data/Kosovo_VPD/Kosovo_VPD.csv"))
    parser.add_argument(
        "--hpd-csvs",
        type=Path,
        nargs="+",
        default=[Path("data/Kosovo_HPD/XKO_HPD_week_1.csv"), Path("data/Kosovo_HPD/XKO_HPD_week_2.csv")],
    )
    parser.add_argument("--ground-truth", type=Path, default=Path("data/Kosovo's nav streets/nav_kosovo.gpkg"))
    parser.add_argument("--ground-truth-layer", default="Kosovo22")
    parser.add_argument("--bbox-file", type=Path, default=Path("data/Kosovo_bounding_box.txt"))
    parser.add_argument("--apply-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-vpd-rows", type=int, default=1500)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=50000)
    parser.add_argument("--search", choices=["random", "grid"], default="random")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cluster-radius-values", default="10,14")
    parser.add_argument("--heading-tolerance-values", default="40,50")
    parser.add_argument("--heading-distance-weight-values", default="0.22,0.35")
    parser.add_argument("--min-edge-support-values", default="2,3")
    parser.add_argument("--reverse-edge-ratio-values", default="0.1,0.2")
    parser.add_argument("--sample-spacing-values", default="8,10")
    parser.add_argument("--max-points-per-trace-values", default="120,240")
    parser.add_argument("--candidate-selection-threshold-values", default="0.52,0.60")
    parser.add_argument("--candidate-density-scale-values", default="0.25,0.35")

    parser.add_argument("--enable-dynamic-weighting-values", default="true,false")
    parser.add_argument("--dyn-lambda-vpd-values", default="1.2,1.6,2.0")
    parser.add_argument("--dyn-road-likeness-beta-values", default="4,6,8")
    parser.add_argument("--dyn-road-likeness-tau-values", default="0.35,0.45,0.55")

    parser.add_argument("--enable-deepmg-topology-postprocess-values", default="true,false")
    parser.add_argument("--post-link-radius-values", default="-1,50,75,100")  # -1 => auto
    parser.add_argument("--post-alpha-virtual-values", default="1.2,1.4,1.8")
    parser.add_argument("--post-min-supp-virtual-values", default="-1,2,3,5")  # -1 => auto
    parser.add_argument("--post-mm-snap-values", default="12,15,20")
    parser.add_argument("--post-dup-eps-values", default="2,3,4")

    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--precision-floor", type=float, default=0.25)
    parser.add_argument("--max-length-ratio", type=float, default=2.0)
    parser.add_argument("--precision-penalty", type=float, default=1.5)
    parser.add_argument("--length-penalty", type=float, default=0.7)

    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tuning_kharita_dynamic"))
    parser.add_argument("--save-best-output", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_gdf = _load_ground_truth(args.ground_truth, args.ground_truth_layer)
    topology_radii_m = (8.0, 15.0)
    eval_context = build_evaluation_context(
        gt_gdf,
        bbox_file=args.bbox_file,
        apply_bbox=args.apply_bbox,
        sample_step_m=args.sample_step_m,
        topology_radii_m=topology_radii_m,
    )

    space = {
        "cluster_radius_m": _parse_float_list(args.cluster_radius_values),
        "heading_tolerance_deg": _parse_float_list(args.heading_tolerance_values),
        "heading_distance_weight_m": _parse_float_list(args.heading_distance_weight_values),
        "min_edge_support": _parse_float_list(args.min_edge_support_values),
        "reverse_edge_ratio": _parse_float_list(args.reverse_edge_ratio_values),
        "sample_spacing_m": _parse_float_list(args.sample_spacing_values),
        "max_points_per_trace": _parse_int_list(args.max_points_per_trace_values),
        "candidate_selection_threshold": _parse_float_list(args.candidate_selection_threshold_values),
        "candidate_density_scale": _parse_float_list(args.candidate_density_scale_values),
        "enable_dynamic_weighting": _parse_bool_list(args.enable_dynamic_weighting_values),
        "dyn_lambda_vpd": _parse_float_list(args.dyn_lambda_vpd_values),
        "dyn_road_likeness_beta": _parse_float_list(args.dyn_road_likeness_beta_values),
        "dyn_road_likeness_tau": _parse_float_list(args.dyn_road_likeness_tau_values),
        "enable_deepmg_topology_postprocess": _parse_bool_list(args.enable_deepmg_topology_postprocess_values),
        "post_link_radius_m": _parse_float_list(args.post_link_radius_values),
        "post_alpha_virtual": _parse_float_list(args.post_alpha_virtual_values),
        "post_min_supp_virtual": _parse_int_list(args.post_min_supp_virtual_values),
        "post_mm_snap_m": _parse_float_list(args.post_mm_snap_values),
        "post_dup_eps_m": _parse_float_list(args.post_dup_eps_values),
    }
    trials = _build_trials(space=space, search=args.search, n_trials=args.trials, seed=args.seed)

    print(f"run_started_at: {run_start_dt.isoformat()}")
    print(f"trials_planned: {len(trials)}")

    rows = []
    best = None
    for idx, p in enumerate(trials, start=1):
        t0 = time.time()
        try:
            cfg = _build_config(p)
            result, metrics = _run_variant(
                cfg=cfg,
                vpd_csv=args.vpd_csv,
                hpd_csvs=args.hpd_csvs,
                bbox_file=args.bbox_file,
                apply_bbox=args.apply_bbox,
                max_vpd_rows=args.max_vpd_rows,
                max_hpd_rows_per_file=args.max_hpd_rows_per_file,
                gt_gdf=gt_gdf,
                eval_context=eval_context,
                buffer_m=args.buffer_m,
                sample_step_m=args.sample_step_m,
                topology_radii_m=topology_radii_m,
            )
            score, precision, recall, length_ratio = _trial_score(
                metrics,
                precision_floor=args.precision_floor,
                max_length_ratio=args.max_length_ratio,
                precision_penalty=args.precision_penalty,
                length_penalty=args.length_penalty,
            )
            topo_f1_8 = _get_nested_float(metrics, ["topology", "topo", "by_radius_m", "8.0", "f1"])
            topo_f1_15 = _get_nested_float(metrics, ["topology", "topo", "by_radius_m", "15.0", "f1"])
            itopo_f1_8 = _get_nested_float(metrics, ["topology", "i_topo", "by_radius_m", "8.0", "f1"])
            itopo_f1_15 = _get_nested_float(metrics, ["topology", "i_topo", "by_radius_m", "15.0", "f1"])
            row = {
                "trial": idx,
                **p,
                "centerline_count": int(len(result.get("centerlines", pd.DataFrame()))),
                "trace_count": int(result.get("trace_count", 0)),
                "sample_point_count": int(result.get("sample_point_count", 0)),
                "length_precision": precision,
                "length_recall": recall,
                "length_f1": float(metrics.get("length_f1", 0.0) or 0.0),
                "topo_f1_r8": topo_f1_8,
                "topo_f1_r15": topo_f1_15,
                "i_topo_f1_r8": itopo_f1_8,
                "i_topo_f1_r15": itopo_f1_15,
                "length_ratio": length_ratio,
                "objective_score": score,
                "elapsed_sec": time.time() - t0,
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            if best is None or row["objective_score"] > best["objective_score"]:
                best = row
            print(
                f"[trial {idx}/{len(trials)}] score={score:.6f} f1={row['length_f1']:.6f} "
                f"topo8={topo_f1_8:.6f} itopo8={itopo_f1_8:.6f} "
                f"p={precision:.6f} r={recall:.6f} n={row['centerline_count']}"
            )
        except Exception as ex:
            row = {
                "trial": idx,
                **p,
                "centerline_count": -1,
                "trace_count": 0,
                "sample_point_count": 0,
                "length_precision": 0.0,
                "length_recall": 0.0,
                "length_f1": 0.0,
                "topo_f1_r8": 0.0,
                "topo_f1_r15": 0.0,
                "i_topo_f1_r8": 0.0,
                "i_topo_f1_r15": 0.0,
                "length_ratio": float("inf"),
                "objective_score": -1e9,
                "elapsed_sec": time.time() - t0,
                "status": "failed",
                "error": str(ex),
            }
            rows.append(row)
            print(f"[trial {idx}/{len(trials)}] failed: {ex}")

    df = pd.DataFrame(rows).sort_values(["objective_score", "length_f1"], ascending=False)
    csv_path = args.out_dir / "tuning_results.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "search": args.search,
        "trials_requested": int(args.trials),
        "trials_executed": int(len(trials)),
        "precision_floor": float(args.precision_floor),
        "max_length_ratio": float(args.max_length_ratio),
        "best": best,
    }
    summary_path = args.out_dir / "tuning_best.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best is None:
        print(f"No successful trials. Wrote: {csv_path}, {summary_path}")
        _print_run_timing_footer()
        return

    best_cfg = _build_config(best)
    if args.save_best_output:
        best_result, _ = _run_variant(
            cfg=best_cfg,
            vpd_csv=args.vpd_csv,
            hpd_csvs=args.hpd_csvs,
            bbox_file=args.bbox_file,
            apply_bbox=args.apply_bbox,
            max_vpd_rows=args.max_vpd_rows,
            max_hpd_rows_per_file=args.max_hpd_rows_per_file,
            gt_gdf=gt_gdf,
            eval_context=eval_context,
            buffer_m=args.buffer_m,
            sample_step_m=args.sample_step_m,
            topology_radii_m=topology_radii_m,
        )
        files = save_centerline_outputs(best_result, args.out_dir, stem="best_kharita_dynamic")
        print("Saved best outputs:")
        for k, v in files.items():
            print(f"  {k}: {v}")

    # Ablations with best tuned base.
    variants = {
        "baseline": best_cfg.__class__(
            **{**best_cfg.__dict__, "enable_dynamic_weighting": False, "enable_deepmg_topology_postprocess": False}
        ),
        "dynamic_only": best_cfg.__class__(
            **{**best_cfg.__dict__, "enable_dynamic_weighting": True, "enable_deepmg_topology_postprocess": False}
        ),
        "postprocess_only": best_cfg.__class__(
            **{**best_cfg.__dict__, "enable_dynamic_weighting": False, "enable_deepmg_topology_postprocess": True}
        ),
        "combined": best_cfg.__class__(
            **{**best_cfg.__dict__, "enable_dynamic_weighting": True, "enable_deepmg_topology_postprocess": True}
        ),
    }
    ablation_metrics = {}
    for name, cfg in variants.items():
        _, metrics = _run_variant(
            cfg=cfg,
            vpd_csv=args.vpd_csv,
            hpd_csvs=args.hpd_csvs,
            bbox_file=args.bbox_file,
            apply_bbox=args.apply_bbox,
            max_vpd_rows=args.max_vpd_rows,
            max_hpd_rows_per_file=args.max_hpd_rows_per_file,
            gt_gdf=gt_gdf,
            eval_context=eval_context,
            buffer_m=args.buffer_m,
            sample_step_m=args.sample_step_m,
            topology_radii_m=topology_radii_m,
        )
        ablation_metrics[name] = metrics
    ablation_path = args.out_dir / "ablation_metrics.json"
    ablation_path.write_text(json.dumps(ablation_metrics, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {ablation_path}")
    _print_run_timing_footer()


if __name__ == "__main__":
    main()
