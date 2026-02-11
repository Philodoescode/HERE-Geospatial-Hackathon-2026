from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_cmd(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _progress(step: int, total: int, message: str) -> None:
    pct = 100.0 * float(step) / float(max(total, 1))
    print(f"[kharita-run] {step}/{total} ({pct:.1f}%) {message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reprocess updated Kosovo navstreet CSV, run Kharita with VPD parquet, "
            "and output centerline results + evaluation metrics."
        )
    )
    parser.add_argument(
        "--nav-csv",
        type=Path,
        default=Path("data/Kosovo's nav streets/nav_kosovo.csv"),
        help="Updated Kosovo navstreet CSV with geom WKT column.",
    )
    parser.add_argument(
        "--vpd-parquet",
        type=Path,
        default=Path("data/Kosovo_VPD/Kosvo_VPD.parquet"),
        help="Parquet VPD file.",
    )
    parser.add_argument(
        "--hpd-csvs",
        type=Path,
        nargs="+",
        default=[
            Path("data/Kosovo_HPD/XKO_HPD_week_1.csv"),
            Path("data/Kosovo_HPD/XKO_HPD_week_2.csv"),
        ],
    )
    parser.add_argument("--bbox-file", type=Path, default=Path("data/Kosovo_bounding_box.txt"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/kharita_parquet_bbox"))
    parser.add_argument("--ground-truth-stem", default="kosovo_navstreet_ground_truth_full_bbox")
    parser.add_argument("--centerline-stem", default="kharita_full_bbox")
    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--max-vpd-rows", type=int, default=None)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=None)
    parser.add_argument("--fused-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clip-generated-to-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip generated centerlines to GT extent during evaluation.",
    )
    parser.add_argument("--clip-buffer-m", type=float, default=0.0)
    parser.add_argument("--compute-topology-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-hausdorff", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    gt_out = args.out_root / "ground_truth"
    centerline_out = args.out_root / "centerlines"
    eval_out = args.out_root / "evaluation"
    gt_out.mkdir(parents=True, exist_ok=True)
    centerline_out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    gt_gpkg = gt_out / f"{args.ground_truth_stem}.gpkg"
    centerline_gpkg = centerline_out / f"{args.centerline_stem}.gpkg"
    metrics_json = eval_out / f"{args.centerline_stem}_overlap_metrics.json"

    start = time.time()
    total_steps = 3

    _progress(1, total_steps, "Preparing navstreet ground truth from updated CSV.")
    prep_cmd = [
        sys.executable,
        "scripts/prepare_navstreet_ground_truth.py",
        "--nav-csv",
        str(args.nav_csv),
        "--out-dir",
        str(gt_out),
        "--stem",
        args.ground_truth_stem,
    ]
    if args.apply_bbox:
        prep_cmd.extend(["--apply-bbox", "--bbox-file", str(args.bbox_file)])
    else:
        prep_cmd.append("--no-apply-bbox")
    _run_cmd(prep_cmd, cwd=ROOT)

    _progress(2, total_steps, "Running Kharita centerline generation using VPD parquet.")
    gen_cmd = [
        sys.executable,
        "scripts/generate_centerlines.py",
        "--algorithm",
        "kharita",
        "--vpd-csv",
        str(args.vpd_parquet),
        "--hpd-csvs",
        *[str(p) for p in args.hpd_csvs],
        "--out-dir",
        str(centerline_out),
        "--stem",
        args.centerline_stem,
        "--bbox-file",
        str(args.bbox_file),
    ]
    gen_cmd.append("--apply-bbox" if args.apply_bbox else "--no-apply-bbox")
    gen_cmd.append("--fused-only" if args.fused_only else "--no-fused-only")
    if args.max_vpd_rows is not None:
        gen_cmd.extend(["--max-vpd-rows", str(args.max_vpd_rows)])
    if args.max_hpd_rows_per_file is not None:
        gen_cmd.extend(["--max-hpd-rows-per-file", str(args.max_hpd_rows_per_file)])
    _run_cmd(gen_cmd, cwd=ROOT)

    _progress(3, total_steps, "Evaluating generated centerlines against navstreet.")
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_centerlines.py",
        "--generated",
        str(centerline_gpkg),
        "--generated-layer",
        "centerlines",
        "--ground-truth",
        str(gt_gpkg),
        "--ground-truth-layer",
        "navstreet",
        "--buffer-m",
        str(args.buffer_m),
        "--sample-step-m",
        str(args.sample_step_m),
        "--clip-buffer-m",
        str(args.clip_buffer_m),
        "--out",
        str(metrics_json),
    ]
    eval_cmd.append("--apply-bbox" if args.apply_bbox else "--no-apply-bbox")
    eval_cmd.append(
        "--clip-generated-to-ground-truth"
        if args.clip_generated_to_ground_truth
        else "--no-clip-generated-to-ground-truth"
    )
    eval_cmd.append(
        "--compute-topology-metrics"
        if args.compute_topology_metrics
        else "--no-compute-topology-metrics"
    )
    eval_cmd.append("--compute-hausdorff" if args.compute_hausdorff else "--no-compute-hausdorff")
    _run_cmd(eval_cmd, cwd=ROOT)

    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    elapsed = time.time() - start

    print("[kharita-run] complete")
    print(f"[kharita-run] elapsed_s: {elapsed:.1f}")
    print(f"[kharita-run] navstreet_gpkg: {gt_gpkg}")
    print(f"[kharita-run] centerlines_gpkg: {centerline_gpkg}")
    print(f"[kharita-run] metrics_json: {metrics_json}")
    for key in [
        "generated_count",
        "ground_truth_count",
        "generated_total_length_m",
        "ground_truth_total_length_m",
        "length_precision",
        "length_recall",
        "length_f1",
    ]:
        if key in metrics:
            print(f"[kharita-run] {key}: {metrics[key]}")
    gen_len = float(metrics.get("generated_total_length_m", 0.0) or 0.0)
    gt_len = float(metrics.get("ground_truth_total_length_m", 0.0) or 0.0)
    if gt_len > 0:
        print(f"[kharita-run] generated_to_gt_length_ratio: {gen_len / gt_len:.4f}")


if __name__ == "__main__":
    main()
