from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.evaluation import (
    evaluate_centerlines,
    generate_evaluation_plots,
    save_metrics,
)


def find_default_ground_truth() -> Path:
    prepared = Path("outputs/ground_truth/kosovo_navstreet_ground_truth.gpkg")
    if prepared.exists():
        return prepared
    fallback = next(Path("data").glob("*nav streets*/Kosovo.csv"), None)
    if fallback is None:
        raise FileNotFoundError("Could not find a ground truth navstreet file.")
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated centerlines against HERE navstreet ground truth.")
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path("outputs/centerlines/kosovo_centerlines.gpkg"),
        help="Generated centerlines file (GPKG/GeoJSON/CSV).",
    )
    parser.add_argument(
        "--generated-layer",
        default="centerlines",
        help="Layer name for generated file if using GPKG.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Ground truth file (prepared GPKG or raw navstreet CSV).",
    )
    parser.add_argument("--ground-truth-layer", default="navstreet")
    parser.add_argument("--buffer-m", type=float, default=15.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument(
        "--compute-topology-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute node-degree similarity and intersection-location error metrics.",
    )
    parser.add_argument(
        "--compute-hausdorff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute optional Hausdorff shape-similarity summary.",
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=Path("data/Kosovo_bounding_box.txt"),
        help="Path to WGS84 bbox text file.",
    )
    parser.add_argument(
        "--apply-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip generated and GT geometries to bbox before scoring.",
    )
    parser.add_argument(
        "--clip-generated-to-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Additionally clip generated geometries to ground-truth extent for aligned scoring.",
    )
    parser.add_argument(
        "--clip-buffer-m",
        type=float,
        default=0.0,
        help="Buffer applied to GT extent when --clip-generated-to-ground-truth is enabled.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluation/centerline_metrics.json"))
    parser.add_argument(
        "--plots-out-dir",
        type=Path,
        default=None,
        help="Optional directory for evaluation plots (overlay + mismatch).",
    )
    parser.add_argument(
        "--plot-stem",
        default="centerline_eval",
        help="Filename stem for plot outputs.",
    )
    args = parser.parse_args()

    gt = args.ground_truth or find_default_ground_truth()

    metrics = evaluate_centerlines(
        generated_centerlines=args.generated,
        ground_truth_navstreet=gt,
        generated_layer=args.generated_layer,
        ground_truth_layer=args.ground_truth_layer,
        buffer_m=args.buffer_m,
        sample_step_m=args.sample_step_m,
        bbox_file=args.bbox_file,
        apply_bbox=args.apply_bbox,
        clip_generated_to_ground_truth=args.clip_generated_to_ground_truth,
        clip_buffer_m=args.clip_buffer_m,
        compute_topology_metrics=args.compute_topology_metrics,
        compute_hausdorff=args.compute_hausdorff,
    )
    save_metrics(metrics, args.out)

    plot_files = None
    if args.plots_out_dir is not None:
        plot_files = generate_evaluation_plots(
            generated_centerlines=args.generated,
            ground_truth_navstreet=gt,
            out_dir=args.plots_out_dir,
            generated_layer=args.generated_layer,
            ground_truth_layer=args.ground_truth_layer,
            buffer_m=args.buffer_m,
            stem=args.plot_stem,
        )

    print(f"Saved evaluation metrics: {args.out}")
    if plot_files is not None:
        for k, v in plot_files.items():
            print(f"{k}: {v}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
