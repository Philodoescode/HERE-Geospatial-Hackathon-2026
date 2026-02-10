from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.evaluation import evaluate_centerlines, save_metrics


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
    parser.add_argument("--out", type=Path, default=Path("outputs/evaluation/centerline_metrics.json"))
    args = parser.parse_args()

    gt = args.ground_truth or find_default_ground_truth()

    metrics = evaluate_centerlines(
        generated_centerlines=args.generated,
        ground_truth_navstreet=gt,
        generated_layer=args.generated_layer,
        ground_truth_layer=args.ground_truth_layer,
        buffer_m=args.buffer_m,
        sample_step_m=args.sample_step_m,
    )
    save_metrics(metrics, args.out)

    print(f"Saved evaluation metrics: {args.out}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
