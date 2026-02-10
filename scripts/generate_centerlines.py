from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.generation import (
    CenterlineConfig,
    generate_centerlines,
    save_centerline_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate road centerlines from VPD (WKT LINESTRING) and HPD probe data "
            "using a Kharita-inspired clustering + co-occurrence graph pipeline."
        )
    )
    parser.add_argument(
        "--vpd-csv", type=Path, default=Path("data/Kosovo_VPD/Kosovo_VPD.csv")
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/centerlines"))
    parser.add_argument("--stem", default="kosovo_centerlines")
    parser.add_argument(
        "--fused-only", action=argparse.BooleanOptionalAction, default=True
    )

    # Tuning knobs
    parser.add_argument("--cluster-radius-m", type=float, default=10.0)
    parser.add_argument("--heading-tolerance-deg", type=float, default=45.0)
    parser.add_argument("--sample-spacing-m", type=float, default=8.0)
    parser.add_argument("--max-points-per-trace", type=int, default=120)
    parser.add_argument("--min-edge-support", type=float, default=2.0)
    parser.add_argument("--smooth-iterations", type=int, default=2)

    # Optional subsampling for rapid experiments
    parser.add_argument("--max-vpd-rows", type=int, default=None)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=None)

    args = parser.parse_args()

    config = CenterlineConfig(
        cluster_radius_m=args.cluster_radius_m,
        heading_tolerance_deg=args.heading_tolerance_deg,
        sample_spacing_m=args.sample_spacing_m,
        max_points_per_trace=args.max_points_per_trace,
        min_edge_support=args.min_edge_support,
        smooth_iterations=args.smooth_iterations,
    )

    result = generate_centerlines(
        vpd_csv=args.vpd_csv,
        hpd_csvs=args.hpd_csvs,
        config=config,
        fused_only=args.fused_only,
        max_vpd_rows=args.max_vpd_rows,
        max_hpd_rows_per_file=args.max_hpd_rows_per_file,
    )
    files = save_centerline_outputs(
        result=result, output_dir=args.out_dir, stem=args.stem
    )

    print("Centerline generation completed.")
    for k, v in files.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
