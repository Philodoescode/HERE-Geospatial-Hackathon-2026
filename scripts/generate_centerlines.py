from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.algorithms import get_algorithm, list_algorithms
from centerline.generation import (
    generate_centerlines_with_algorithm,
    save_centerline_outputs,
)


def main() -> None:
    # ------------------------------------------------------------------
    # Phase 1: parse only --algorithm so we can ask the selected
    # algorithm to register its own CLI flags before the full parse.
    # ------------------------------------------------------------------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--algorithm",
        type=str,
        default="roadster",
        help=(
            "Centerline generation algorithm to use.  "
            f"Available: {', '.join(name for name, _ in list_algorithms())}"
        ),
    )
    pre_parser.add_argument(
        "--list-algorithms",
        action="store_true",
        default=False,
        help="List available algorithms and exit.",
    )
    pre_args, _remaining = pre_parser.parse_known_args()

    if pre_args.list_algorithms:
        print("Available centerline generation algorithms:")
        for name, desc in list_algorithms():
            print(f"  {name:20s}  {desc}")
        sys.exit(0)

    algorithm = get_algorithm(pre_args.algorithm)

    # ------------------------------------------------------------------
    # Phase 2: build the full parser with algorithm-specific arguments.
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description=(
            "Generate road centerlines from VPD and HPD probe data "
            "using a pluggable algorithm system."
        ),
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
        help="Clip input traces to bbox before centerline generation.",
    )

    # Optional subsampling for rapid experiments
    parser.add_argument("--max-vpd-rows", type=int, default=None)
    parser.add_argument("--max-hpd-rows-per-file", type=int, default=None)

    # Let the selected algorithm add its own CLI args
    algorithm.add_cli_args(parser)

    args = parser.parse_args()

    # Let the algorithm read back its parsed CLI args
    algorithm.configure(args)

    # ------------------------------------------------------------------
    # Run the pipeline
    # ------------------------------------------------------------------
    print(f"Algorithm: {algorithm.name}")
    result = generate_centerlines_with_algorithm(
        vpd_csv=args.vpd_csv,
        hpd_csvs=args.hpd_csvs,
        algorithm=algorithm,
        fused_only=args.fused_only,
        max_vpd_rows=args.max_vpd_rows,
        max_hpd_rows_per_file=args.max_hpd_rows_per_file,
        bbox_file=args.bbox_file,
        apply_bbox=args.apply_bbox,
    )
    files = save_centerline_outputs(
        result=result, output_dir=args.out_dir, stem=args.stem
    )

    print("Centerline generation completed.")
    for k, v in files.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
