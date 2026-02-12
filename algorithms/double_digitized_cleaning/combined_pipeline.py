"""
Combined Double-Digitized Cleaning Pipeline

REVERSED LOGIC - Correct Order:
1. FIRST: Remove opposite-direction clashing lines (GPS noise)
2. THEN: Clean same-direction duplicates

This ensures:
- Realistic road direction patterns (no mid-road direction flips)
- Better connectivity preservation
- Aggressive removal of inference noise

Author: Augment Code
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algorithms.double_digitized_cleaning.cleaner import (
    CleaningConfig,
    clean_double_digitized,
)
from algorithms.double_digitized_cleaning.opposite_direction_dedup import (
    OppositeDirectionConfig,
    deduplicate_opposite_directions,
)


def run_combined_pipeline(
    input_file: str,
    output_file: str,
    opposite_config: OppositeDirectionConfig | None = None,
    same_dir_config: CleaningConfig | None = None,
) -> gpd.GeoDataFrame:
    """
    Run the combined cleaning pipeline in correct order.

    Order:
    1. Remove opposite-direction clashing lines (aggressive)
    2. Clean same-direction duplicates

    Args:
        input_file: Path to input GeoPackage file
        output_file: Path to output GeoPackage file
        opposite_config: Configuration for opposite-direction cleaning
        same_dir_config: Configuration for same-direction cleaning

    Returns:
        Final cleaned GeoDataFrame
    """
    print("\n" + "=" * 80)
    print("COMBINED DOUBLE-DIGITIZED CLEANING PIPELINE")
    print("=" * 80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nPIPELINE ORDER:")
    print("  1. Remove opposite-direction clashing lines (AGGRESSIVE)")
    print("  2. Clean same-direction duplicates")
    print("=" * 80 + "\n")

    # Load input data
    print("Loading input data...")
    try:
        gdf = gpd.read_file(input_file)
    except Exception as e:
        print(f"Error loading input file: {e}")
        raise

    print(f"Loaded {len(gdf)} geometries")
    print(f"CRS: {gdf.crs}")
    input_count = len(gdf)

    # Initialize configs
    if opposite_config is None:
        opposite_config = OppositeDirectionConfig()

    if same_dir_config is None:
        same_dir_config = CleaningConfig()

    # =========================================================================
    # STEP 1: Remove Opposite-Direction Clashing Lines (AGGRESSIVE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: OPPOSITE-DIRECTION CLASHING LINE REMOVAL")
    print("=" * 80)
    print("This step removes GPS noise and inference errors that create")
    print("unrealistic opposite-direction segments on the same road.")
    print("=" * 80 + "\n")

    step1_output = deduplicate_opposite_directions(gdf, opposite_config)

    step1_reduction = input_count - len(step1_output)
    step1_pct = 100 * step1_reduction / input_count if input_count > 0 else 0

    print(f"\nStep 1 Results:")
    print(f"  Input: {input_count} geometries")
    print(f"  Output: {len(step1_output)} geometries")
    print(f"  Removed: {step1_reduction} geometries ({step1_pct:.1f}%)")

    # =========================================================================
    # STEP 2: Clean Same-Direction Duplicates
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: SAME-DIRECTION DUPLICATE CLEANING")
    print("=" * 80)
    print("This step removes redundant same-direction segments while")
    print("preserving legitimate bidirectional lanes.")
    print("=" * 80 + "\n")

    step2_input_count = len(step1_output)
    step2_output = clean_double_digitized(step1_output, same_dir_config)

    step2_reduction = step2_input_count - len(step2_output)
    step2_pct = (
        100 * step2_reduction / step2_input_count if step2_input_count > 0 else 0
    )

    print(f"\nStep 2 Results:")
    print(f"  Input: {step2_input_count} geometries")
    print(f"  Output: {len(step2_output)} geometries")
    print(f"  Removed: {step2_reduction} geometries ({step2_pct:.1f}%)")

    # =========================================================================
    # Save Final Output
    # =========================================================================
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    total_reduction = input_count - len(step2_output)
    total_pct = 100 * total_reduction / input_count if input_count > 0 else 0

    print(f"  Original Input: {input_count} geometries")
    print(f"  Final Output: {len(step2_output)} geometries")
    print(f"  Total Removed: {total_reduction} geometries ({total_pct:.1f}%)")
    print(f"\n  Breakdown:")
    print(f"    - Opposite-direction clashing: {step1_reduction} ({step1_pct:.1f}%)")
    print(f"    - Same-direction duplicates: {step2_reduction} ({step2_pct:.1f}%)")
    print("=" * 80 + "\n")

    print(f"Saving final output to {output_file}...")
    step2_output.to_file(output_file, driver="GPKG")
    print("Saved successfully!")

    return step2_output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Combined Double-Digitized Cleaning Pipeline (Opposite-Dir First)"
    )
    parser.add_argument(
        "input_file", nargs="?", default=None, help="Input GeoPackage file path"
    )
    parser.add_argument(
        "output_file", nargs="?", default=None, help="Output GeoPackage file path"
    )

    # Opposite-direction config
    parser.add_argument(
        "--opp-proximity",
        type=float,
        default=8.0,
        help="Opposite-direction spatial search radius in meters (default: 8.0)",
    )
    parser.add_argument(
        "--opp-merge-threshold",
        type=float,
        default=5.0,
        help="Opposite-direction merge threshold in meters (default: 5.0)",
    )

    # Same-direction config
    parser.add_argument(
        "--highway-threshold",
        type=float,
        default=3.6,
        help="Highway duplicate detection threshold in meters (default: 3.6)",
    )
    parser.add_argument(
        "--urban-threshold",
        type=float,
        default=3.3,
        help="Urban duplicate detection threshold in meters (default: 3.3)",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=5.0,
        help="Mean Shift bandwidth in meters (default: 5.0)",
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    if args.input_file is None:
        input_file = (
            project_root
            / "outputs"
            / "kharita_full_tuned_no_deepmg"
            / "kharita_full_tuned_no_deepmg.gpkg"
        )
    else:
        input_file = Path(args.input_file)

    if args.output_file is None:
        output_dir = project_root / "outputs" / "double_digitized_cleaned"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "kharita_combined_cleaned.gpkg"
    else:
        output_file = Path(args.output_file)

    # Create configurations
    opposite_config = OppositeDirectionConfig()
    opposite_config.max_proximity = args.opp_proximity
    opposite_config.merge_distance_threshold = args.opp_merge_threshold

    same_dir_config = CleaningConfig(
        highway_threshold=args.highway_threshold,
        urban_threshold=args.urban_threshold,
        mean_shift_bandwidth=args.bandwidth,
    )

    # Run pipeline
    try:
        cleaned_gdf = run_combined_pipeline(
            str(input_file),
            str(output_file),
            opposite_config=opposite_config,
            same_dir_config=same_dir_config,
        )
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
