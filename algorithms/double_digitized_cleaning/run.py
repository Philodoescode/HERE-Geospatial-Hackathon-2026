"""
Runner script for Double-Digitized Geometry Cleaning Algorithm

This script applies the cleaning algorithm to the input GeoPackage file
and outputs the cleaned network.

Usage:
    python run.py [input_file] [output_file]

If no arguments provided, uses default paths.

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


def run_cleaning(
    input_file: str, output_file: str, config: CleaningConfig | None = None
) -> gpd.GeoDataFrame:
    """
    Run the double-digitized geometry cleaning algorithm.

    Args:
        input_file: Path to input GeoPackage file
        output_file: Path to output GeoPackage file
        config: Optional cleaning configuration

    Returns:
        Cleaned GeoDataFrame
    """
    print(f"\n{'=' * 60}")
    print("Double-Digitized Geometry Cleaning")
    print(f"{'=' * 60}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Load input data
    print("Loading input data...")
    try:
        gdf = gpd.read_file(input_file)
    except Exception as e:
        print(f"Error loading input file: {e}")
        raise

    print(f"Loaded {len(gdf)} geometries")
    print(f"CRS: {gdf.crs}")

    # Run cleaning algorithm
    if config is None:
        config = CleaningConfig()

    cleaned_gdf = clean_double_digitized(gdf, config)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned data
    print(f"\nSaving cleaned data to {output_file}...")
    cleaned_gdf.to_file(output_file, driver="GPKG")
    print("Saved successfully!")

    return cleaned_gdf


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Double-Digitized Geometry Cleaning Algorithm"
    )
    parser.add_argument(
        "input_file", nargs="?", default=None, help="Input GeoPackage file path"
    )
    parser.add_argument(
        "output_file", nargs="?", default=None, help="Output GeoPackage file path"
    )
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
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="B-spline smoothing factor (default: 0.5)",
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
        output_file = output_dir / "kharita_double_digitized_cleaned.gpkg"
    else:
        output_file = Path(args.output_file)

    # Create configuration
    config = CleaningConfig(
        highway_threshold=args.highway_threshold,
        urban_threshold=args.urban_threshold,
        mean_shift_bandwidth=args.bandwidth,
        spline_smoothing=args.smoothing,
    )

    # Run cleaning
    try:
        cleaned_gdf = run_cleaning(str(input_file), str(output_file), config=config)
        print("\nCleaning completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
