"""
Script to perform headless EDA.

Loads data, generates plots in outputs/figures/, and prints statistical summaries.
"""

import logging
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda_runner")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.loaders import load_hpd, load_nav_streets, load_vpd
from src.visualization.eda import (
    calculate_overlap_stats,
    plot_distributions,
    plot_spatial_coverage,
)
from src.config import OUTPUTS_DIR


def main():
    t0 = time.time()

    # 1. Load Data
    logger.info("=== Loading Data ===")
    nav_gdf = load_nav_streets()
    hpd_gdf = load_hpd()
    vpd_gdf = load_vpd()  # This takes ~10 mins unoptimized, be patient

    logger.info(f"Loaded: Nav={len(nav_gdf)}, HPD={len(hpd_gdf)}, VPD={len(vpd_gdf)}")

    # 2. Generate Plots
    logger.info("=== Generating Plots ===")
    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Coverage Map
    plot_spatial_coverage(
        vpd_gdf,
        hpd_gdf,
        nav_gdf,
        save_path=figures_dir / "spatial_coverage_comparison.png",
    )

    # Distributions
    plot_distributions(vpd_gdf, hpd_gdf, nav_gdf, save_dir=figures_dir)

    # 3. Calculate Stats & Overlap
    logger.info("=== Calculating Statistics ===")

    print("\n--- EDA Report ---\n")

    # VPD Stats
    print(f"VPD (Vehicle Paths):")
    print(f"  Count: {len(vpd_gdf)}")
    if "lengthm" in vpd_gdf.columns:
        print(f"  Avg Length: {vpd_gdf['lengthm'].mean():.1f} m")
    if "pathqualityscore" in vpd_gdf.columns:
        print(f"  Avg Quality: {vpd_gdf['pathqualityscore'].mean():.2f}")

    # HPD Stats
    print(f"\nHPD (Probes):")
    print(f"  Count: {len(hpd_gdf)}")
    if "avg_speed" in hpd_gdf.columns:
        print(f"  Avg Speed: {hpd_gdf['avg_speed'].mean():.1f} km/h")
    if "point_count" in hpd_gdf.columns:
        print(f"  Avg Points/Trace: {hpd_gdf['point_count'].mean():.1f}")

    # Overlap Analysis
    # We want to know: "How much of VPD is NOT covered by Nav Streets?"
    # This is the "New Road" potential.
    logger.info("Running overlap analysis (this may take a moment)...")
    overlap = calculate_overlap_stats(vpd_gdf, nav_gdf, buffer_m=15.0)  # 15m tolerance

    if overlap:
        print(f"\nOverlap Analysis (VPD vs Nav Streets, 15m buffer):")
        print(
            f"  Total VPD Length:      {overlap.get('total_vpd_length_km', 0):.2f} km"
        )
        print(
            f"  Covered by Nav Streets: {overlap.get('covered_vpd_length_km', 0):.2f} km"
        )
        print(f"  Overlap Percentage:    {overlap.get('overlap_percentage', 0):.1f}%")
        print(
            f"  Potential New Roads:   {overlap.get('potential_new_road_km', 0):.2f} km"
        )

    print("\n--- End Report ---\n")
    logger.info(f"EDA completed in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
