"""
Exploratory Data Analysis (EDA) plotting helpers.

Functions to visualize spatial coverage and statistical profiles of
VPD, HPD, and Nav Streets datasets.
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)

FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_spatial_coverage(
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot spatial coverage of the three datasets.

    Generates a 1x3 subplot figure:
      1. VPD (Fused) Paths
      2. HPD (Probe) Traces
      3. Nav Streets (Reference)
    """
    logger.info("Generating spatial coverage map...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharex=True, sharey=True)

    # 1. VPD
    ax = axes[0]
    if not vpd_gdf.empty:
        # Plot a subset if too large for speed
        sample = vpd_gdf.sample(min(len(vpd_gdf), 10000))
        sample.plot(ax=ax, color="blue", linewidth=0.5, alpha=0.3, label="VPD Fused")
    ax.set_title(f"VPD (Fused) Coverage\n(n={len(vpd_gdf)})")
    ax.set_axis_off()

    # 2. HPD
    ax = axes[1]
    if not hpd_gdf.empty:
        hpd_gdf.plot(ax=ax, color="red", linewidth=0.5, alpha=0.3, label="HPD Traces")
    ax.set_title(f"HPD (Probe) Coverage\n(n={len(hpd_gdf)})")
    ax.set_axis_off()

    # 3. Nav Streets
    ax = axes[2]
    if not nav_gdf.empty:
        nav_gdf.plot(
            ax=ax, color="black", linewidth=1.0, alpha=0.7, label="Nav Streets"
        )
    ax.set_title(f"Nav Streets (Reference)\n(n={len(nav_gdf)})")
    ax.set_axis_off()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved spatial coverage map to {save_path}")
    plt.close(fig)


def plot_distributions(
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """
    Plot statistical distributions of key attributes.
    """
    logger.info("Generating statistical distribution plots...")

    # VPD: Path Length & Quality
    if not vpd_gdf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Length
        if "lengthm" in vpd_gdf.columns:
            vpd_gdf["lengthm"].hist(
                ax=axes[0], bins=50, color="skyblue", edgecolor="black"
            )
            axes[0].set_title("VPD Path Length (m)")
            axes[0].set_xlabel("Length (m)")

        # Path Quality Score
        if "pathqualityscore" in vpd_gdf.columns:
            # Check if numeric, coerce if needed (it might be categorical or mixed)
            try:
                pd.to_numeric(vpd_gdf["pathqualityscore"], errors="coerce").hist(
                    ax=axes[1], bins=20, color="orange", edgecolor="black"
                )
                axes[1].set_title("VPD Path Quality Score")
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig(save_dir / "vpd_distributions.png")
        plt.close(fig)

    # HPD: Speed & Point Count
    if not hpd_gdf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Speed
        if "avg_speed" in hpd_gdf.columns:
            hpd_gdf["avg_speed"].hist(
                ax=axes[0], bins=30, color="salmon", edgecolor="black"
            )
            axes[0].set_title(
                "HPD Average Speed (km/h?)"
            )  # Unit assumption needs checking

        # Point Count
        if "point_count" in hpd_gdf.columns:
            hpd_gdf["point_count"].hist(
                ax=axes[1], bins=30, range=(0, 100), color="purple", edgecolor="black"
            )
            axes[1].set_title("HPD Points per Trace (0-100)")

        plt.tight_layout()
        plt.savefig(save_dir / "hpd_distributions.png")
        plt.close(fig)

    # Nav Streets: Functional Class
    if not nav_gdf.empty and "func_class" in nav_gdf.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        nav_gdf["func_class"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="green", edgecolor="black"
        )
        ax.set_title("Nav Streets: Functional Class Distribution")
        ax.set_xlabel("Class (1=Major, 5=Minor)")
        plt.tight_layout()
        plt.savefig(save_dir / "nav_distributions.png")
        plt.close(fig)


def calculate_overlap_stats(
    vpd_gdf: gpd.GeoDataFrame, nav_gdf: gpd.GeoDataFrame, buffer_m: float = 10.0
) -> dict:
    """
    Calculate rough overlap between VPD (new data) and Nav Streets (reference).

    Returns percentage of VPD length that falls within `buffer_m` of a Nav Street.
    This helps identify 'new roads' (low overlap).
    """
    if vpd_gdf.empty or nav_gdf.empty:
        return {}

    # Project to metric CRS for buffering (approximate UTM for Kosovo - EPSG:32634)
    # 21E is UTM zone 34N.
    target_crs = "EPSG:32634"

    try:
        vpd_proj = vpd_gdf.to_crs(target_crs)
        nav_proj = nav_gdf.to_crs(target_crs)

        # Create a single polygon of the road network buffer
        logger.info(f"Buffering Nav Streets by {buffer_m}m...")
        nav_buffer = nav_proj.geometry.buffer(buffer_m).unary_union

        # Calculate intersection
        logger.info("Calculating intersection of VPD with Nav buffer...")
        vpd_in_nav = vpd_proj.geometry.intersection(nav_buffer)

        total_len = vpd_proj.geometry.length.sum()
        covered_len = vpd_in_nav.length.sum()

        overlap_pct = (covered_len / total_len) * 100 if total_len > 0 else 0

        return {
            "total_vpd_length_km": total_len / 1000.0,
            "covered_vpd_length_km": covered_len / 1000.0,
            "overlap_percentage": overlap_pct,
            "potential_new_road_km": (total_len - covered_len) / 1000.0,
        }
    except Exception as e:
        logger.error(f"Overlap calculation failed: {e}")
        return {}
