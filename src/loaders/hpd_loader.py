"""
HPD (Probe) data loader.

Reads the weekly HPD CSVs, converts lat/lon points into per-trace
LineString geometries, and returns a GeoDataFrame of reconstructed traces.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, box

from src.config import BBOX, CRS, HPD_WEEK_1_CSV, HPD_WEEK_2_CSV

logger = logging.getLogger(__name__)


def _reconstruct_traces(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Group individual probe points by ``traceid`` and rebuild each trace
    as a LineString ordered by ``time``.

    Points are sorted by (day, time) within each trace.  Traces with
    fewer than 2 points are dropped (cannot form a line).
    """
    logger.info("Reconstructing traces from %d probe points…", len(df))

    # Sort globally so the groupby preserves temporal order
    df = df.sort_values(["traceid", "day", "time"]).reset_index(drop=True)

    records: list[dict] = []

    for trace_id, group in df.groupby("traceid", sort=False):
        if len(group) < 2:
            continue

        coords = list(zip(group["longitude"], group["latitude"]))
        line = LineString(coords)

        if line.is_empty:
            continue

        # Aggregate scalar attributes
        records.append(
            {
                "traceid": trace_id,
                "geometry": line,
                "point_count": len(group),
                "avg_speed": group["speed"].mean() if "speed" in group.columns else None,
                "avg_heading": group["heading"].mean() if "heading" in group.columns else None,
                "day_start": group["day"].iloc[0],
                "day_end": group["day"].iloc[-1],
                "time_start": group["time"].iloc[0],
                "time_end": group["time"].iloc[-1],
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS)
    logger.info("Reconstructed %d traces.", len(gdf))
    return gdf


def load_hpd(
    week_1_path: Optional[str] = None,
    week_2_path: Optional[str] = None,
    clip_to_bbox: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load and reconstruct the HPD (probe) dataset.

    Steps:
        1. Read both weekly CSVs and concatenate.
        2. Group points by ``traceid``, sorted by time.
        3. Build a LineString per trace.
        4. Optionally clip to the study-area bounding box.
        5. Return a GeoDataFrame in EPSG:4326.

    Parameters
    ----------
    week_1_path, week_2_path : str, optional
        Override paths (default to ``config.HPD_WEEK_*_CSV``).
    clip_to_bbox : bool
        If True, drop traces whose geometry falls entirely outside the bbox.

    Returns
    -------
    gpd.GeoDataFrame
        Reconstructed HPD traces.
    """
    path1 = week_1_path or str(HPD_WEEK_1_CSV)
    path2 = week_2_path or str(HPD_WEEK_2_CSV)

    logger.info("Loading HPD week 1 from %s…", path1)
    df1 = pd.read_csv(path1)
    logger.info("Loading HPD week 2 from %s…", path2)
    df2 = pd.read_csv(path2)

    df = pd.concat([df1, df2], ignore_index=True)
    logger.info("Combined HPD: %d probe points.", len(df))

    # ── Type coercion ─────────────────────────────────────────────────
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["heading"] = pd.to_numeric(df["heading"], errors="coerce")

    # Drop rows with invalid coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # ── Reconstruct traces ────────────────────────────────────────────
    gdf = _reconstruct_traces(df)

    # ── Optional bbox clip ────────────────────────────────────────────
    if clip_to_bbox:
        bbox_geom = box(*BBOX)
        before = len(gdf)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        gdf = gdf.reset_index(drop=True)
        logger.info("Bbox clip: %d → %d traces.", before, len(gdf))

    logger.info("HPD loaded: %d traces.", len(gdf))
    return gdf
