"""
VPD (Vehicle Per Day) data loader.

Reads the ~4 GB VPD CSV in chunks, filters to fused-only high-quality
drives, parses WKT geometries, and returns a GeoDataFrame.
"""

from __future__ import annotations

import ast
import logging
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString

from src.config import (
    BBOX,
    CRS,
    VPD_CHUNK_SIZE,
    VPD_CSV,
    VPD_KEEP_COLUMNS,
)

logger = logging.getLogger(__name__)


def _parse_geometry(wkt_str: str) -> Optional[LineString]:
    """Safely parse a WKT string into a Shapely geometry."""
    try:
        geom = wkt.loads(wkt_str)
        if geom.is_empty:
            return None
        return geom
    except Exception:
        return None


def _parse_altitudes(alt_str: str) -> list[float]:
    """Parse the altitudes column (stored as a string representation of a list)."""
    if pd.isna(alt_str) or alt_str == "":
        return []
    try:
        return [float(x) for x in ast.literal_eval(alt_str)]
    except (ValueError, SyntaxError):
        return []


def load_vpd(
    filepath: Optional[str] = None,
    chunk_size: int = VPD_CHUNK_SIZE,
    clip_to_bbox: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load and filter the VPD dataset.

    Steps:
        1. Read the CSV in chunks (to handle the ~4 GB file).
        2. Keep only rows where ``fused == 'Yes'``.
        3. Subset to the columns listed in ``VPD_KEEP_COLUMNS``.
        4. Parse ``path`` (WKT) into Shapely geometries.
        5. Parse ``altitudes`` into Python lists.
        6. Optionally clip to the study-area bounding box.
        7. Return a GeoDataFrame in EPSG:4326.

    Parameters
    ----------
    filepath : str, optional
        Override for the VPD CSV path (defaults to ``config.VPD_CSV``).
    chunk_size : int
        Number of rows per chunk when reading the CSV.
    clip_to_bbox : bool
        If True, drop rows whose geometry falls entirely outside the
        study-area bounding box.

    Returns
    -------
    gpd.GeoDataFrame
        Filtered, geometry-parsed VPD data.
    """
    src = filepath or str(VPD_CSV)
    logger.info("Loading VPD from %s (chunk_size=%d)…", src, chunk_size)

    chunks: list[pd.DataFrame] = []
    total_read = 0
    total_kept = 0

    for chunk in pd.read_csv(src, chunksize=chunk_size, low_memory=False):
        total_read += len(chunk)

        # ── Filter: fused only ────────────────────────────────────────
        fused_mask = chunk["fused"].astype(str).str.strip().str.lower() == "yes"
        filtered = chunk.loc[fused_mask].copy()

        if filtered.empty:
            continue

        # ── Subset columns (keep only those present) ──────────────────
        cols_present = [c for c in VPD_KEEP_COLUMNS if c in filtered.columns]
        filtered = filtered[cols_present]

        total_kept += len(filtered)
        chunks.append(filtered)

    if not chunks:
        logger.warning("No fused VPD rows found — returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(columns=VPD_KEEP_COLUMNS, geometry="geometry", crs=CRS)

    logger.info("Read %d rows total, kept %d fused rows.", total_read, total_kept)
    df = pd.concat(chunks, ignore_index=True)

    # ── Parse geometries ──────────────────────────────────────────────
    logger.info("Parsing WKT geometries…")
    df["geometry"] = df["path"].apply(_parse_geometry)
    df = df.dropna(subset=["geometry"])

    # ── Parse altitudes ───────────────────────────────────────────────
    if "altitudes" in df.columns:
        df["altitudes"] = df["altitudes"].apply(_parse_altitudes)

    # ── Build GeoDataFrame ────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)

    # ── Optional bbox clip ────────────────────────────────────────────
    if clip_to_bbox:
        from shapely.geometry import box

        bbox_geom = box(*BBOX)
        before = len(gdf)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        logger.info("Bbox clip: %d → %d rows.", before, len(gdf))

    # ── Clean up ──────────────────────────────────────────────────────
    gdf = gdf.drop(columns=["path"], errors="ignore")
    gdf = gdf.reset_index(drop=True)

    logger.info("VPD loaded: %d rows, %d columns.", len(gdf), len(gdf.columns))
    return gdf
