"""
VPD (Vehicle Per Day) data loader.

Reads the ~4 GB VPD CSV in chunks, filters to fused-only high-quality
drives, parses WKT geometries, and returns a GeoDataFrame.

Memory-optimised: supports an explicit ``bbox`` parameter so that rows
outside a tight overlap region are discarded **during** chunk reading
(before geometry parsing), dramatically reducing peak RAM usage.
"""

from __future__ import annotations

import ast
import gc
import logging
import re
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, box

from src.config import (
    BBOX,
    CRS,
    VPD_CHUNK_SIZE,
    VPD_CSV,
    VPD_KEEP_COLUMNS,
)

logger = logging.getLogger(__name__)

# Regex to quickly extract the first coordinate pair from a WKT LINESTRING
# e.g.  "LINESTRING (21.1 42.6, 21.2 42.7)"  →  ("21.1", "42.6")
_FIRST_COORD_RE = re.compile(r"LINESTRING\s*\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)")


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


def _fast_bbox_overlap(wkt_str: str, minx: float, miny: float,
                       maxx: float, maxy: float) -> bool:
    """
    Quick-reject rows whose WKT path cannot intersect the target bbox.

    Strategy: extract **all** coordinate pairs from the WKT string with a
    lightweight regex and check whether *any* vertex falls inside the bbox
    (with a generous 0.01° buffer ≈ 1 km).  This avoids constructing a
    full Shapely object for the vast majority of rows that are clearly
    outside the overlap region.
    """
    if pd.isna(wkt_str):
        return False
    try:
        # Extract all coordinate pairs  (lon lat, lon lat, …)
        # WKT format: LINESTRING (x1 y1, x2 y2, ...)
        paren_start = wkt_str.index("(")
        inner = wkt_str[paren_start + 1 : wkt_str.rindex(")")]
        for pair in inner.split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                if minx <= x <= maxx and miny <= y <= maxy:
                    return True
        return False
    except Exception:
        # If parsing fails, keep the row (safe fallback)
        return True


def load_vpd(
    filepath: Optional[str] = None,
    chunk_size: int = VPD_CHUNK_SIZE,
    clip_to_bbox: bool = True,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    fused_only: bool = True,
    drop_altitudes: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load and filter the VPD dataset — memory-optimised.

    Steps:
        1. Read the CSV in chunks (to handle the ~4 GB file).
        2. Keep only rows where ``fused == 'Yes'`` (if *fused_only*).
        3. **Fast-reject** rows whose WKT path is outside *bbox* before
           parsing geometry (saves >70 % RAM on a tight overlap bbox).
        4. Subset to the columns listed in ``VPD_KEEP_COLUMNS``.
        5. Parse ``path`` (WKT) into Shapely geometries.
        6. Optionally clip to the bounding box with true spatial test.
        7. Return a GeoDataFrame in EPSG:4326.

    Parameters
    ----------
    filepath : str, optional
        Override for the VPD CSV path (defaults to ``config.VPD_CSV``).
    chunk_size : int
        Number of rows per chunk when reading the CSV.
    clip_to_bbox : bool
        If True, drop rows whose geometry falls entirely outside the bbox.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Explicit bounding box for spatial filtering.  If None, falls back
        to the study-area ``BBOX`` from config.  **Pass the tight overlap
        bbox here to avoid loading data you don't need.**
    fused_only : bool
        If True (default), keep only rows where fused == 'Yes'.
    drop_altitudes : bool
        If True, skip parsing the ``altitudes`` column to save RAM.

    Returns
    -------
    gpd.GeoDataFrame
        Filtered, geometry-parsed VPD data.
    """
    src = filepath or str(VPD_CSV)
    target_bbox = bbox or BBOX
    logger.info("Loading VPD from %s (chunk_size=%d)…", src, chunk_size)
    logger.info("Target bbox for fast pre-filter: %s", target_bbox)

    # Add a small buffer to the pre-filter to avoid rejecting edge traces
    BUF = 0.005  # ~0.5 km at this latitude
    pf_minx = target_bbox[0] - BUF
    pf_miny = target_bbox[1] - BUF
    pf_maxx = target_bbox[2] + BUF
    pf_maxy = target_bbox[3] + BUF

    chunks: list[pd.DataFrame] = []
    total_read = 0
    total_kept = 0
    total_prefiltered = 0

    for chunk in pd.read_csv(src, chunksize=chunk_size, low_memory=False):
        total_read += len(chunk)

        # ── Filter: fused only ────────────────────────────────────────
        if fused_only:
            fused_mask = chunk["fused"].astype(str).str.strip().str.lower() == "yes"
            filtered = chunk.loc[fused_mask]
        else:
            filtered = chunk

        if filtered.empty:
            continue

        # ── Fast bbox pre-filter on raw WKT (before geometry parsing) ─
        if clip_to_bbox and "path" in filtered.columns:
            bbox_mask = filtered["path"].apply(
                lambda s: _fast_bbox_overlap(s, pf_minx, pf_miny, pf_maxx, pf_maxy)
            )
            n_before = len(filtered)
            filtered = filtered.loc[bbox_mask]
            total_prefiltered += n_before - len(filtered)

        if filtered.empty:
            continue

        # ── Subset columns (keep only those present) ──────────────────
        cols_present = [c for c in VPD_KEEP_COLUMNS if c in filtered.columns]
        filtered = filtered[cols_present].copy()

        total_kept += len(filtered)
        chunks.append(filtered)

        # Periodic GC to keep peak memory in check
        if len(chunks) % 50 == 0:
            gc.collect()

    if not chunks:
        logger.warning("No fused VPD rows found — returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(columns=VPD_KEEP_COLUMNS, geometry="geometry", crs=CRS)

    logger.info(
        "Read %d rows total, kept %d fused rows (%d pre-filtered by bbox).",
        total_read, total_kept, total_prefiltered,
    )
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # ── Parse geometries ──────────────────────────────────────────────
    logger.info("Parsing WKT geometries for %d rows…", len(df))
    df["geometry"] = df["path"].apply(_parse_geometry)
    df = df.dropna(subset=["geometry"])

    # Drop path column immediately to free memory
    df = df.drop(columns=["path"], errors="ignore")
    gc.collect()

    # ── Parse altitudes (optional) ────────────────────────────────────
    if not drop_altitudes and "altitudes" in df.columns:
        df["altitudes"] = df["altitudes"].apply(_parse_altitudes)
    elif drop_altitudes and "altitudes" in df.columns:
        df = df.drop(columns=["altitudes"], errors="ignore")

    # ── Build GeoDataFrame ────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)
    del df
    gc.collect()

    # ── Precise bbox clip (true spatial test) ─────────────────────────
    if clip_to_bbox:
        bbox_geom = box(*target_bbox)
        before = len(gdf)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        logger.info("Bbox clip: %d → %d rows.", before, len(gdf))

    gdf = gdf.reset_index(drop=True)
    gc.collect()

    logger.info("VPD loaded: %d rows, %d columns.", len(gdf), len(gdf.columns))
    return gdf
