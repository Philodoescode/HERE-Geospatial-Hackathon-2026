"""
Nav Streets reference data loader.

Loads the HERE Nav Streets road network (GeoPackage or CSV) as a
GeoDataFrame that serves as ground truth for validation and labelling.
"""

from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import box

from src.config import BBOX, CRS, NAV_STREETS_CSV, NAV_STREETS_GPKG

logger = logging.getLogger(__name__)


def load_nav_streets(
    filepath: Optional[str] = None,
    prefer_gpkg: bool = True,
    clip_to_bbox: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load the Nav Streets reference road network.

    Attempts to read the GeoPackage first (faster, already has geometries).
    Falls back to the CSV if the GPKG is unavailable or ``prefer_gpkg``
    is False, parsing the WKT ``geom`` column manually.

    Parameters
    ----------
    filepath : str, optional
        Explicit path override.  When provided, the extension determines
        the parsing strategy.
    prefer_gpkg : bool
        If True (default) and no explicit filepath is given, try the
        GeoPackage before the CSV.
    clip_to_bbox : bool
        If True, drop road links that fall entirely outside the bbox.

    Returns
    -------
    gpd.GeoDataFrame
        Nav Streets road links with geometry + attributes.
    """
    if filepath:
        src = filepath
    elif prefer_gpkg and NAV_STREETS_GPKG.exists():
        src = str(NAV_STREETS_GPKG)
    else:
        src = str(NAV_STREETS_CSV)

    logger.info("Loading Nav Streets from %s…", src)

    if src.endswith(".gpkg"):
        gdf = gpd.read_file(src)
    else:
        # CSV path — parse WKT geometry manually
        df = pd.read_csv(src)
        df["geometry"] = df["geom"].apply(_safe_parse_wkt)
        df = df.dropna(subset=["geometry"])
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)
        gdf = gdf.drop(columns=["geom"], errors="ignore")

    # ── Ensure CRS ────────────────────────────────────────────────────
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS)
    elif str(gdf.crs) != CRS:
        gdf = gdf.to_crs(CRS)

    # ── Optional bbox clip ────────────────────────────────────────────
    if clip_to_bbox:
        bbox_geom = box(*BBOX)
        before = len(gdf)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        gdf = gdf.reset_index(drop=True)
        logger.info("Bbox clip: %d → %d road links.", before, len(gdf))

    # ── Type coercion for key columns ─────────────────────────────────
    int_cols = ["ref_zlevel", "nref_zlevel", "func_class", "fr_spd_lim", "to_spd_lim"]
    for col in int_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    logger.info(
        "Nav Streets loaded: %d road links, %d columns.",
        len(gdf),
        len(gdf.columns),
    )
    return gdf


def _safe_parse_wkt(wkt_str: str):
    """Safely parse a WKT string, returning None on failure."""
    try:
        geom = wkt.loads(str(wkt_str))
        return None if geom.is_empty else geom
    except Exception:
        return None
