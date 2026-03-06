"""
Preprocessing and Cleaning Utilities (Phase 4).

Reusable functions for geometry cleaning, densification, simplification,
and attribute computation (e.g., heading).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point, box
from shapely.ops import snap

logger = logging.getLogger(__name__)


def clip_to_bbox(
    gdf: gpd.GeoDataFrame, bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to the specified bounding box.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input data.
    bbox : tuple
        (minx, miny, maxx, maxy).

    Returns
    -------
    gpd.GeoDataFrame
        Subset of gdf intersecting the bbox.
    """
    bbox_geom = box(*bbox)
    # intersection(bbox) is safer than just checking intersects() if we want strict clipping,
    # but usually for road data we just want to keep features *inside* the area.
    # The loaders used `intersects`. Here we'll stick to `intersects` to avoid
    # cutting linestrings in half unless strict clipping is requested.
    # For this utility, let's just filter.
    return gdf[gdf.geometry.intersects(bbox_geom)].copy()


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fix invalid geometries using the buffer(0) trick and remove empty/None ones.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input data.

    Returns
    -------
    gpd.GeoDataFrame
        Cleaned data with valid geometries.
    """
    initial_count = len(gdf)

    # 1. Drop None/NaN
    gdf = gdf.dropna(subset=["geometry"]).copy()

    # 2. Drop Empty
    gdf = gdf[~gdf.geometry.is_empty].copy()

    # 3. Fix Invalid (Self-intersecting, etc.)
    # The buffer(0) trick often fixes self-intersections in polygons,
    # but for LineStrings it might not do much or could convert them to Polygons if closed.
    # For LineStrings, 'make_valid' is preferred in newer Shapely/GEOS.

    mask_invalid = ~gdf.geometry.is_valid
    if mask_invalid.any():
        logger.info(f"Attempting to fix {mask_invalid.sum()} invalid geometries...")

        # Try buffer(0) first - standard trick
        gdf.loc[mask_invalid, "geometry"] = gdf.loc[mask_invalid, "geometry"].buffer(0)

        # Re-check
        still_invalid = ~gdf.geometry.is_valid
        if still_invalid.any():
            logger.warning(
                f"Dropping {still_invalid.sum()} geometries that could not be fixed."
            )
            gdf = gdf[~still_invalid].copy()

    # Ensure we haven't created empty geometries during fixing
    gdf = gdf[~gdf.geometry.is_empty].copy()

    dropped = initial_count - len(gdf)
    if dropped > 0:
        logger.info(f"validate_geometries: Dropped {dropped} rows total.")

    return gdf


def densify_line(geom: LineString, max_dist: float) -> LineString:
    """
    Add interpolated points to a LineString so no segment is longer than max_dist.

    Parameters
    ----------
    geom : LineString
        Input geometry.
    max_dist : float
        Maximum segment length (in CRS units, usually degrees if EPSG:4326).
        Note: If CRS is degrees, max_dist should be in degrees (e.g. 0.0001).

    Returns
    -------
    LineString
        Densified geometry.
    """
    if geom.is_empty:
        return geom

    # Simple densification by interpolation
    # If using degrees, 10m is approx 0.0001 deg.

    length = geom.length
    if length <= max_dist:
        return geom

    num_segments = int(math.ceil(length / max_dist))
    if num_segments == 0:
        return geom

    points = []
    for i in range(num_segments + 1):
        fraction = i / num_segments
        point = geom.interpolate(fraction, normalized=True)
        points.append(point)

    return LineString(points)


def densify_gdf(gdf: gpd.GeoDataFrame, max_dist: float) -> gpd.GeoDataFrame:
    """Apply densify_line to an entire GeoDataFrame."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda g: densify_line(g, max_dist))
    return gdf


def simplify_line(geom: LineString, tolerance: float) -> LineString:
    """
    Simplify geometry using Douglas-Peucker algorithm.

    Parameters
    ----------
    geom : LineString
        Input geometry.
    tolerance : float
        Max distance deviation (in CRS units).

    Returns
    -------
    LineString
        Simplified geometry.
    """
    return geom.simplify(tolerance, preserve_topology=True)


def simplify_gdf(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    """Apply simplify_line to an entire GeoDataFrame."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].simplify(tolerance, preserve_topology=True)
    return gdf


def compute_heading(geom: LineString) -> float:
    """
    Compute the overall heading (bearing) of a LineString.

    For a straight line, it's the angle.
    For a complex path, we take the angle between start and end points.
    Returns degrees [0, 360).
    """
    if geom.is_empty or len(geom.coords) < 2:
        return 0.0

    start = geom.coords[0]
    end = geom.coords[-1]

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    # Atan2 returns radians in (-pi, pi]
    # 0 is East (positive x), pi/2 is North (positive y)
    # We usually want 0 = North, 90 = East (Navigation bearing)
    # But let's stick to standard math angle for now or convert if needed.
    # Standard math: 0=East, 90=North.
    # Navigation: 0=North, 90=East.

    # Let's return standard geographic bearing (0=N, 90=E)
    # bearing = atan2(dx, dy) (swapped x/y for N reference)

    rads = math.atan2(dx, dy)
    degrees = math.degrees(rads)

    if degrees < 0:
        degrees += 360

    return degrees


def snap_to_grid(gdf: gpd.GeoDataFrame, precision: float) -> gpd.GeoDataFrame:
    """
    Snap geometries to a grid of given precision (rounding coordinates).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    precision : float
        Rounding precision (e.g. 0.00001).

    Returns
    -------
    gpd.GeoDataFrame
    """
    # Simply rounding coordinates is the easiest way
    # Or usage shapely.ops.snap if snapping to other geometries
    # Here we just round for data reduction/alignment

    def round_coords(geom):
        if geom.is_empty:
            return geom
        return wkt.loads(
            wkt.dumps(geom, rounding_precision=int(-math.log10(precision)))
        )

    # Using WKT round-trip is one way, but slow.
    # geopandas doesn't have a direct 'snap to grid' but we can use set_precision (in recent versions)

    try:
        return gdf.set_precision(precision)
    except AttributeError:
        # Fallback for older geopandas/shapely
        return gdf
