"""
Module 1: Data Loading & Preprocessing.

Loads VPD and HPD traces, clips to bounding box, projects to UTM,
and returns cleaned LineStrings ready for rasterization.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import linemerge, transform


@dataclass
class LoadConfig:
    """Configuration for data loading."""

    bbox_wgs84: Tuple[float, float, float, float] = (
        21.088588,
        42.571255,
        21.188588,
        42.671255,
    )
    fused_only: bool = True
    min_trace_length_m: float = 10.0
    simplify_tolerance_m: float = 1.5
    target_crs_epsg: int = 32634  # UTM zone 34N for Kosovo


def _parse_wkt(text: object) -> Optional[LineString]:
    """Parse a WKT string into a LineString, handling MultiLineString."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    try:
        geom = wkt.loads(str(text))
    except Exception:
        return None
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        if isinstance(merged, MultiLineString):
            parts = list(merged.geoms)
            return max(parts, key=lambda g: g.length) if parts else None
    return None


def _parse_wkt_multi(text: object) -> List[LineString]:
    """Parse a WKT string, returning all LineString parts."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    try:
        geom = wkt.loads(str(text))
    except Exception:
        return []
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    return []


def _read_table(
    path: Path, usecols: Optional[list] = None, max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Read CSV or Parquet into DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if usecols:
            try:
                df = pd.read_parquet(path, columns=usecols)
            except Exception:
                df = pd.read_parquet(path)
                available = [c for c in usecols if c in df.columns]
                df = df[available]
        else:
            df = pd.read_parquet(path)
        if max_rows is not None:
            df = df.head(max_rows)
        return df
    return pd.read_csv(path, usecols=usecols, nrows=max_rows)


def _clip_to_bbox(geom: LineString, clip_poly) -> Optional[LineString]:
    """Clip a LineString to a polygon, returning the largest piece."""
    try:
        clipped = geom.intersection(clip_poly)
    except Exception:
        return None
    if clipped is None or clipped.is_empty:
        return None
    if isinstance(clipped, LineString):
        return clipped
    if isinstance(clipped, MultiLineString):
        parts = [
            g for g in clipped.geoms if isinstance(g, LineString) and not g.is_empty
        ]
        return max(parts, key=lambda g: g.length) if parts else None
    return None


@dataclass
class TraceData:
    """Container for loaded trace geometries."""

    lines_utm: List[LineString]
    sources: List[str]  # "VPD" or "HPD" per line
    utm_crs: CRS
    utm_bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy in UTM
    wgs84_bbox: Tuple[float, float, float, float]
    n_vpd: int = 0
    n_hpd: int = 0


def load_vpd(
    path: Path,
    bbox: Tuple[float, float, float, float],
    fused_only: bool = True,
    max_rows: Optional[int] = None,
) -> List[LineString]:
    """Load VPD traces, filter fused, clip to bbox, return WGS84 LineStrings."""
    usecols = ["driveid", "fused", "path"]
    df = _read_table(path, usecols=usecols, max_rows=max_rows)

    if fused_only and "fused" in df.columns:
        df = df[df["fused"].astype(str).str.lower().isin({"yes", "true", "1"})].copy()

    clip_poly = box(*bbox)
    lines = []
    for _, row in df.iterrows():
        parts = _parse_wkt_multi(row.get("path"))
        for geom in parts:
            clipped = _clip_to_bbox(geom, clip_poly)
            if clipped is not None and clipped.length > 0:
                lines.append(clipped)

    return lines


def load_hpd(
    paths: List[Path],
    bbox: Tuple[float, float, float, float],
    max_rows_per_file: Optional[int] = None,
) -> List[LineString]:
    """Load HPD probe points, group by trace, form LineStrings, clip to bbox."""
    usecols = ["heading", "latitude", "longitude", "traceid", "speed", "day", "time"]
    frames = []
    for p in paths:
        sub = _read_table(p, usecols=usecols, max_rows=max_rows_per_file)
        sub["latitude"] = pd.to_numeric(sub["latitude"], errors="coerce")
        sub["longitude"] = pd.to_numeric(sub["longitude"], errors="coerce")
        sub["day"] = pd.to_numeric(sub["day"], errors="coerce")
        sub = sub.dropna(subset=["latitude", "longitude", "traceid", "time"])
        frames.append(sub)

    if not frames:
        return []

    all_pts = pd.concat(frames, ignore_index=True)
    all_pts["time"] = pd.to_datetime(
        all_pts["time"], format="%H:%M:%S", errors="coerce"
    )
    all_pts = all_pts.dropna(subset=["time"])

    clip_poly = box(*bbox)
    lines = []
    grouped = all_pts.sort_values(["traceid", "day", "time"]).groupby(
        ["traceid", "day"], sort=False
    )
    for _, grp in grouped:
        if len(grp) < 2:
            continue
        coords = list(zip(grp["longitude"].tolist(), grp["latitude"].tolist()))
        try:
            geom = LineString(coords)
        except Exception:
            continue
        if geom.is_empty or geom.length == 0:
            continue
        clipped = _clip_to_bbox(geom, clip_poly)
        if clipped is not None and clipped.length > 0:
            lines.append(clipped)

    return lines


def load_all_traces(cfg: LoadConfig, data_dir: Path) -> TraceData:
    """
    Load all VPD and HPD traces, project to UTM, clean, and return.

    Expects:
        data_dir/Kosovo_VPD/Kosovo_VPD.parquet (or .csv)
        data_dir/Kosovo_HPD/XKO_HPD_week_1.parquet (or .csv)
        data_dir/Kosovo_HPD/XKO_HPD_week_2.parquet (or .csv)
    """
    # Find VPD file
    vpd_dir = data_dir / "Kosovo_VPD"
    vpd_file = None
    for ext in [".parquet", ".csv"]:
        candidate = vpd_dir / f"Kosovo_VPD{ext}"
        if candidate.exists():
            vpd_file = candidate
            break
    if vpd_file is None:
        # Try any parquet/csv in the dir
        for f in vpd_dir.glob("*"):
            if f.suffix.lower() in {".parquet", ".csv"}:
                vpd_file = f
                break

    # Find HPD files
    hpd_dir = data_dir / "Kosovo_HPD"
    hpd_files = []
    for ext in [".parquet", ".csv"]:
        for f in sorted(hpd_dir.glob(f"*{ext}")):
            hpd_files.append(f)
        if hpd_files:
            break

    print(f"[data_loader] VPD file: {vpd_file}")
    print(f"[data_loader] HPD files: {hpd_files}")

    # Load traces in WGS84
    vpd_lines = load_vpd(vpd_file, cfg.bbox_wgs84, cfg.fused_only) if vpd_file else []
    hpd_lines = load_hpd(hpd_files, cfg.bbox_wgs84) if hpd_files else []

    print(
        f"[data_loader] Loaded {len(vpd_lines)} VPD traces, {len(hpd_lines)} HPD traces"
    )

    # Set up projection
    utm_crs = CRS.from_epsg(cfg.target_crs_epsg)
    to_utm = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)

    # Project and clean
    lines_utm = []
    sources = []

    for geom in vpd_lines:
        projected = transform(to_utm.transform, geom)
        if projected.is_empty or projected.length < cfg.min_trace_length_m:
            continue
        if cfg.simplify_tolerance_m > 0:
            simplified = projected.simplify(
                cfg.simplify_tolerance_m, preserve_topology=False
            )
            if (
                isinstance(simplified, LineString)
                and not simplified.is_empty
                and len(simplified.coords) >= 2
            ):
                projected = simplified
        lines_utm.append(projected)
        sources.append("VPD")

    for geom in hpd_lines:
        projected = transform(to_utm.transform, geom)
        if projected.is_empty or projected.length < cfg.min_trace_length_m:
            continue
        if cfg.simplify_tolerance_m > 0:
            simplified = projected.simplify(
                cfg.simplify_tolerance_m * 2, preserve_topology=False
            )
            if (
                isinstance(simplified, LineString)
                and not simplified.is_empty
                and len(simplified.coords) >= 2
            ):
                projected = simplified
        lines_utm.append(projected)
        sources.append("HPD")

    # Compute UTM bounds
    if lines_utm:
        all_coords = np.vstack([np.array(l.coords) for l in lines_utm])
        utm_bounds = (
            float(all_coords[:, 0].min()),
            float(all_coords[:, 1].min()),
            float(all_coords[:, 0].max()),
            float(all_coords[:, 1].max()),
        )
    else:
        # Project bbox corners
        x1, y1 = to_utm.transform(cfg.bbox_wgs84[0], cfg.bbox_wgs84[1])
        x2, y2 = to_utm.transform(cfg.bbox_wgs84[2], cfg.bbox_wgs84[3])
        utm_bounds = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    n_vpd = sum(1 for s in sources if s == "VPD")
    n_hpd = sum(1 for s in sources if s == "HPD")
    print(f"[data_loader] After projection & cleaning: {n_vpd} VPD, {n_hpd} HPD traces")
    print(f"[data_loader] UTM bounds: {utm_bounds}")

    return TraceData(
        lines_utm=lines_utm,
        sources=sources,
        utm_crs=utm_crs,
        utm_bounds=utm_bounds,
        wgs84_bbox=cfg.bbox_wgs84,
        n_vpd=n_vpd,
        n_hpd=n_hpd,
    )
