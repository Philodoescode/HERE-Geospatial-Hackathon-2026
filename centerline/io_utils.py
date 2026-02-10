from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import linemerge


def _parse_list(value: object) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, list) else []
    except Exception:
        pass
    try:
        loaded = ast.literal_eval(text)
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def _to_linestring(geom_obj) -> Optional[LineString]:
    if geom_obj is None or geom_obj.is_empty:
        return None
    if isinstance(geom_obj, LineString):
        return geom_obj
    if isinstance(geom_obj, MultiLineString):
        merged = linemerge(geom_obj)
        if isinstance(merged, LineString):
            return merged
        if isinstance(merged, MultiLineString):
            lines = list(merged.geoms)
            if not lines:
                return None
            return max(lines, key=lambda g: g.length)
    if hasattr(geom_obj, "geoms"):
        lines = []
        for g in geom_obj.geoms:
            if isinstance(g, LineString):
                lines.append(g)
            elif isinstance(g, MultiLineString):
                lines.extend(list(g.geoms))
        if lines:
            merged = linemerge(lines)
            if isinstance(merged, LineString):
                return merged
            if isinstance(merged, MultiLineString):
                return max(list(merged.geoms), key=lambda gg: gg.length)
    return None


def _to_linestring_list(geom_obj) -> List[LineString]:
    if geom_obj is None or geom_obj.is_empty:
        return []
    if isinstance(geom_obj, LineString):
        return [geom_obj]
    if isinstance(geom_obj, MultiLineString):
        return [g for g in geom_obj.geoms if isinstance(g, LineString) and not g.is_empty]
    if hasattr(geom_obj, "geoms"):
        lines: List[LineString] = []
        for g in geom_obj.geoms:
            lines.extend(_to_linestring_list(g))
        return lines
    return []


def _safe_parse_wkt_linestring(text: object) -> Optional[LineString]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    try:
        geom_obj = wkt.loads(str(text))
    except Exception:
        return None
    return _to_linestring(geom_obj)


def _safe_parse_wkt_linestrings(text: object) -> List[LineString]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    try:
        geom_obj = wkt.loads(str(text))
    except Exception:
        return []
    return _to_linestring_list(geom_obj)


def infer_local_projected_crs(geometries: Sequence[LineString]) -> CRS:
    """Infer a metric CRS (UTM) from geometry centroid in WGS84."""
    valid = [g for g in geometries if g is not None and not g.is_empty]
    if not valid:
        return CRS.from_epsg(3857)
    lon = float(np.mean([g.centroid.x for g in valid]))
    lat = float(np.mean([g.centroid.y for g in valid]))
    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def load_vpd_traces(csv_path: str | Path, fused_only: bool = True, max_rows: int | None = None) -> pd.DataFrame:
    """
    Load VPD traces with WKT LINESTRING geometry and key attributes.
    """
    usecols = [
        "driveid",
        "fused",
        "path",
        "altitudes",
        "constructionpercent",
        "crosswalktypes",
        "trafficsignalcount",
        "day",
        "startoffset",
        "endoffset",
        "pathqualityscore",
        "sensorqualityscore",
    ]
    df = pd.read_csv(csv_path, usecols=usecols, nrows=max_rows)
    if fused_only and "fused" in df.columns:
        df = df[df["fused"].astype(str).str.lower().isin({"yes", "true", "1"})].copy()

    df["geometry_parts"] = df["path"].map(_safe_parse_wkt_linestrings)
    df = df[df["geometry_parts"].map(len) > 0].copy()

    df["altitudes"] = df["altitudes"].map(_parse_list)
    df["crosswalk_types"] = df["crosswalktypes"].map(_parse_list)
    df["construction_percent"] = pd.to_numeric(df["constructionpercent"], errors="coerce").fillna(0.0)
    df["traffic_signal_count"] = pd.to_numeric(df["trafficsignalcount"], errors="coerce").fillna(0.0)

    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    if "startoffset" in df.columns:
        start_offset = pd.to_numeric(df["startoffset"], errors="coerce")
        df["hour"] = ((start_offset % 86400) // 3600).astype("Int64")
    else:
        df["hour"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df["trace_id"] = df["driveid"].astype(str)
    df["source"] = "VPD"
    df["path_quality_score"] = pd.to_numeric(df["pathqualityscore"], errors="coerce")
    df["sensor_quality_score"] = pd.to_numeric(df["sensorqualityscore"], errors="coerce")

    rows = []
    for row in df.itertuples(index=False):
        parts = list(getattr(row, "geometry_parts", []))
        if not parts:
            continue
        for idx, geom in enumerate(parts):
            if geom is None or geom.is_empty or geom.length <= 0:
                continue
            rows.append(
                {
                    "trace_id": f"{row.trace_id}_p{idx}" if len(parts) > 1 else row.trace_id,
                    "source": row.source,
                    "geometry": geom,
                    "day": row.day,
                    "hour": row.hour,
                    "construction_percent": row.construction_percent,
                    "altitudes": row.altitudes,
                    "crosswalk_types": row.crosswalk_types,
                    "traffic_signal_count": row.traffic_signal_count,
                    "path_quality_score": row.path_quality_score,
                    "sensor_quality_score": row.sensor_quality_score,
                    # Optional per-point timestamps (not available for VPD in this release).
                    "point_times": [],
                }
            )
    columns = [
        "trace_id",
        "source",
        "geometry",
        "day",
        "hour",
        "construction_percent",
        "altitudes",
        "crosswalk_types",
        "traffic_signal_count",
        "path_quality_score",
        "sensor_quality_score",
        "point_times",
    ]
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def load_hpd_traces(csv_paths: Iterable[str | Path], max_rows_per_file: int | None = None) -> pd.DataFrame:
    """
    Convert HPD points to per-trace LINESTRING geometries.
    """
    frames: List[pd.DataFrame] = []
    usecols = ["heading", "latitude", "longitude", "traceid", "speed", "day", "time"]

    for p in csv_paths:
        sub = pd.read_csv(p, usecols=usecols, nrows=max_rows_per_file)
        sub["heading"] = pd.to_numeric(sub["heading"], errors="coerce")
        sub["latitude"] = pd.to_numeric(sub["latitude"], errors="coerce")
        sub["longitude"] = pd.to_numeric(sub["longitude"], errors="coerce")
        sub["speed"] = pd.to_numeric(sub["speed"], errors="coerce")
        sub["day"] = pd.to_numeric(sub["day"], errors="coerce")
        sub = sub.dropna(subset=["latitude", "longitude", "traceid", "time"])
        frames.append(sub)

    if not frames:
        return pd.DataFrame(
            columns=[
                "trace_id",
                "source",
                "geometry",
                "day",
                "hour",
                "construction_percent",
                "altitudes",
                "crosswalk_types",
                "traffic_signal_count",
                "path_quality_score",
                "sensor_quality_score",
                "point_times",
            ]
        )

    all_points = pd.concat(frames, ignore_index=True)
    all_points["time"] = pd.to_datetime(all_points["time"], format="%H:%M:%S", errors="coerce")
    all_points = all_points.dropna(subset=["time"])

    rows = []
    grouped = all_points.sort_values(["traceid", "day", "time"]).groupby(["traceid", "day"], sort=False)
    for (traceid, day), grp in grouped:
        if len(grp) < 2:
            continue
        coords = list(zip(grp["longitude"].tolist(), grp["latitude"].tolist()))
        try:
            geom = LineString(coords)
        except Exception:
            continue
        if geom.is_empty or geom.length == 0:
            continue

        hours = grp["time"].dt.hour
        hour = int(hours.mode().iloc[0]) if not hours.empty else None
        rows.append(
            {
                "trace_id": f"{traceid}_{int(day) if pd.notnull(day) else 'NA'}",
                "source": "HPD",
                "geometry": geom,
                "day": day,
                "hour": hour,
                "construction_percent": 0.0,
                "altitudes": [],
                "crosswalk_types": [],
                "traffic_signal_count": 0.0,
                "path_quality_score": np.nan,
                "sensor_quality_score": np.nan,
                "point_times": [
                    int(t.hour) * 3600 + int(t.minute) * 60 + int(t.second)
                    for t in grp["time"]
                ],
                "hpd_median_speed": float(np.nanmedian(grp["speed"])) if grp["speed"].notnull().any() else np.nan,
                "hpd_median_heading": float(np.nanmedian(grp["heading"])) if grp["heading"].notnull().any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def load_navstreet_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load HERE navstreet CSV as WGS84 lines.
    """
    df = pd.read_csv(csv_path)
    if "geom" not in df.columns:
        raise ValueError("Expected 'geom' column in navstreet CSV.")
    df["geometry_parts"] = df["geom"].map(_safe_parse_wkt_linestrings)
    df = df[df["geometry_parts"].map(len) > 0].copy()

    rows = []
    for row in df.itertuples(index=False):
        parts = list(getattr(row, "geometry_parts", []))
        for idx, geom in enumerate(parts):
            payload = row._asdict()
            payload.pop("geometry_parts", None)
            payload["geometry"] = geom
            payload["geom_part_idx"] = int(idx)
            rows.append(payload)

    return pd.DataFrame(rows)


def load_bbox_from_txt(bbox_path: str | Path) -> Tuple[float, float, float, float]:
    """
    Parse a bbox text file such as:
    Kosovo:
    [21.088588, 42.571255, 21.188588, 42.671255]
    """
    text = Path(bbox_path).read_text(encoding="utf-8")
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    if len(nums) < 4:
        raise ValueError(f"Could not parse bbox from {bbox_path}")
    min_lon, min_lat, max_lon, max_lat = nums[-4], nums[-3], nums[-2], nums[-1]
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(f"Invalid bbox in {bbox_path}: {(min_lon, min_lat, max_lon, max_lat)}")
    return (min_lon, min_lat, max_lon, max_lat)


def clip_line_geometries_to_bbox(
        df: pd.DataFrame,
        bbox_wgs84: Tuple[float, float, float, float],
        geometry_col: str = "geometry",
) -> pd.DataFrame:
    """
    Clip line geometries to a WGS84 bbox and keep non-empty line parts.
    """
    if df.empty or geometry_col not in df.columns:
        return df.copy()
    clip_poly = box(*bbox_wgs84)
    out = df.copy()
    out[geometry_col] = out[geometry_col].map(
        lambda g: _to_linestring(g.intersection(clip_poly)) if g is not None else None
    )
    out = out[out[geometry_col].notnull()].copy()
    out = out[~out[geometry_col].map(lambda g: g.is_empty)].copy()
    return out.reset_index(drop=True)
