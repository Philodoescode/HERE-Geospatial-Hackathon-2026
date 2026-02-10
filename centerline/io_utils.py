from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
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
    return None


def _safe_parse_wkt_linestring(text: object) -> Optional[LineString]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    try:
        geom_obj = wkt.loads(str(text))
    except Exception:
        return None
    return _to_linestring(geom_obj)


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

    df["geometry"] = df["path"].map(_safe_parse_wkt_linestring)
    df = df[df["geometry"].notnull()].copy()

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
    ]
    return df[columns].reset_index(drop=True)


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
    df["geometry"] = df["geom"].map(_safe_parse_wkt_linestring)
    df = df[df["geometry"].notnull()].copy()
    return df
