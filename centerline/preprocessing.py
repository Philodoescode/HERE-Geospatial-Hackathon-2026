from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.ops import transform

from .utils import angle_diff_deg, bearing_from_xy


@dataclass
class SourcePreprocessConfig:
    """Preprocessing knobs applied per source (VPD or HPD/Probe)."""

    resample_spacing_m: float
    simplify_tolerance_m: float
    max_gap_m: float
    max_time_gap_s: float | None
    min_segment_length_m: float
    spike_deflection_deg: float
    spike_edge_max_m: float
    split_turn_deflection_deg: float | None
    split_turn_min_progress_m: float


def default_source_preprocess_config(source: str) -> SourcePreprocessConfig:
    src = str(source).upper()
    if src == "VPD":
        return SourcePreprocessConfig(
            resample_spacing_m=4.0,
            simplify_tolerance_m=1.5,
            max_gap_m=60.0,
            max_time_gap_s=None,
            min_segment_length_m=25.0,
            spike_deflection_deg=120.0,
            spike_edge_max_m=20.0,
            split_turn_deflection_deg=95.0,
            split_turn_min_progress_m=35.0,
        )
    return SourcePreprocessConfig(
        resample_spacing_m=8.0,
        simplify_tolerance_m=3.0,
        max_gap_m=90.0,
        max_time_gap_s=300.0,
        min_segment_length_m=25.0,
        spike_deflection_deg=110.0,
        spike_edge_max_m=25.0,
        split_turn_deflection_deg=100.0,
        split_turn_min_progress_m=40.0,
    )


def _linestring_from_coords(coords: np.ndarray) -> LineString | None:
    if coords is None or len(coords) < 2:
        return None
    try:
        line = LineString([(float(x), float(y)) for x, y in coords])
    except Exception:
        return None
    if line.is_empty or line.length <= 0.0:
        return None
    return line


def _cumdist(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(coords) == 1:
        return np.array([0.0], dtype=np.float64)
    d = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def _prepare_scalar_series(values: Sequence[float] | None, n: int) -> np.ndarray:
    if values is None:
        return np.full(n, np.nan, dtype=np.float64)
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == n:
        return arr
    if len(arr) < 2 or n < 2:
        return np.full(n, np.nan, dtype=np.float64)
    src_x = np.linspace(0.0, 1.0, len(arr))
    dst_x = np.linspace(0.0, 1.0, n)
    return np.interp(dst_x, src_x, arr)


def _remove_duplicate_consecutive(
        coords: np.ndarray,
        altitudes: np.ndarray,
        point_times: np.ndarray,
        min_dist_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(coords) < 2:
        return coords, altitudes, point_times
    keep = [0]
    for i in range(1, len(coords)):
        if float(np.hypot(*(coords[i] - coords[keep[-1]]))) >= min_dist_m:
            keep.append(i)
    keep_idx = np.asarray(keep, dtype=np.int64)
    return coords[keep_idx], altitudes[keep_idx], point_times[keep_idx]


def _remove_spikes(
        coords: np.ndarray,
        altitudes: np.ndarray,
        point_times: np.ndarray,
        max_deflection_deg: float,
        max_edge_len_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(coords) < 3:
        return coords, altitudes, point_times

    keep = np.ones(len(coords), dtype=bool)
    for i in range(1, len(coords) - 1):
        if not keep[i - 1] or not keep[i]:
            continue
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        l1 = float(np.hypot(v1[0], v1[1]))
        l2 = float(np.hypot(v2[0], v2[1]))
        if l1 <= 1e-6 or l2 <= 1e-6:
            keep[i] = False
            continue
        h1 = bearing_from_xy(coords[i - 1, 0], coords[i - 1, 1], coords[i, 0], coords[i, 1])
        h2 = bearing_from_xy(coords[i, 0], coords[i, 1], coords[i + 1, 0], coords[i + 1, 1])
        deflection = angle_diff_deg(h1, h2)
        if deflection >= max_deflection_deg and min(l1, l2) <= max_edge_len_m:
            keep[i] = False

    return coords[keep], altitudes[keep], point_times[keep]


def _split_indices_by_gaps(
        coords: np.ndarray,
        point_times: np.ndarray,
        max_gap_m: float,
        max_time_gap_s: float | None,
) -> List[int]:
    split_idx = [0]
    if len(coords) < 2:
        return split_idx + [len(coords)]
    for i in range(1, len(coords)):
        gap = float(np.hypot(*(coords[i] - coords[i - 1])))
        should_split = gap > max_gap_m
        if (
                not should_split
                and max_time_gap_s is not None
                and np.isfinite(point_times[i - 1])
                and np.isfinite(point_times[i])
                and (point_times[i] - point_times[i - 1]) > max_time_gap_s
        ):
            should_split = True
        if should_split and i - split_idx[-1] >= 2:
            split_idx.append(i)
    if split_idx[-1] != len(coords):
        split_idx.append(len(coords))
    return split_idx


def _split_by_turns(
        coords: np.ndarray,
        altitudes: np.ndarray,
        point_times: np.ndarray,
        turn_deflection_deg: float,
        min_progress_m: float,
) -> List[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(coords) < 3:
        return [(coords, altitudes, point_times)]

    breaks = [0]
    progress_since_last_break = 0.0
    for i in range(1, len(coords) - 1):
        progress_since_last_break += float(np.hypot(*(coords[i] - coords[i - 1])))
        h1 = bearing_from_xy(coords[i - 1, 0], coords[i - 1, 1], coords[i, 0], coords[i, 1])
        h2 = bearing_from_xy(coords[i, 0], coords[i, 1], coords[i + 1, 0], coords[i + 1, 1])
        if angle_diff_deg(h1, h2) >= turn_deflection_deg and progress_since_last_break >= min_progress_m:
            breaks.append(i)
            progress_since_last_break = 0.0

    if breaks[-1] != len(coords) - 1:
        breaks.append(len(coords) - 1)

    out = []
    for i in range(1, len(breaks)):
        a = breaks[i - 1]
        b = breaks[i]
        if b - a < 1:
            continue
        out.append((coords[a: b + 1], altitudes[a: b + 1], point_times[a: b + 1]))
    return out


def _resample_segment(
        coords: np.ndarray,
        altitudes: np.ndarray,
        point_times: np.ndarray,
        spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(coords) < 2:
        return coords, altitudes, point_times

    cd = _cumdist(coords)
    total = float(cd[-1])
    if total <= 0.0:
        return coords, altitudes, point_times

    d = np.arange(0.0, total, max(spacing_m, 0.5), dtype=np.float64)
    if len(d) == 0 or d[-1] < total:
        d = np.concatenate([d, [total]])

    x = np.interp(d, cd, coords[:, 0])
    y = np.interp(d, cd, coords[:, 1])
    z = np.interp(d, cd, altitudes) if np.isfinite(altitudes).any() else np.full(len(d), np.nan, dtype=np.float64)
    t = np.interp(d, cd, point_times) if np.isfinite(point_times).any() else np.full(len(d), np.nan, dtype=np.float64)
    xy = np.column_stack([x, y]).astype(np.float64)
    return xy, z.astype(np.float64), t.astype(np.float64)


def _headings(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.zeros(0, dtype=np.float64)
    h = np.zeros(len(coords), dtype=np.float64)
    for i in range(len(coords)):
        i0 = max(0, i - 1)
        i1 = min(len(coords) - 1, i + 1)
        if i0 == i1:
            h[i] = 0.0
        else:
            h[i] = bearing_from_xy(
                float(coords[i0, 0]),
                float(coords[i0, 1]),
                float(coords[i1, 0]),
                float(coords[i1, 1]),
            )
    return h


def preprocess_trace_geometry(
        line_xy: LineString,
        altitudes: Sequence[float] | None,
        point_times: Sequence[float] | None,
        cfg: SourcePreprocessConfig,
) -> List[dict]:
    coords = np.asarray(line_xy.coords, dtype=np.float64)
    if len(coords) < 2:
        return []

    z = _prepare_scalar_series(altitudes, len(coords))
    t = _prepare_scalar_series(point_times, len(coords))

    coords, z, t = _remove_duplicate_consecutive(coords, z, t)
    coords, z, t = _remove_spikes(
        coords,
        z,
        t,
        max_deflection_deg=cfg.spike_deflection_deg,
        max_edge_len_m=cfg.spike_edge_max_m,
    )
    if len(coords) < 2:
        return []

    split_idx = _split_indices_by_gaps(
        coords,
        t,
        max_gap_m=cfg.max_gap_m,
        max_time_gap_s=cfg.max_time_gap_s,
    )

    segments: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(1, len(split_idx)):
        a = split_idx[i - 1]
        b = split_idx[i]
        if b - a < 2:
            continue
        segments.append((coords[a:b], z[a:b], t[a:b]))

    if cfg.split_turn_deflection_deg is not None:
        turn_split_segments: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for cxy, cz, ct in segments:
            turn_split_segments.extend(
                _split_by_turns(
                    cxy,
                    cz,
                    ct,
                    turn_deflection_deg=float(cfg.split_turn_deflection_deg),
                    min_progress_m=cfg.split_turn_min_progress_m,
                )
            )
        segments = turn_split_segments

    out = []
    for cxy, cz, ct in segments:
        seg_line = _linestring_from_coords(cxy)
        if seg_line is None or seg_line.length < cfg.min_segment_length_m:
            continue

        cxy, cz, ct = _resample_segment(
            cxy,
            cz,
            ct,
            spacing_m=cfg.resample_spacing_m,
        )
        if len(cxy) < 2:
            continue
        seg_line = _linestring_from_coords(cxy)
        if seg_line is None or seg_line.length < cfg.min_segment_length_m:
            continue

        if cfg.simplify_tolerance_m > 0:
            simp = seg_line.simplify(float(cfg.simplify_tolerance_m), preserve_topology=False)
            if isinstance(simp, LineString) and not simp.is_empty and simp.length > 0:
                simp_coords = np.asarray(simp.coords, dtype=np.float64)
                if len(simp_coords) >= 2:
                    cxy = simp_coords
                    cz = _prepare_scalar_series(cz, len(cxy))
                    ct = _prepare_scalar_series(ct, len(cxy))
                    seg_line = simp
                    cxy, cz, ct = _resample_segment(cxy, cz, ct, spacing_m=cfg.resample_spacing_m)
                    seg_line = _linestring_from_coords(cxy) or seg_line

        if seg_line.length < cfg.min_segment_length_m:
            continue

        out.append(
            {
                "geometry_xy": seg_line,
                "coords_xy": cxy,
                "altitudes": cz,
                "point_times": ct,
                "headings": _headings(cxy),
                "length_m": float(seg_line.length),
            }
        )
    return out


def preprocess_traces_dataframe(
        traces: pd.DataFrame,
        to_proj: Transformer,
        to_wgs: Transformer,
        source_cfgs: Dict[str, SourcePreprocessConfig] | None = None,
) -> pd.DataFrame:
    """Preprocess traces into cleaned/resampled projected segments."""
    if traces.empty:
        return pd.DataFrame()

    source_cfgs = source_cfgs or {}
    rows = []
    seg_id = 0

    for row in traces.itertuples(index=False):
        line_wgs = getattr(row, "geometry", None)
        if line_wgs is None or line_wgs.is_empty:
            continue
        line_xy = transform(to_proj.transform, line_wgs)
        if line_xy.is_empty or line_xy.length <= 0:
            continue

        source = str(getattr(row, "source", "VPD")).upper()
        cfg = source_cfgs.get(source, default_source_preprocess_config(source))

        segments = preprocess_trace_geometry(
            line_xy=line_xy,
            altitudes=getattr(row, "altitudes", []),
            point_times=getattr(row, "point_times", []),
            cfg=cfg,
        )

        for idx, seg in enumerate(segments):
            line_seg_xy = seg["geometry_xy"]
            line_seg_wgs = transform(to_wgs.transform, line_seg_xy)
            rows.append(
                {
                    "segment_id": seg_id,
                    "segment_part": idx,
                    "trace_id": str(getattr(row, "trace_id", "")),
                    "source": source,
                    "geometry_xy": line_seg_xy,
                    "geometry": line_seg_wgs,
                    "coords_xy": seg["coords_xy"],
                    "altitudes": seg["altitudes"],
                    "point_times": seg["point_times"],
                    "headings": seg["headings"],
                    "length_m": seg["length_m"],
                    "construction_percent": float(getattr(row, "construction_percent", 0.0) or 0.0),
                    "traffic_signal_count": float(getattr(row, "traffic_signal_count", 0.0) or 0.0),
                    "crosswalk_types": (
                        getattr(row, "crosswalk_types", [])
                        if isinstance(getattr(row, "crosswalk_types", []), list)
                        else []
                    ),
                    "day": int(getattr(row, "day")) if pd.notnull(getattr(row, "day", np.nan)) else None,
                    "hour": int(getattr(row, "hour")) if pd.notnull(getattr(row, "hour", np.nan)) else None,
                    "path_quality_score": float(getattr(row, "path_quality_score"))
                    if pd.notnull(getattr(row, "path_quality_score", np.nan))
                    else np.nan,
                    "sensor_quality_score": float(getattr(row, "sensor_quality_score"))
                    if pd.notnull(getattr(row, "sensor_quality_score", np.nan))
                    else np.nan,
                    "hpd_median_speed": float(getattr(row, "hpd_median_speed"))
                    if pd.notnull(getattr(row, "hpd_median_speed", np.nan))
                    else np.nan,
                }
            )
            seg_id += 1

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
