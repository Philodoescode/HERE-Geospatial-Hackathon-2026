from __future__ import annotations

import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from centerline.algorithms.kharita import KharitaAlgorithm, KharitaConfig


def _trace_row(
    trace_id: str, source: str, geom: LineString, day: int, hour: int
) -> dict:
    return {
        "trace_id": trace_id,
        "source": source,
        "geometry": geom,
        "day": day,
        "hour": hour,
        "construction_percent": 0.0,
        "altitudes": [],
        "crosswalk_types": [],
        "traffic_signal_count": 0.0,
        "path_quality_score": 0.8 if source == "VPD" else None,
        "sensor_quality_score": 0.8 if source == "VPD" else None,
        "point_times": [],
        "hpd_median_speed": 35.0 if source == "HPD" else None,
    }


def test_kharita_outputs_dynamic_and_postprocess_columns() -> None:
    traces = pd.DataFrame(
        [
            _trace_row(
                "v1",
                "VPD",
                LineString([(21.10, 42.60), (21.11, 42.60), (21.12, 42.60)]),
                1,
                9,
            ),
            _trace_row(
                "h1",
                "HPD",
                LineString([(21.105, 42.599), (21.115, 42.601), (21.125, 42.602)]),
                2,
                10,
            ),
        ]
    )

    crs = CRS.from_epsg(32634)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    cfg = KharitaConfig(
        min_edge_support=1.0,
        max_transition_distance_m=120.0,
        enable_dynamic_weighting=True,
    )
    out = KharitaAlgorithm(config=cfg).generate(
        traces=traces,
        projected_crs=crs,
        to_proj=to_proj,
        to_wgs=to_wgs,
    )

    assert "edges" in out and "centerlines" in out
    if not out["edges"].empty:
        for c in [
            "effective_support",
            "dyn_w_probe",
            "dyn_w_vpd",
            "road_likeness_score",
        ]:
            assert c in out["edges"].columns
    if not out["centerlines"].empty:
        for c in [
            "effective_support",
            "dyn_w_probe",
            "dyn_w_vpd",
            "road_likeness_score",
        ]:
            assert c in out["centerlines"].columns
