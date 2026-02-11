from __future__ import annotations

import math

import geopandas as gpd
from shapely.geometry import LineString

from centerline.evaluation import (
    _infer_graph,
    build_evaluation_context,
    evaluate_centerline_geodataframes,
)


def _wgs84_gdf(lines: list[LineString]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": lines}, geometry="geometry", crs="EPSG:4326")


def _t_junction_lines(include_north_arm: bool = True) -> list[LineString]:
    lines = [
        LineString([(21.0000, 42.0050), (21.0050, 42.0050)]),
        LineString([(21.0050, 42.0050), (21.0100, 42.0050)]),
    ]
    if include_north_arm:
        lines.append(LineString([(21.0050, 42.0050), (21.0050, 42.0100)]))
    return lines


def test_regression_fields_and_identity_topo_itopo() -> None:
    gt = _wgs84_gdf(_t_junction_lines(include_north_arm=True))
    gen = _wgs84_gdf(_t_junction_lines(include_north_arm=True))

    metrics = evaluate_centerline_geodataframes(
        generated_gdf=gen,
        ground_truth_gdf=gt,
        buffer_m=15.0,
        sample_step_m=10.0,
        compute_topology_metrics=True,
        compute_itopo=True,
        topology_radii_m=(8.0, 15.0),
        compute_hausdorff=False,
    )

    for k in (
        "length_precision",
        "length_recall",
        "length_f1",
        "generated_to_groundtruth_distance_m",
        "groundtruth_to_generated_distance_m",
        "topology",
    ):
        assert k in metrics

    assert math.isclose(metrics["length_precision"], 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["length_recall"], 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(metrics["length_f1"], 1.0, rel_tol=1e-6, abs_tol=1e-6)

    for radius_key in ("8.0", "15.0"):
        topo_f1 = metrics["topology"]["topo"]["by_radius_m"][radius_key]["f1"]
        itopo_f1 = metrics["topology"]["i_topo"]["by_radius_m"][radius_key]["f1"]
        assert topo_f1 > 0.99
        assert itopo_f1 > 0.99


def test_itopo_drop_for_missing_junction_arm() -> None:
    gt = _wgs84_gdf(_t_junction_lines(include_north_arm=True))
    gen = _wgs84_gdf(_t_junction_lines(include_north_arm=False))

    metrics = evaluate_centerline_geodataframes(
        generated_gdf=gen,
        ground_truth_gdf=gt,
        buffer_m=15.0,
        sample_step_m=10.0,
        compute_topology_metrics=True,
        compute_itopo=True,
        topology_radii_m=(8.0, 15.0),
        compute_hausdorff=False,
    )

    topo_f1 = float(metrics["topology"]["topo"]["by_radius_m"]["8.0"]["f1"])
    itopo_f1 = float(metrics["topology"]["i_topo"]["by_radius_m"]["8.0"]["f1"])
    assert topo_f1 < 1.0
    assert itopo_f1 < 1.0
    assert itopo_f1 <= topo_f1


def test_infer_graph_with_and_without_explicit_ids() -> None:
    gdf_with_ids = gpd.GeoDataFrame(
        {
            "u": [1, 2],
            "v": [2, 3],
            "geometry": [
                LineString([(0.0, 0.0), (50.0, 0.0)]),
                LineString([(50.0, 0.0), (100.0, 0.0)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:32634",
    )
    nodes_id, edges_id = _infer_graph(gdf_with_ids, snap_m=2.0)
    assert set(nodes_id.keys()) == {1, 2, 3}
    assert len(edges_id) == 2

    gdf_no_ids = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0.0, 0.0), (50.0, 0.0)]),
                LineString([(50.7, 0.3), (100.0, 0.0)]),
            ]
        },
        geometry="geometry",
        crs="EPSG:32634",
    )
    nodes_snap, edges_snap = _infer_graph(gdf_no_ids, snap_m=2.0)
    assert len(nodes_snap) == 3
    assert len(edges_snap) == 2


def test_context_reuse_matches_no_context_and_caches_steps() -> None:
    gt = _wgs84_gdf(_t_junction_lines(include_north_arm=True))
    gen = _wgs84_gdf(_t_junction_lines(include_north_arm=True))

    context = build_evaluation_context(
        gt,
        sample_step_m=10.0,
        topology_radii_m=(8.0, 15.0),
    )

    m_ctx = evaluate_centerline_geodataframes(
        generated_gdf=gen,
        ground_truth_gdf=gt,
        buffer_m=15.0,
        sample_step_m=10.0,
        context=context,
        compute_topology_metrics=True,
        compute_itopo=True,
        topology_radii_m=(8.0, 15.0),
    )
    m_plain = evaluate_centerline_geodataframes(
        generated_gdf=gen,
        ground_truth_gdf=gt,
        buffer_m=15.0,
        sample_step_m=10.0,
        compute_topology_metrics=True,
        compute_itopo=True,
        topology_radii_m=(8.0, 15.0),
    )

    for key in ("length_precision", "length_recall", "length_f1"):
        assert math.isclose(float(m_ctx[key]), float(m_plain[key]), rel_tol=1e-9, abs_tol=1e-9)

    for radius_key in ("8.0", "15.0"):
        a = float(m_ctx["topology"]["topo"]["by_radius_m"][radius_key]["f1"])
        b = float(m_plain["topology"]["topo"]["by_radius_m"][radius_key]["f1"])
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

    m_ctx_6 = evaluate_centerline_geodataframes(
        generated_gdf=gen,
        ground_truth_gdf=gt,
        buffer_m=15.0,
        sample_step_m=6.0,
        context=context,
        compute_topology_metrics=True,
        compute_itopo=True,
        topology_radii_m=(8.0, 15.0),
    )
    assert math.isclose(float(m_ctx_6["sample_step_m"]), 6.0)
    assert len(context.gt_sampled_points_cache) >= 2


def test_topo_one_to_one_prevents_parallel_overmatching() -> None:
    gt = _wgs84_gdf([LineString([(21.0000, 42.0050), (21.0100, 42.0050)])])
    gen = _wgs84_gdf(
        [
            LineString([(21.0000, 42.0050), (21.0100, 42.0050)]),
            LineString([(21.0000, 42.00505), (21.0100, 42.00505)]),
        ]
    )

    metrics = evaluate_centerline_geodataframes(
        generated_gdf=gen,
        ground_truth_gdf=gt,
        buffer_m=15.0,
        sample_step_m=10.0,
        compute_topology_metrics=True,
        compute_itopo=True,
        topology_radii_m=(8.0, 15.0),
        compute_hausdorff=False,
    )

    topo8 = metrics["topology"]["topo"]["by_radius_m"]["8.0"]
    assert int(topo8["matched_sample_count"]) <= int(topo8["ground_truth_sample_count"])
    assert float(topo8["precision"]) < 1.0
