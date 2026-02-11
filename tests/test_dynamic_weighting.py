from __future__ import annotations

from centerline.dynamic_weighting import DynamicWeightConfig, apply_dynamic_weighting_to_edges


def _base_edge(vpd: float, hpd: float) -> dict:
    return {
        "support": vpd + hpd,
        "weighted_support": vpd + hpd,
        "vpd_support": vpd,
        "hpd_support": hpd,
        "construction_sum": 0.0,
        "path_quality_sum": 1.0,
        "path_quality_count": 1.0,
        "sensor_quality_sum": 1.0,
        "sensor_quality_count": 1.0,
        "day_counter": {1: 2, 2: 1},
        "heading_sin_sum": 1.0,
        "heading_cos_sum": 8.0,
        "heading_count": 8.0,
        "hpd_speed_sum": 100.0,
        "hpd_speed_sq_sum": 2500.0,
        "hpd_speed_count": 10.0,
    }


def test_probe_weight_increases_with_probe_support() -> None:
    cfg = DynamicWeightConfig(enabled=True)
    node_xy = {0: (0.0, 0.0), 1: (20.0, 0.0)}

    low = {(0, 1): _base_edge(vpd=10.0, hpd=2.0)}
    high = {(0, 1): _base_edge(vpd=10.0, hpd=8.0)}
    apply_dynamic_weighting_to_edges(edge_support=low, node_xy=node_xy, config=cfg)
    apply_dynamic_weighting_to_edges(edge_support=high, node_xy=node_xy, config=cfg)

    assert high[(0, 1)]["dyn_w_probe"] > low[(0, 1)]["dyn_w_probe"]


def test_probe_weight_decreases_with_stronger_vpd_signal() -> None:
    cfg = DynamicWeightConfig(enabled=True)
    node_xy = {0: (0.0, 0.0), 1: (20.0, 0.0)}

    low_vpd = {(0, 1): _base_edge(vpd=3.0, hpd=8.0)}
    high_vpd = {(0, 1): _base_edge(vpd=12.0, hpd=8.0)}
    apply_dynamic_weighting_to_edges(edge_support=low_vpd, node_xy=node_xy, config=cfg)
    apply_dynamic_weighting_to_edges(edge_support=high_vpd, node_xy=node_xy, config=cfg)

    assert high_vpd[(0, 1)]["dyn_w_probe"] < low_vpd[(0, 1)]["dyn_w_probe"]

