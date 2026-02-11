from __future__ import annotations

import numpy as np

from centerline.postprocess_deepmg_kharita import (
    DeepMGPostprocessConfig,
    run_deepmg_kharita_postprocess,
)


def _edge(is_virtual: bool = False) -> dict:
    return {
        "support": 3.0,
        "weighted_support": 3.0,
        "effective_support": 3.0,
        "vpd_support": 1.0,
        "hpd_support": 2.0,
        "construction_sum": 0.0,
        "traffic_signal_sum": 0.0,
        "altitude_sum": 0.0,
        "altitude_count": 0,
        "crosswalk_types": set(),
        "day_counter": {},
        "hour_counter": {},
        "is_virtual": is_virtual,
        "postprocess_tags": set(),
    }


def test_dead_end_link_generation_adds_virtual_edges() -> None:
    node_xy = {
        0: (0.0, 0.0),
        1: (10.0, 0.0),   # dead-end candidate
        2: (30.0, 0.0),   # dead-end candidate
        3: (40.0, 0.0),
    }
    edge_support = {
        (0, 1): _edge(),
        (1, 0): _edge(),
        (2, 3): _edge(),
        (3, 2): _edge(),
    }
    cfg = DeepMGPostprocessConfig(
        enabled=True,
        link_radius_m=25.0,
        min_supp_virtual=0,
        pred_min_supp=0,
    )
    labels = np.array([0, 1, 2, 3], dtype=np.int32)
    trace_ranges = {0: (0, len(labels))}
    out = run_deepmg_kharita_postprocess(
        edge_support=edge_support,
        node_xy=node_xy,
        labels=labels,
        trace_ranges=trace_ranges,
        sample_point_count=100,
        config=cfg,
    )
    assert out["virtual_links_added"] >= 1
    assert any(bool(s.get("is_virtual", False)) for s in edge_support.values())


def test_virtual_link_pruned_without_support() -> None:
    node_xy = {0: (0.0, 0.0), 1: (10.0, 0.0)}
    edge_support = {
        (0, 1): _edge(is_virtual=True),
        (1, 0): _edge(is_virtual=True),
    }
    cfg = DeepMGPostprocessConfig(
        enabled=True,
        link_radius_m=50.0,
        min_supp_virtual=3,
        pred_min_supp=0,
    )
    labels = np.array([5, 6, 7], dtype=np.int32)  # unrelated nodes => no support
    trace_ranges = {0: (0, len(labels))}
    run_deepmg_kharita_postprocess(
        edge_support=edge_support,
        node_xy=node_xy,
        labels=labels,
        trace_ranges=trace_ranges,
        sample_point_count=10,
        config=cfg,
    )
    assert len(edge_support) == 0

