"""
Dynamic Weighting for VPD/Probe Fusion

Implements quality-aware weighting that balances VPD (high-precision vehicle
path data) with HPD probe data based on:
  - Heading consistency (circular concentration)
  - Temporal repeatability (multi-day presence)
  - Speed consistency (coefficient of variation)
  - VPD quality signals (path quality, sensor quality, construction status)
  - Road-likeness scoring (density, connectivity, geometry)

This module provides edge-level and point-level weighting to improve
centerline quality by emphasizing reliable data sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class DynamicWeightConfig:
    """Dynamic weighting knobs for VPD/probe fusion."""

    enabled: bool = False
    lambda_vpd: float = 1.6
    probe_repeatability_days_min: int = 2
    probe_heading_entropy_max_deg: float = 35.0
    probe_speed_cv_max: float = 0.65
    road_likeness_beta: float = 6.0
    road_likeness_tau: float = 0.45


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 35.0:
        return 1.0
    if x <= -35.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _safe_heading_consistency(edge_stats: dict) -> float:
    """
    Compute heading consistency from circular statistics.
    
    Uses the resultant vector length (R) as a measure of heading concentration.
    R = 1.0 means all headings are identical (perfectly aligned).
    R = 0.0 means headings are uniformly distributed (no dominant direction).
    """
    n = float(edge_stats.get("heading_count", 0.0) or 0.0)
    if n <= 0.0:
        return 0.5
    
    s = float(edge_stats.get("heading_sin_sum", 0.0) or 0.0)
    c = float(edge_stats.get("heading_cos_sum", 0.0) or 0.0)
    r = float(np.hypot(s, c) / max(n, 1e-6))
    return float(np.clip(r, 0.0, 1.0))


def _safe_probe_speed_consistency(edge_stats: dict, speed_cv_max: float) -> float:
    """
    Compute speed consistency from coefficient of variation.
    
    Low CV means consistent speeds (likely a road with regular traffic).
    High CV means variable speeds (possibly mixed traffic or noise).
    """
    n = float(edge_stats.get("hpd_speed_count", 0.0) or 0.0)
    if n <= 1.0:
        return 0.5
    
    s = float(edge_stats.get("hpd_speed_sum", 0.0) or 0.0)
    s2 = float(edge_stats.get("hpd_speed_sq_sum", 0.0) or 0.0)
    mean = s / max(n, 1e-6)
    
    if mean <= 1e-6:
        return 0.5
    
    var = max(0.0, (s2 / max(n, 1e-6)) - (mean * mean))
    cv = math.sqrt(var) / max(mean, 1e-6)
    return float(np.clip(1.0 - cv / max(speed_cv_max, 1e-6), 0.0, 1.0))


def _safe_temporal_repeatability(edge_stats: dict, days_min: int) -> float:
    """
    Compute temporal repeatability from day counter.
    
    Roads appear on multiple days. Noise (parking lots, shortcuts) may
    only appear once or twice.
    """
    day_counter = edge_stats.get("day_counter", None)
    if not day_counter:
        return 0.5
    n_days = float(len(day_counter))
    return float(np.clip(n_days / max(float(days_min), 1.0), 0.0, 1.0))


def _safe_vpd_quality(edge_stats: dict) -> float:
    """
    Compute VPD quality from available signals.
    
    Combines:
      - Construction status (lower is better)
      - Path quality score
      - Sensor quality score
    """
    support = float(edge_stats.get("support", 0.0) or 0.0)
    
    # Construction penalty: 0% construction = 1.0, 100% = 0.0
    construction_mean = (
        float(edge_stats.get("construction_sum", 0.0) or 0.0) / max(support, 1.0)
    )
    construction_quality = float(np.clip(1.0 - (construction_mean / 100.0), 0.0, 1.0))

    # Path and sensor quality from VPD metadata
    pq_n = float(edge_stats.get("path_quality_count", 0.0) or 0.0)
    sq_n = float(edge_stats.get("sensor_quality_count", 0.0) or 0.0)
    path_q = (
        float(edge_stats.get("path_quality_sum", 0.0) or 0.0) / max(pq_n, 1.0)
        if pq_n > 0
        else 0.5
    )
    sensor_q = (
        float(edge_stats.get("sensor_quality_sum", 0.0) or 0.0) / max(sq_n, 1.0)
        if sq_n > 0
        else 0.5
    )
    
    return float(
        np.clip(
            0.5 * path_q + 0.3 * sensor_q + 0.2 * construction_quality,
            0.0,
            1.0,
        )
    )


def _road_likeness(
    *,
    weighted_support: float,
    edge_len_m: float,
    degree_u: float,
    degree_v: float,
    temporal_repeatability: float,
    heading_consistency: float,
) -> float:
    """
    Compute road-likeness score for an edge.
    
    Combines multiple signals that distinguish roads from noise:
      - Support density (traces per meter)
      - Connectivity (node degree at endpoints)
      - Temporal repeatability (multi-day presence)
      - Heading consistency (aligned traces)
    """
    # Support density normalized to typical road values
    support_density = weighted_support / max(edge_len_m, 1e-6)
    density_n = float(np.clip(support_density / 0.35, 0.0, 1.0))
    
    # Connectivity: higher degree = more likely to be a real intersection
    connectivity_n = float(np.clip((degree_u + degree_v) / 8.0, 0.0, 1.0))
    
    # Weighted combination
    score = (
        0.35 * density_n
        + 0.25 * connectivity_n
        + 0.20 * temporal_repeatability
        + 0.20 * heading_consistency
    )
    return float(np.clip(score, 0.0, 1.0))


def apply_dynamic_weighting_to_edges(
    *,
    edge_support: Dict[Tuple[int, int], dict],
    node_xy: Dict[int, Tuple[float, float]],
    config: DynamicWeightConfig,
) -> None:
    """
    Annotate in-place dynamic weights and effective support per edge.
    
    For each edge, computes:
      - dyn_w_probe: Weight for probe contribution
      - dyn_w_vpd: Weight for VPD contribution  
      - road_likeness_score: Overall road-likeness
      - probe_consistency: Probe data quality
      - vpd_quality: VPD data quality
      - effective_support: Quality-weighted support value
    """
    if not edge_support:
        return

    # Compute node degrees (undirected)
    deg: Dict[int, int] = {}
    for u, v in edge_support:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    for (u, v), s in edge_support.items():
        support = float(s.get("support", 0.0) or 0.0)
        weighted_support = float(s.get("weighted_support", 0.0) or 0.0)
        vpd_support = float(s.get("vpd_support", 0.0) or 0.0)
        probe_support = float(s.get("hpd_support", 0.0) or 0.0)

        # Edge length
        x1, y1 = node_xy.get(u, (0.0, 0.0))
        x2, y2 = node_xy.get(v, (0.0, 0.0))
        edge_len_m = float(np.hypot(x2 - x1, y2 - y1))

        # Quality signals
        heading_consistency = _safe_heading_consistency(s)
        speed_consistency = _safe_probe_speed_consistency(s, config.probe_speed_cv_max)
        temporal_repeatability = _safe_temporal_repeatability(
            s, config.probe_repeatability_days_min
        )
        
        # Combined probe consistency
        probe_consistency = float(
            np.clip(
                0.45 * heading_consistency
                + 0.35 * temporal_repeatability
                + 0.20 * speed_consistency,
                0.0,
                1.0,
            )
        )
        vpd_quality = _safe_vpd_quality(s)

        # Road-likeness score
        road_likeness = _road_likeness(
            weighted_support=weighted_support,
            edge_len_m=edge_len_m,
            degree_u=float(deg.get(u, 0)),
            degree_v=float(deg.get(v, 0)),
            temporal_repeatability=temporal_repeatability,
            heading_consistency=heading_consistency,
        )

        # Dynamic weight computation
        denom = (
            probe_support * probe_consistency
            + config.lambda_vpd * vpd_support * vpd_quality
            + 1e-6
        )
        w_probe_raw = (
            (probe_support * probe_consistency) / denom if denom > 0.0 else 0.5
        )
        
        # Gate by road-likeness (low road-likeness = reduce probe weight)
        gate = _sigmoid(config.road_likeness_beta * (road_likeness - config.road_likeness_tau))
        w_probe = float(np.clip(w_probe_raw * gate, 0.0, 1.0))
        w_vpd = float(np.clip(1.0 - w_probe, 0.0, 1.0))

        # Effective support with quality adjustment
        if config.enabled:
            source_factor = 0.8 + 0.4 * max(w_probe, w_vpd)
            quality_factor = 0.35 + 0.65 * road_likeness
            effective_support = weighted_support * source_factor * quality_factor
        else:
            effective_support = weighted_support

        # Store computed values
        s["dyn_w_probe"] = w_probe
        s["dyn_w_vpd"] = w_vpd
        s["road_likeness_score"] = road_likeness
        s["probe_consistency"] = probe_consistency
        s["vpd_quality"] = vpd_quality
        s["effective_support"] = float(max(effective_support, 0.0))


def compute_trace_weight(
    *,
    source: str,
    length_m: float,
    n_points: int,
    heading_consistency: float = 0.5,
    path_quality: float = 0.5,
    sensor_quality: float = 0.5,
    construction_percent: float = 0.0,
    vpd_base_weight: float = 1.2,
    hpd_base_weight: float = 1.0,
) -> float:
    """
    Compute quality weight for a trace.
    
    Used during trace sampling to weight points by source reliability.
    """
    # Base weight by source
    base_w = vpd_base_weight if source.upper() == "VPD" else hpd_base_weight
    
    # Length factor (longer traces are more reliable)
    length_factor = 1.0 / (1.0 + math.exp(-(length_m - 50) / 80))
    length_factor = 0.3 + 0.7 * length_factor
    
    # Point density factor
    pts_per_m = n_points / max(length_m, 1.0)
    density_factor = min(1.0, pts_per_m * 5.0)
    
    # Quality boosts from metadata
    base_w += 0.25 * float(np.clip(path_quality, 0.0, 1.0))
    base_w += 0.15 * float(np.clip(sensor_quality, 0.0, 1.0))
    
    # Construction penalty
    construction_penalty = 0.65 * (construction_percent / 100.0)
    base_w *= max(0.35, 1.0 - construction_penalty)
    
    # Combine factors
    weight = base_w * length_factor * (0.4 + 0.3 * density_factor + 0.3 * heading_consistency)
    return float(np.clip(weight, 0.1, 3.0))
