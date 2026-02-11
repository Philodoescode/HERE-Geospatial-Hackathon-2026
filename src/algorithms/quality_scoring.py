"""
Quality Scoring & Dynamic Weighting Module

Implements road-likeness scoring and dynamic VPD/HPD weighting 
adapted from research algorithms (Kharita, Roadster, DeepMG).

Key concepts:
  - Road-likeness: Multi-factor score (density, connectivity, temporal, heading)
  - Dynamic weighting: Source-aware fusion between VPD (high-precision) and HPD (high-recall)
  - Probe consistency: Heading alignment, speed consistency, temporal repeatability
  - Candidate selection: Filter weak edges based on composite quality score
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString


@dataclass
class QualityConfig:
    """Configuration for quality scoring and dynamic weighting."""
    
    # Dynamic weighting parameters
    enabled: bool = True
    lambda_vpd: float = 1.6  # VPD preference factor (>1 favors VPD)
    
    # Probe consistency thresholds
    probe_repeatability_days_min: int = 2
    probe_heading_entropy_max_deg: float = 35.0
    probe_speed_cv_max: float = 0.65
    
    # Road-likeness parameters
    road_likeness_beta: float = 6.0
    road_likeness_tau: float = 0.45
    support_density_scale: float = 0.35  # support/meter threshold
    connectivity_max_degree: float = 8.0
    
    # Candidate selection thresholds
    candidate_selection_enabled: bool = True
    candidate_selection_threshold: float = 0.52
    candidate_force_keep_support: float = 18.0
    candidate_short_length_m: float = 10.0
    candidate_low_support: float = 5.0
    candidate_dangling_max_length_m: float = 35.0
    candidate_dangling_min_support: float = 8.0
    candidate_length_scale_m: float = 70.0
    candidate_density_scale: float = 0.25


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 35.0:
        return 1.0
    if x <= -35.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _angle_diff_deg(a: float, b: float) -> float:
    """Compute minimal angle difference between two headings (0-180)."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def compute_heading_consistency(heading_sin_sum: float, heading_cos_sum: float, 
                                  count: float) -> float:
    """
    Compute heading consistency from circular statistics.
    
    Higher values (→1.0) mean headings are highly aligned.
    Uses resultant vector length as concentration measure.
    """
    if count <= 0.0:
        return 0.5  # neutral if no data
    
    r = float(np.hypot(heading_sin_sum, heading_cos_sum) / max(count, 1e-6))
    return float(np.clip(r, 0.0, 1.0))


def compute_speed_consistency(speed_sum: float, speed_sq_sum: float, 
                               count: float, cv_max: float = 0.65) -> float:
    """
    Compute speed consistency based on coefficient of variation.
    
    Low CV (consistent speeds) → high score.
    High CV (variable speeds, e.g. stop-and-go) → low score.
    """
    if count <= 1.0:
        return 0.5
    
    mean = speed_sum / max(count, 1e-6)
    if mean <= 1e-6:
        return 0.5
    
    var = max(0.0, (speed_sq_sum / max(count, 1e-6)) - (mean * mean))
    cv = math.sqrt(var) / max(mean, 1e-6)
    
    return float(np.clip(1.0 - cv / max(cv_max, 1e-6), 0.0, 1.0))


def compute_temporal_repeatability(day_counter: Counter, days_min: int = 2) -> float:
    """
    Compute temporal repeatability score.
    
    Roads should have traces appearing on multiple days.
    Noise/parking lots typically only appear once.
    """
    if not day_counter:
        return 0.5
    
    n_days = float(len(day_counter))
    return float(np.clip(n_days / max(float(days_min), 1.0), 0.0, 1.0))


def compute_vpd_quality(support: float, construction_sum: float,
                         path_quality_sum: float, path_quality_count: float,
                         sensor_quality_sum: float, sensor_quality_count: float) -> float:
    """
    Compute VPD quality score.
    
    Factors:
      - Construction zones (lower quality during construction)
      - Path quality score from fused data
      - Sensor quality score
    """
    # Construction penalty (100% construction → 0 quality)
    construction_mean = construction_sum / max(support, 1.0)
    construction_quality = float(np.clip(1.0 - (construction_mean / 100.0), 0.0, 1.0))
    
    # Path quality
    path_q = (
        path_quality_sum / max(path_quality_count, 1.0)
        if path_quality_count > 0 else 0.5
    )
    
    # Sensor quality
    sensor_q = (
        sensor_quality_sum / max(sensor_quality_count, 1.0)
        if sensor_quality_count > 0 else 0.5
    )
    
    # Weighted combination
    return float(np.clip(
        0.5 * path_q + 0.3 * sensor_q + 0.2 * construction_quality,
        0.0, 1.0
    ))


def compute_road_likeness(
    weighted_support: float,
    edge_len_m: float,
    degree_u: float,
    degree_v: float,
    temporal_repeatability: float,
    heading_consistency: float,
    config: Optional[QualityConfig] = None
) -> float:
    """
    Compute road-likeness score for an edge.
    
    Combines multiple factors:
      - Support density (traces per meter)
      - Network connectivity (node degrees)
      - Temporal repeatability (multi-day presence)
      - Heading consistency (aligned headings)
    
    Returns value in [0, 1] where 1 = definitely a road.
    """
    cfg = config or QualityConfig()
    
    # Support density
    support_density = weighted_support / max(edge_len_m, 1e-6)
    density_n = float(np.clip(support_density / cfg.support_density_scale, 0.0, 1.0))
    
    # Connectivity
    connectivity_n = float(np.clip(
        (degree_u + degree_v) / cfg.connectivity_max_degree, 0.0, 1.0
    ))
    
    # Weighted combination
    score = (
        0.35 * density_n +
        0.25 * connectivity_n +
        0.20 * temporal_repeatability +
        0.20 * heading_consistency
    )
    
    return float(np.clip(score, 0.0, 1.0))


def compute_probe_consistency(
    heading_consistency: float,
    temporal_repeatability: float,
    speed_consistency: float
) -> float:
    """
    Compute overall probe data consistency score.
    
    High consistency → reliable probe data, trust it more.
    Low consistency → noisy/unreliable, prefer VPD.
    """
    return float(np.clip(
        0.45 * heading_consistency +
        0.35 * temporal_repeatability +
        0.20 * speed_consistency,
        0.0, 1.0
    ))


def compute_dynamic_weights(
    vpd_support: float,
    vpd_quality: float,
    probe_support: float,
    probe_consistency: float,
    road_likeness: float,
    config: Optional[QualityConfig] = None
) -> Tuple[float, float, float]:
    """
    Compute dynamic weights for VPD vs probe fusion.
    
    Returns:
        (w_vpd, w_probe, effective_support)
        
    The effective_support incorporates quality factors for filtering.
    """
    cfg = config or QualityConfig()
    
    # Raw weight calculation
    denom = (
        probe_support * probe_consistency +
        cfg.lambda_vpd * vpd_support * vpd_quality +
        1e-6
    )
    
    w_probe_raw = (probe_support * probe_consistency) / denom if denom > 0.0 else 0.5
    
    # Road-likeness gate: high road-likeness → trust probe more
    gate = _sigmoid(cfg.road_likeness_beta * (road_likeness - cfg.road_likeness_tau))
    
    w_probe = float(np.clip(w_probe_raw * gate, 0.0, 1.0))
    w_vpd = float(np.clip(1.0 - w_probe, 0.0, 1.0))
    
    # Effective support for downstream filtering
    weighted_support = vpd_support * vpd_quality + probe_support * probe_consistency
    source_factor = 0.8 + 0.4 * max(w_probe, w_vpd)
    quality_factor = 0.35 + 0.65 * road_likeness
    effective_support = weighted_support * source_factor * quality_factor
    
    return w_vpd, w_probe, float(max(effective_support, 0.0))


def apply_quality_scoring_to_segments(
    gdf: pd.DataFrame,
    config: Optional[QualityConfig] = None
) -> pd.DataFrame:
    """
    Apply quality scoring to a GeoDataFrame of line segments.
    
    Adds columns:
      - road_likeness_score
      - effective_support
      - is_candidate (for candidate selection)
      - selection_score
    
    Expects columns like 'source', 'geometry', and optionally:
      - support, weighted_support
      - day_counter, speed stats, heading stats
    """
    cfg = config or QualityConfig()
    
    if gdf.empty:
        return gdf
    
    result = gdf.copy()
    n = len(result)
    
    # Initialize scores
    road_likeness = np.full(n, 0.5, dtype=np.float64)
    effective_support = np.full(n, 1.0, dtype=np.float64)
    
    # Compute connectivity (node degrees)
    # Use endpoint coordinates as node identifiers
    degree = Counter()
    for idx, row in result.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            start = tuple(np.round(geom.coords[0], 1))
            end = tuple(np.round(geom.coords[-1], 1))
            degree[start] += 1
            degree[end] += 1
    
    for i, row in enumerate(result.itertuples(index=False)):
        geom = row.geometry
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            continue
        
        start = tuple(np.round(geom.coords[0], 1))
        end = tuple(np.round(geom.coords[-1], 1))
        deg_u = float(degree.get(start, 1))
        deg_v = float(degree.get(end, 1))
        
        edge_len_m = geom.length
        
        # Get support if available (from Phase 3/4 data)
        support = getattr(row, 'weighted_support', 1.0) if hasattr(row, 'weighted_support') else 1.0
        if support is None:
            support = 1.0
        
        # Get heading consistency if available (from Phase 3 data)
        heading_cons = getattr(row, 'heading_consistency', 0.5) if hasattr(row, 'heading_consistency') else 0.5
        if heading_cons is None:
            heading_cons = 0.5
        
        # Default temporal (could be derived from date columns if available)
        temporal = 0.5
        
        # Compute road-likeness
        rl = compute_road_likeness(
            weighted_support=support,
            edge_len_m=edge_len_m,
            degree_u=deg_u,
            degree_v=deg_v,
            temporal_repeatability=temporal,
            heading_consistency=heading_cons,
            config=cfg
        )
        road_likeness[i] = rl
        
        # Effective support
        source = getattr(row, 'source', 'VPD') if hasattr(row, 'source') else 'VPD'
        if str(source).upper() == 'VPD':
            effective_support[i] = support * (0.5 + 0.5 * rl)
        else:
            effective_support[i] = support * (0.3 + 0.7 * rl)
    
    result['road_likeness_score'] = road_likeness
    result['effective_support'] = effective_support
    
    # Candidate selection
    if cfg.candidate_selection_enabled:
        result = _apply_candidate_selection(result, cfg)
    else:
        result['is_candidate'] = True
        result['selection_score'] = 1.0
        result['selection_reason'] = 'selection_disabled'
    
    return result


def _apply_candidate_selection(gdf: pd.DataFrame, 
                                config: QualityConfig) -> pd.DataFrame:
    """Apply candidate selection filtering based on quality scores."""
    out = gdf.copy()
    
    if out.empty:
        out['is_candidate'] = pd.Series(dtype=bool)
        out['selection_score'] = pd.Series(dtype=np.float64)
        out['selection_reason'] = pd.Series(dtype=object)
        return out
    
    # Compute node degrees for connectivity
    degree = Counter()
    for idx, row in out.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            start = tuple(np.round(geom.coords[0], 1))
            end = tuple(np.round(geom.coords[-1], 1))
            degree[start] += 1
            degree[end] += 1
    
    # Get metrics
    lengths = out.geometry.length.values
    road_likeness = out.get('road_likeness_score', pd.Series([0.5] * len(out))).values
    eff_support = out.get('effective_support', pd.Series([1.0] * len(out))).values
    
    # Compute per-segment metrics
    u_deg = np.zeros(len(out), dtype=np.float64)
    v_deg = np.zeros(len(out), dtype=np.float64)
    
    for i, row in enumerate(out.itertuples(index=False)):
        geom = row.geometry
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            start = tuple(np.round(geom.coords[0], 1))
            end = tuple(np.round(geom.coords[-1], 1))
            u_deg[i] = float(degree.get(start, 1))
            v_deg[i] = float(degree.get(end, 1))
    
    dangling = ((u_deg <= 1.0) | (v_deg <= 1.0)).astype(np.float64)
    
    # Normalized metrics
    support_n = np.clip(eff_support / max(config.candidate_force_keep_support, 1e-6), 0.0, 1.0)
    length_n = np.clip(lengths / max(config.candidate_length_scale_m, 1e-6), 0.0, 1.0)
    connectivity_n = np.clip((u_deg + v_deg) / 8.0, 0.0, 1.0)
    connectivity_n = np.where(dangling > 0.0, connectivity_n * 0.35, np.maximum(connectivity_n, 0.65))
    
    # Selection score
    score = (
        0.35 * support_n +
        0.25 * road_likeness +
        0.20 * length_n +
        0.20 * connectivity_n
    )
    
    # Selection logic
    selected = np.ones(len(out), dtype=bool)
    reasons: List[str] = []
    
    for i in range(len(out)):
        reason = "score_threshold"
        
        # Force keep high-support edges
        if eff_support[i] >= config.candidate_force_keep_support:
            reason = "force_keep_support"
            selected[i] = True
        # Drop short, low-support edges
        elif (lengths[i] < config.candidate_short_length_m and 
              eff_support[i] < config.candidate_low_support):
            reason = "drop_short_low_support"
            selected[i] = False
        # Drop weak dangling edges
        elif (dangling[i] > 0.0 and 
              lengths[i] < config.candidate_dangling_max_length_m and
              eff_support[i] < config.candidate_dangling_min_support):
            reason = "drop_dangling_weak"
            selected[i] = False
        # Standard threshold
        else:
            selected[i] = bool(score[i] >= config.candidate_selection_threshold)
            if not selected[i]:
                reason = "drop_low_score"
        
        reasons.append(reason)
    
    out['is_candidate'] = selected
    out['selection_score'] = score
    out['selection_reason'] = reasons
    out['u_degree'] = u_deg
    out['v_degree'] = v_deg
    out['is_dangling'] = dangling.astype(bool)
    
    return out


class HeadingAwareClusterer:
    """
    Heading-aware incremental clustering (Kharita-style).
    
    Clusters points by both spatial proximity AND heading similarity.
    This preserves dual carriageways and prevents merging opposite directions.
    """
    
    def __init__(
        self,
        cluster_radius_m: float = 10.0,
        heading_tolerance_deg: float = 45.0,
        heading_distance_weight_m: float = 0.22,
        min_cluster_points: int = 1
    ):
        self.cluster_radius_m = cluster_radius_m
        self.heading_tolerance_deg = heading_tolerance_deg
        self.heading_distance_weight_m = heading_distance_weight_m
        self.min_cluster_points = min_cluster_points
    
    def cluster(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        headings: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Perform heading-aware clustering.
        
        Args:
            xs: X coordinates (projected)
            ys: Y coordinates (projected)
            headings: Heading angles in degrees (0-360)
            weights: Optional point weights
            
        Returns:
            (labels, nodes_df) where:
              - labels: cluster assignment per point
              - nodes_df: DataFrame with cluster centroids
        """
        n = len(xs)
        if n == 0:
            return np.array([], dtype=np.int32), pd.DataFrame(
                columns=["node_id", "x", "y", "heading", "weight", "point_count"]
            )
        
        if weights is None:
            weights = np.ones(n, dtype=np.float32)
        
        labels = np.full(n, -1, dtype=np.int32)
        
        # Cluster state
        cx: List[float] = []
        cy: List[float] = []
        cweight: List[float] = []
        csin: List[float] = []
        ccos: List[float] = []
        ccount: List[int] = []
        
        tree = None
        dirty_count = 0
        
        def rebuild_tree():
            nonlocal tree
            if cx:
                arr = np.column_stack([
                    np.asarray(cx, dtype=np.float32),
                    np.asarray(cy, dtype=np.float32)
                ])
                tree = cKDTree(arr)
        
        for i in range(n):
            x = float(xs[i])
            y = float(ys[i])
            heading = float(headings[i])
            w = float(max(weights[i], 1e-6))
            
            # First point starts first cluster
            if not cx:
                cx.append(x)
                cy.append(y)
                cweight.append(w)
                rad = math.radians(heading)
                csin.append(math.sin(rad) * w)
                ccos.append(math.cos(rad) * w)
                ccount.append(1)
                labels[i] = 0
                rebuild_tree()
                continue
            
            # Periodic tree rebuild
            if tree is None or dirty_count >= 2000 or i % 20000 == 0:
                rebuild_tree()
                dirty_count = 0
            
            # Find nearby clusters
            candidate_ids = (
                tree.query_ball_point([x, y], r=self.cluster_radius_m)
                if tree is not None else []
            )
            
            # Find best matching cluster (heading-aware)
            best_cid = -1
            best_score = float("inf")
            
            for cid in candidate_ids:
                cheading = (math.degrees(math.atan2(csin[cid], ccos[cid])) + 360.0) % 360.0
                ad = _angle_diff_deg(heading, cheading)
                
                if ad > self.heading_tolerance_deg:
                    continue
                
                sd = math.hypot(x - cx[cid], y - cy[cid])
                score = sd + self.heading_distance_weight_m * ad
                
                if score < best_score:
                    best_score = score
                    best_cid = cid
            
            # Create new cluster or merge
            if best_cid == -1:
                cid = len(cx)
                cx.append(x)
                cy.append(y)
                cweight.append(w)
                rad = math.radians(heading)
                csin.append(math.sin(rad) * w)
                ccos.append(math.cos(rad) * w)
                ccount.append(1)
                labels[i] = cid
                dirty_count += 1
            else:
                # Merge into existing cluster
                cid = best_cid
                old_w = cweight[cid]
                new_w = old_w + w
                cx[cid] = (cx[cid] * old_w + x * w) / new_w
                cy[cid] = (cy[cid] * old_w + y * w) / new_w
                cweight[cid] = new_w
                rad = math.radians(heading)
                csin[cid] += math.sin(rad) * w
                ccos[cid] += math.cos(rad) * w
                ccount[cid] += 1
                labels[i] = cid
                dirty_count += 1
        
        # Compact labels and build output
        unique = sorted(set(int(v) for v in labels if v >= 0))
        to_new = {old: new for new, old in enumerate(unique)}
        labels = np.asarray([to_new.get(int(v), -1) for v in labels], dtype=np.int32)
        
        # Build node DataFrame
        rows = []
        for old_id in unique:
            new_id = to_new[old_id]
            heading = (math.degrees(math.atan2(csin[old_id], ccos[old_id])) + 360.0) % 360.0
            rows.append({
                "node_id": new_id,
                "x": cx[old_id],
                "y": cy[old_id],
                "heading": heading,
                "weight": cweight[old_id],
                "point_count": ccount[old_id],
            })
        
        return labels, pd.DataFrame(rows)


def transitive_prune_edges(
    edge_list: List[Tuple[int, int, float]],
    node_xy: Dict[int, Tuple[float, float]],
    edge_support: Dict[Tuple[int, int], float],
    max_hops: int = 4,
    ratio: float = 1.03,
    max_checks: int = 25000
) -> List[Tuple[int, int]]:
    """
    Remove transitive edges (shortcuts) when alternative path exists.
    
    If edge A→B can be reached via A→C→...→B with similar or shorter
    distance, the direct edge is considered redundant.
    
    Args:
        edge_list: List of (u, v, length) edges
        node_xy: Node coordinates
        edge_support: Support values per edge
        max_hops: Maximum path length to check
        ratio: Distance ratio threshold (1.03 = 3% longer is acceptable)
        max_checks: Maximum edges to consider for pruning
        
    Returns:
        List of edges to remove
    """
    from collections import defaultdict
    import heapq
    
    # Build adjacency
    graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    edge_lengths = {}
    
    for u, v, length in edge_list:
        graph[u].append((v, length))
        edge_lengths[(u, v)] = length
    
    def shortest_alternative(source: int, target: int, skip_edge: Tuple[int, int], 
                              max_dist: float) -> float:
        """Find shortest path avoiding the direct edge, with hop limit."""
        dist = {source: 0.0}
        hops = {source: 0}
        pq = [(0.0, 0, source)]
        
        while pq:
            d, h, u = heapq.heappop(pq)
            
            if u == target:
                return d
            
            if d > dist.get(u, float("inf")):
                continue
            
            if h >= max_hops:
                continue
            
            for v, length in graph[u]:
                if (u, v) == skip_edge:
                    continue
                
                nd = d + length
                nh = h + 1
                
                if nd > max_dist:
                    continue
                
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    hops[v] = nh
                    heapq.heappush(pq, (nd, nh, v))
        
        return float("inf")
    
    # Sort by support (lowest first) for pruning priority
    candidates = sorted(
        [(e, edge_support.get(e, 0.0)) for e in edge_lengths.keys()],
        key=lambda x: (x[1], -edge_lengths[x[0]])
    )[:max_checks]
    
    to_remove = []
    
    for (u, v), _ in candidates:
        if (u, v) not in edge_lengths:
            continue
        
        direct = edge_lengths[(u, v)]
        if direct <= 1.0:
            continue
        
        alt = shortest_alternative(u, v, (u, v), direct * ratio)
        
        if np.isfinite(alt) and alt <= direct * ratio:
            to_remove.append((u, v))
    
    return to_remove


# Convenience function for integration
def enhance_segments_with_quality(
    gdf: pd.DataFrame,
    config: Optional[QualityConfig] = None
) -> pd.DataFrame:
    """
    Main entry point for adding quality scores to segment GeoDataFrame.
    
    Use this in Phase 4 or Phase 5 to filter segments by quality.
    """
    return apply_quality_scoring_to_segments(gdf, config)
