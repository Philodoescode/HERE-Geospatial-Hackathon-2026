"""
Quality Metrics for Double-Digitized Geometry Cleaning

This module computes various quality metrics to evaluate the cleaning results:

1. Geometric Fidelity:
   - RMSE: Root Mean Square Error between original and fitted lines

2. Topological Consistency:
   - F1-Score (TOPO): Precision and recall for graph structure
   - RelGED: Relative Graph Edit Distance

3. Navigation Performance:
   - APLS: Average Path Length Similarity

4. Lane-Specific Quality:
   - Lane Width Deviation
   - Centerline Smoothness (Curvature Continuity)

Author: Augment Code
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


@dataclass
class QualityMetrics:
    """Container for all quality metrics."""
    # Geometric Fidelity
    rmse: float = 0.0
    
    # Topological Consistency
    topo_precision: float = 0.0
    topo_recall: float = 0.0
    topo_f1: float = 0.0
    rel_ged: float = 0.0
    
    # Navigation Performance
    apls: float = 0.0
    
    # Lane-Specific Quality
    lane_width_mean: float = 0.0
    lane_width_std: float = 0.0
    lane_width_deviation: float = 0.0  # Deviation from target
    curvature_continuity: float = 0.0  # G2 continuity measure
    smoothness_score: float = 0.0
    
    # Summary statistics
    input_segment_count: int = 0
    output_segment_count: int = 0
    reduction_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'geometric_fidelity': {
                'rmse': self.rmse,
            },
            'topological_consistency': {
                'precision': self.topo_precision,
                'recall': self.topo_recall,
                'f1_score': self.topo_f1,
                'relative_ged': self.rel_ged,
            },
            'navigation_performance': {
                'apls': self.apls,
            },
            'lane_specific': {
                'lane_width_mean': self.lane_width_mean,
                'lane_width_std': self.lane_width_std,
                'lane_width_deviation': self.lane_width_deviation,
                'curvature_continuity': self.curvature_continuity,
                'smoothness_score': self.smoothness_score,
            },
            'summary': {
                'input_segments': self.input_segment_count,
                'output_segments': self.output_segment_count,
                'reduction_percent': self.reduction_percent,
            }
        }


# =============================================================================
# Geometric Fidelity Metrics
# =============================================================================

def compute_rmse(
    original_gdf: gpd.GeoDataFrame,
    cleaned_gdf: gpd.GeoDataFrame,
    sample_spacing: float = 5.0
) -> float:
    """
    Compute Root Mean Square Error between original and cleaned geometries.
    
    Measures the spatial deviation between original probe points and fitted smooth lines.
    
    Args:
        original_gdf: Original GeoDataFrame
        cleaned_gdf: Cleaned GeoDataFrame
        sample_spacing: Spacing for sampling points along lines
        
    Returns:
        RMSE value in the same units as the geometry
    """
    if len(original_gdf) == 0 or len(cleaned_gdf) == 0:
        return 0.0
    
    # Sample points from original geometries
    original_points = []
    for geom in original_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        num_samples = max(2, int(geom.length / sample_spacing))
        for i in range(num_samples + 1):
            frac = i / num_samples
            pt = geom.interpolate(frac, normalized=True)
            original_points.append([pt.x, pt.y])
    
    if not original_points:
        return 0.0
    
    original_points = np.array(original_points)
    
    # Build spatial index for cleaned geometries
    cleaned_points = []
    for geom in cleaned_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        num_samples = max(2, int(geom.length / sample_spacing))
        for i in range(num_samples + 1):
            frac = i / num_samples
            pt = geom.interpolate(frac, normalized=True)
            cleaned_points.append([pt.x, pt.y])
    
    if not cleaned_points:
        return 0.0
    
    cleaned_points = np.array(cleaned_points)
    tree = cKDTree(cleaned_points)
    
    # Find nearest distance for each original point
    distances, _ = tree.query(original_points, k=1)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    
    return float(rmse)


# =============================================================================
# Topological Consistency Metrics
# =============================================================================

def build_graph_from_gdf(gdf: gpd.GeoDataFrame, snap_tolerance: float = 2.0) -> nx.Graph:
    """
    Build a NetworkX graph from a GeoDataFrame of LineStrings.
    
    Args:
        gdf: GeoDataFrame with LineString geometries
        snap_tolerance: Tolerance for snapping endpoints
        
    Returns:
        NetworkX Graph
    """
    G = nx.Graph()
    
    # Collect all endpoints
    endpoints = []
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        endpoints.append((coords[0], idx, 'start'))
        endpoints.append((coords[-1], idx, 'end'))
    
    if not endpoints:
        return G
    
    # Cluster endpoints
    coords = np.array([e[0] for e in endpoints])
    tree = cKDTree(coords)
    
    # Find clusters
    node_map = {}
    node_counter = 0
    visited = set()
    
    for i in range(len(endpoints)):
        if i in visited:
            continue
        
        # Find all points within tolerance
        nearby = tree.query_ball_point(coords[i], snap_tolerance)
        
        # Assign same node ID to all nearby points
        centroid = np.mean(coords[nearby], axis=0)
        node_id = node_counter
        node_counter += 1
        
        for j in nearby:
            node_map[j] = (node_id, tuple(centroid))
            visited.add(j)
    
    # Build graph
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        
        start_idx = idx * 2
        end_idx = idx * 2 + 1
        
        if start_idx in node_map and end_idx in node_map:
            u = node_map[start_idx][0]
            v = node_map[end_idx][0]
            
            if u != v:
                G.add_edge(u, v, length=geom.length, geometry=geom)
    
    return G


def compute_topo_f1(
    original_gdf: gpd.GeoDataFrame,
    cleaned_gdf: gpd.GeoDataFrame,
    buffer_distance: float = 5.0
) -> Tuple[float, float, float]:
    """
    Compute topological F1-score (precision and recall for graph structure).
    
    Precision: percentage of output segments that match real roads
    Recall: percentage of real roads captured in output
    
    Args:
        original_gdf: Original (ground truth) GeoDataFrame
        cleaned_gdf: Cleaned GeoDataFrame
        buffer_distance: Buffer distance for matching
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if len(original_gdf) == 0 or len(cleaned_gdf) == 0:
        return 0.0, 0.0, 0.0
    
    # Create buffers around original geometries
    original_union = original_gdf.geometry.buffer(buffer_distance).unary_union
    
    # Create buffers around cleaned geometries
    cleaned_union = cleaned_gdf.geometry.buffer(buffer_distance).unary_union
    
    # Calculate precision: how much of cleaned is within original
    cleaned_total_length = sum(g.length for g in cleaned_gdf.geometry if g is not None)
    matched_cleaned_length = 0
    
    for geom in cleaned_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        intersection = geom.intersection(original_union)
        if not intersection.is_empty:
            matched_cleaned_length += intersection.length
    
    precision = matched_cleaned_length / cleaned_total_length if cleaned_total_length > 0 else 0
    
    # Calculate recall: how much of original is within cleaned
    original_total_length = sum(g.length for g in original_gdf.geometry if g is not None)
    matched_original_length = 0
    
    for geom in original_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        intersection = geom.intersection(cleaned_union)
        if not intersection.is_empty:
            matched_original_length += intersection.length
    
    recall = matched_original_length / original_total_length if original_total_length > 0 else 0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return float(precision), float(recall), float(f1)


def compute_relative_ged(
    original_gdf: gpd.GeoDataFrame,
    cleaned_gdf: gpd.GeoDataFrame,
    snap_tolerance: float = 2.0
) -> float:
    """
    Compute Relative Graph Edit Distance.
    
    Measures the number of graph edit operations needed to transform
    the cleaned graph into the original graph, normalized by graph size.
    
    Args:
        original_gdf: Original GeoDataFrame
        cleaned_gdf: Cleaned GeoDataFrame
        snap_tolerance: Tolerance for building graphs
        
    Returns:
        Relative GED (lower is better)
    """
    G_orig = build_graph_from_gdf(original_gdf, snap_tolerance)
    G_clean = build_graph_from_gdf(cleaned_gdf, snap_tolerance)
    
    # Simple approximation: difference in nodes and edges
    node_diff = abs(G_orig.number_of_nodes() - G_clean.number_of_nodes())
    edge_diff = abs(G_orig.number_of_edges() - G_clean.number_of_edges())
    
    total_orig = G_orig.number_of_nodes() + G_orig.number_of_edges()
    
    if total_orig == 0:
        return 0.0
    
    rel_ged = (node_diff + edge_diff) / total_orig

    return float(rel_ged)


# =============================================================================
# Navigation Performance Metrics
# =============================================================================

def compute_apls(
    original_gdf: gpd.GeoDataFrame,
    cleaned_gdf: gpd.GeoDataFrame,
    num_samples: int = 100,
    snap_tolerance: float = 5.0
) -> float:
    """
    Compute Average Path Length Similarity (APLS).

    Compares optimal routing path lengths between original and cleaned networks.

    Args:
        original_gdf: Original GeoDataFrame
        cleaned_gdf: Cleaned GeoDataFrame
        num_samples: Number of random node pairs to sample
        snap_tolerance: Tolerance for building graphs

    Returns:
        APLS score (0-1, higher is better)
    """
    G_orig = build_graph_from_gdf(original_gdf, snap_tolerance)
    G_clean = build_graph_from_gdf(cleaned_gdf, snap_tolerance)

    if G_orig.number_of_nodes() < 2 or G_clean.number_of_nodes() < 2:
        return 0.0

    # Sample random node pairs from original graph
    orig_nodes = list(G_orig.nodes())
    clean_nodes = list(G_clean.nodes())

    if len(orig_nodes) < 2:
        return 0.0

    similarities = []

    for _ in range(min(num_samples, len(orig_nodes) * (len(orig_nodes) - 1) // 2)):
        # Random source and target
        src, tgt = np.random.choice(len(orig_nodes), 2, replace=False)
        src_node = orig_nodes[src]
        tgt_node = orig_nodes[tgt]

        try:
            # Path length in original graph
            orig_length = nx.shortest_path_length(G_orig, src_node, tgt_node, weight='length')
        except nx.NetworkXNoPath:
            continue

        # Find nearest nodes in cleaned graph
        # (simplified: use same node IDs if they exist)
        if src_node in G_clean and tgt_node in G_clean:
            try:
                clean_length = nx.shortest_path_length(G_clean, src_node, tgt_node, weight='length')

                # Compute similarity
                if orig_length > 0:
                    sim = 1 - min(abs(orig_length - clean_length) / orig_length, 1)
                    similarities.append(sim)
            except nx.NetworkXNoPath:
                similarities.append(0)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


# =============================================================================
# Lane-Specific Quality Metrics
# =============================================================================

def compute_lane_width_metrics(
    cleaned_gdf: gpd.GeoDataFrame,
    target_width: float = 3.6  # 12 feet in meters
) -> Tuple[float, float, float]:
    """
    Compute lane width metrics.

    Measures spacing between parallel lines and deviation from target standards.

    Args:
        cleaned_gdf: Cleaned GeoDataFrame
        target_width: Target lane width in meters

    Returns:
        Tuple of (mean_width, std_width, deviation_from_target)
    """
    if len(cleaned_gdf) < 2:
        return 0.0, 0.0, 0.0

    # Build spatial index
    sindex = cleaned_gdf.sindex

    widths = []

    for idx, geom in enumerate(cleaned_gdf.geometry):
        if geom is None or geom.is_empty:
            continue

        # Find nearby geometries
        bbox = geom.buffer(target_width * 2).bounds
        candidates = list(sindex.intersection(bbox))

        for cand_idx in candidates:
            if cand_idx <= idx:  # Avoid duplicates
                continue

            cand_geom = cleaned_gdf.geometry.iloc[cand_idx]
            if cand_geom is None or cand_geom.is_empty:
                continue

            # Check if parallel (simplified check)
            dist = geom.distance(cand_geom)
            if dist < target_width * 2 and dist > 0:
                widths.append(dist)

    if not widths:
        return 0.0, 0.0, 0.0

    mean_width = np.mean(widths)
    std_width = np.std(widths)
    deviation = abs(mean_width - target_width)

    return float(mean_width), float(std_width), float(deviation)


def compute_curvature(line: LineString) -> np.ndarray:
    """
    Compute curvature at each point along a LineString.

    Args:
        line: Input LineString

    Returns:
        Array of curvature values
    """
    coords = np.array(line.coords)

    if len(coords) < 3:
        return np.array([0.0])

    # Compute first and second derivatives
    dx = np.gradient(coords[:, 0])
    dy = np.gradient(coords[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2)

    # Avoid division by zero
    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)

    return curvature


def compute_smoothness_metrics(cleaned_gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    """
    Compute smoothness metrics for the cleaned geometries.

    Measures G2-continuity (curvature continuity) and overall smoothness.

    Args:
        cleaned_gdf: Cleaned GeoDataFrame

    Returns:
        Tuple of (curvature_continuity, smoothness_score)
    """
    if len(cleaned_gdf) == 0:
        return 0.0, 0.0

    curvature_variations = []
    smoothness_scores = []

    for geom in cleaned_gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        if len(list(geom.coords)) < 3:
            continue

        curvature = compute_curvature(geom)

        # Curvature continuity: low variation in curvature
        if len(curvature) > 1:
            curvature_var = np.std(np.diff(curvature))
            curvature_variations.append(curvature_var)

        # Smoothness: inverse of max curvature
        max_curv = np.max(np.abs(curvature))
        if max_curv > 0:
            smoothness_scores.append(1 / (1 + max_curv))
        else:
            smoothness_scores.append(1.0)

    if not curvature_variations:
        return 0.0, 0.0

    # Lower curvature variation = better G2 continuity
    # Normalize to 0-1 scale (1 = perfect continuity)
    mean_var = np.mean(curvature_variations)
    curvature_continuity = 1 / (1 + mean_var)

    smoothness = np.mean(smoothness_scores)

    return float(curvature_continuity), float(smoothness)


# =============================================================================
# Main Metrics Computation
# =============================================================================

def compute_all_metrics(
    original_gdf: gpd.GeoDataFrame,
    cleaned_gdf: gpd.GeoDataFrame,
    target_lane_width: float = 3.6
) -> QualityMetrics:
    """
    Compute all quality metrics for the cleaning results.

    Args:
        original_gdf: Original input GeoDataFrame
        cleaned_gdf: Cleaned output GeoDataFrame
        target_lane_width: Target lane width in meters (default: 3.6m = 12ft)

    Returns:
        QualityMetrics object with all computed metrics
    """
    print("\nComputing Quality Metrics...")

    metrics = QualityMetrics()

    # Summary statistics
    metrics.input_segment_count = len(original_gdf)
    metrics.output_segment_count = len(cleaned_gdf)
    if metrics.input_segment_count > 0:
        metrics.reduction_percent = 100 * (1 - metrics.output_segment_count / metrics.input_segment_count)

    # Geometric Fidelity
    print("  Computing RMSE...")
    metrics.rmse = compute_rmse(original_gdf, cleaned_gdf)

    # Topological Consistency
    print("  Computing Topological F1-Score...")
    metrics.topo_precision, metrics.topo_recall, metrics.topo_f1 = compute_topo_f1(
        original_gdf, cleaned_gdf
    )

    print("  Computing Relative GED...")
    metrics.rel_ged = compute_relative_ged(original_gdf, cleaned_gdf)

    # Navigation Performance
    print("  Computing APLS...")
    metrics.apls = compute_apls(original_gdf, cleaned_gdf)

    # Lane-Specific Quality
    print("  Computing Lane Width Metrics...")
    metrics.lane_width_mean, metrics.lane_width_std, metrics.lane_width_deviation = \
        compute_lane_width_metrics(cleaned_gdf, target_lane_width)

    print("  Computing Smoothness Metrics...")
    metrics.curvature_continuity, metrics.smoothness_score = compute_smoothness_metrics(cleaned_gdf)

    print("  Done!")

    return metrics


def print_metrics_report(metrics: QualityMetrics) -> str:
    """
    Generate a formatted report of the quality metrics.

    Args:
        metrics: QualityMetrics object

    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("QUALITY METRICS REPORT")
    report.append("=" * 60)

    report.append("\n--- Summary ---")
    report.append(f"Input Segments:  {metrics.input_segment_count}")
    report.append(f"Output Segments: {metrics.output_segment_count}")
    report.append(f"Reduction:       {metrics.reduction_percent:.1f}%")

    report.append("\n--- Geometric Fidelity ---")
    report.append(f"RMSE: {metrics.rmse:.4f} meters")

    report.append("\n--- Topological Consistency ---")
    report.append(f"Precision:     {metrics.topo_precision:.4f}")
    report.append(f"Recall:        {metrics.topo_recall:.4f}")
    report.append(f"F1-Score:      {metrics.topo_f1:.4f}")
    report.append(f"Relative GED:  {metrics.rel_ged:.4f} (lower is better)")

    report.append("\n--- Navigation Performance ---")
    report.append(f"APLS: {metrics.apls:.4f} (higher is better)")

    report.append("\n--- Lane-Specific Quality ---")
    report.append(f"Lane Width Mean:      {metrics.lane_width_mean:.2f} meters")
    report.append(f"Lane Width Std:       {metrics.lane_width_std:.2f} meters")
    report.append(f"Lane Width Deviation: {metrics.lane_width_deviation:.2f} meters from target")
    report.append(f"Curvature Continuity: {metrics.curvature_continuity:.4f} (higher is better)")
    report.append(f"Smoothness Score:     {metrics.smoothness_score:.4f} (higher is better)")

    report.append("\n" + "=" * 60)

    return "\n".join(report)
