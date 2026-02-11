"""
Centerline Evaluation Metrics (Phase 5A).

Provides both reference-comparison metrics (generated vs Nav Streets)
and intrinsic quality metrics (smoothness, topology, redundancy).
All distance-based metrics operate in EPSG:32634 (UTM 34N, metres).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points, unary_union

logger = logging.getLogger(__name__)

METRIC_CRS = "EPSG:32634"  # UTM 34N — metric CRS for Kosovo


# ──────────────────────────────────────────────────────────────────────
# Data class to hold all metrics in one place
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CenterlineMetrics:
    """Container for all centerline evaluation metrics."""

    # Reference comparison (vs Nav Streets)
    nav_recovery_pct: float = 0.0       # % of Nav Streets length within buffer of generated
    nav_precision_pct: float = 0.0      # % of generated length within buffer of Nav Streets
    new_road_km: float = 0.0            # generated length NOT near Nav Streets (km)
    mean_hausdorff_m: float = 0.0       # mean per-link Hausdorff distance (metres)
    median_hausdorff_m: float = 0.0     # median per-link Hausdorff distance (metres)

    # Intrinsic quality
    mean_angular_deflection_deg: float = 0.0   # smoothness: lower = smoother
    median_angular_deflection_deg: float = 0.0
    total_length_km: float = 0.0
    num_lines: int = 0

    # Topology
    num_components: int = 0             # connected components (fewer = better)
    num_dangling_ends: int = 0          # degree-1 nodes (should be few, mostly at bbox edge)
    num_pseudo_nodes: int = 0           # degree-2 nodes (can be simplified)
    num_intersections: int = 0          # degree >= 3 nodes

    # Redundancy
    redundant_length_km: float = 0.0    # near-duplicate centreline length
    redundancy_pct: float = 0.0         # redundant / total %


# ──────────────────────────────────────────────────────────────────────
# Helper: project to metric CRS if needed
# ──────────────────────────────────────────────────────────────────────

def _to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in METRIC_CRS."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() != METRIC_CRS:
        return gdf.to_crs(METRIC_CRS)
    return gdf


# ──────────────────────────────────────────────────────────────────────
# 1. Reference Comparison Metrics
# ──────────────────────────────────────────────────────────────────────

def nav_recovery_and_precision(
    generated: gpd.GeoDataFrame,
    nav: gpd.GeoDataFrame,
    buffer_m: float = 15.0,
) -> dict:
    """
    Compute coverage metrics between generated centerlines and Nav Streets.

    Uses a line-by-line approach to avoid memory issues with large
    unary_union + buffer operations on complex geometry collections.

    Parameters
    ----------
    generated : GeoDataFrame of generated centerlines (LineStrings)
    nav : GeoDataFrame of Nav Streets reference (LineStrings)
    buffer_m : float  — buffer distance in metres

    Returns
    -------
    dict with keys:
        nav_recovery_pct  — % of Nav Streets length covered by generated
        nav_precision_pct — % of generated length that overlaps Nav Streets
        new_road_km       — generated length outside Nav buffer (km)
    """
    gen_proj = _to_metric(generated)
    nav_proj = _to_metric(nav)

    # Build Nav buffer (Nav Streets is usually small enough for unary_union)
    nav_union = nav_proj.geometry.unary_union
    nav_buffer = nav_union.buffer(buffer_m)

    # --- Precision: how much generated length is near Nav ---
    gen_total = 0.0
    gen_covered = 0.0
    for geom in gen_proj.geometry:
        if geom is None or geom.is_empty:
            continue
        length = geom.length
        gen_total += length
        try:
            clipped = geom.intersection(nav_buffer)
            gen_covered += clipped.length
        except Exception:
            pass

    precision = (gen_covered / gen_total * 100) if gen_total > 0 else 0.0
    new_road_length = gen_total - gen_covered

    # --- Recovery: how much Nav length is near generated ---
    # Build generated buffer using per-line buffers merged via unary_union
    # Process in chunks to avoid memory blow-up
    CHUNK = 500
    gen_buffers = []
    for start in range(0, len(gen_proj), CHUNK):
        chunk = gen_proj.iloc[start:start + CHUNK]
        chunk_buf = chunk.geometry.buffer(buffer_m).unary_union
        gen_buffers.append(chunk_buf)

    gen_buffer_union = unary_union(gen_buffers)

    nav_total = nav_union.length
    try:
        nav_in_gen = nav_union.intersection(gen_buffer_union)
        recovery = (nav_in_gen.length / nav_total * 100) if nav_total > 0 else 0.0
    except Exception:
        # Fallback: per-link recovery
        nav_covered = 0.0
        for geom in nav_proj.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                clipped = geom.intersection(gen_buffer_union)
                nav_covered += clipped.length
            except Exception:
                pass
        recovery = (nav_covered / nav_total * 100) if nav_total > 0 else 0.0

    return {
        "nav_recovery_pct": round(recovery, 2),
        "nav_precision_pct": round(precision, 2),
        "new_road_km": round(max(new_road_length, 0) / 1000.0, 3),
    }


def per_link_hausdorff(
    generated: gpd.GeoDataFrame,
    nav: gpd.GeoDataFrame,
    buffer_m: float = 30.0,
) -> dict:
    """
    For each Nav Street link that has a nearby generated centerline,
    compute the Hausdorff distance between them.

    Uses spatial index for efficient neighbour lookup instead of a
    massive unary_union.

    Returns dict with mean and median Hausdorff in metres.
    """
    gen_proj = _to_metric(generated)
    nav_proj = _to_metric(nav)

    # Build spatial index on generated lines
    gen_sindex = gen_proj.sindex
    hausdorff_distances = []

    for _, row in nav_proj.iterrows():
        nav_line = row.geometry
        if nav_line is None or nav_line.is_empty:
            continue

        # Spatial index query: find generated lines within buffer_m of this nav link
        nav_buf = nav_line.buffer(buffer_m)
        candidate_idxs = list(gen_sindex.intersection(nav_buf.bounds))
        if not candidate_idxs:
            continue

        # Collect nearby generated geometry
        nearby_lines = gen_proj.geometry.iloc[candidate_idxs]
        nearby_lines = nearby_lines[nearby_lines.intersects(nav_buf)]
        if nearby_lines.empty:
            continue

        nearby_union = nearby_lines.unary_union
        hd = nav_line.hausdorff_distance(nearby_union)
        hausdorff_distances.append(hd)

    if hausdorff_distances:
        return {
            "mean_hausdorff_m": round(float(np.mean(hausdorff_distances)), 2),
            "median_hausdorff_m": round(float(np.median(hausdorff_distances)), 2),
            "n_links_matched": len(hausdorff_distances),
        }
    return {"mean_hausdorff_m": 0.0, "median_hausdorff_m": 0.0, "n_links_matched": 0}


# ──────────────────────────────────────────────────────────────────────
# 2. Intrinsic Quality: Smoothness
# ──────────────────────────────────────────────────────────────────────

def _angular_deflections(line: LineString) -> list[float]:
    """
    Compute angular deflection (in degrees) at every interior vertex.
    A perfectly straight line has deflection 0 everywhere.
    """
    coords = list(line.coords)
    if len(coords) < 3:
        return []

    deflections = []
    for i in range(1, len(coords) - 1):
        x0, y0 = coords[i - 1][:2]
        x1, y1 = coords[i][:2]
        x2, y2 = coords[i + 1][:2]

        # Vectors
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1

        dot = v1x * v2x + v1y * v2y
        mag1 = math.sqrt(v1x**2 + v1y**2)
        mag2 = math.sqrt(v2x**2 + v2y**2)

        if mag1 == 0 or mag2 == 0:
            continue

        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_angle))
        deflection = 180.0 - angle   # 0 = straight, 180 = U-turn
        deflections.append(deflection)

    return deflections


def smoothness_metrics(generated: gpd.GeoDataFrame) -> dict:
    """
    Compute mean and median angular deflection across all generated centerlines.
    Lower values = smoother lines.
    """
    gen_proj = _to_metric(generated)

    all_deflections = []
    for geom in gen_proj.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            for part in geom.geoms:
                all_deflections.extend(_angular_deflections(part))
        elif isinstance(geom, LineString):
            all_deflections.extend(_angular_deflections(geom))

    if all_deflections:
        return {
            "mean_angular_deflection_deg": round(float(np.mean(all_deflections)), 3),
            "median_angular_deflection_deg": round(float(np.median(all_deflections)), 3),
        }
    return {"mean_angular_deflection_deg": 0.0, "median_angular_deflection_deg": 0.0}


# ──────────────────────────────────────────────────────────────────────
# 3. Topology Metrics (graph-based)
# ──────────────────────────────────────────────────────────────────────

def _build_graph(gdf: gpd.GeoDataFrame, snap_tolerance: float = 1.0) -> nx.Graph:
    """
    Build a NetworkX graph from LineStrings.

    Nodes are rounded endpoint coordinates (snapped to tolerance).
    Edges carry the geometry and length.
    """
    G = nx.Graph()

    def _snap_coord(coord):
        """Round coordinate to snap_tolerance grid."""
        return (
            round(coord[0] / snap_tolerance) * snap_tolerance,
            round(coord[1] / snap_tolerance) * snap_tolerance,
        )

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lines = []
        if isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        elif isinstance(geom, LineString):
            lines = [geom]

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            start = _snap_coord(coords[0])
            end = _snap_coord(coords[-1])

            if start == end:
                continue  # Skip zero-length loops

            G.add_edge(start, end, geometry=line, length=line.length)

    return G


def topology_metrics(generated: gpd.GeoDataFrame, snap_tolerance: float = 1.0) -> dict:
    """
    Analyse network topology of generated centerlines.

    Parameters
    ----------
    generated : GeoDataFrame of LineStrings
    snap_tolerance : float, metres — nodes closer than this are merged

    Returns
    -------
    dict with topology statistics
    """
    gen_proj = _to_metric(generated)
    G = _build_graph(gen_proj, snap_tolerance)

    if G.number_of_nodes() == 0:
        return {
            "num_components": 0,
            "num_dangling_ends": 0,
            "num_pseudo_nodes": 0,
            "num_intersections": 0,
        }

    degrees = dict(G.degree())
    n_components = nx.number_connected_components(G)
    dangling = sum(1 for d in degrees.values() if d == 1)
    pseudo = sum(1 for d in degrees.values() if d == 2)
    intersections = sum(1 for d in degrees.values() if d >= 3)

    return {
        "num_components": n_components,
        "num_dangling_ends": dangling,
        "num_pseudo_nodes": pseudo,
        "num_intersections": intersections,
    }


# ──────────────────────────────────────────────────────────────────────
# 4. Redundancy Detection
# ──────────────────────────────────────────────────────────────────────

def redundancy_metrics(
    generated: gpd.GeoDataFrame,
    buffer_m: float = 5.0,
) -> dict:
    """
    Detect near-duplicate centerlines (lines within buffer_m of each other).

    For each line, check how much of it overlaps with the union of all
    OTHER lines buffered. High overlap = redundancy.

    This is O(n²)-ish so we use spatial indexing via unary_union.
    """
    gen_proj = _to_metric(generated)

    total_length = gen_proj.geometry.length.sum()
    if total_length == 0 or len(gen_proj) < 2:
        return {"redundant_length_km": 0.0, "redundancy_pct": 0.0}

    # Build union of all lines EXCEPT each one in turn — too expensive.
    # Instead: for each line, buffer it and measure overlap with full union
    # minus itself. Approximate by using the full union (self-overlap is fine
    # since a line covers itself 100% — we compare to the BUFFERED others).

    all_union = gen_proj.geometry.unary_union
    redundant_length = 0.0

    for idx, row in gen_proj.iterrows():
        line = row.geometry
        if line is None or line.is_empty:
            continue

        # Buffer this line and see how much of the OTHER lines cover the same area
        buf = line.buffer(buffer_m)
        # Intersection of the union with this buffer gives nearby geometry
        nearby = all_union.intersection(buf)
        # Subtract this line's own length
        overlap_length = nearby.length - line.length
        if overlap_length > 0:
            redundant_length += min(overlap_length, line.length)

    # Each redundant segment is counted from both overlapping lines,
    # so halve the total
    redundant_length /= 2.0

    return {
        "redundant_length_km": round(redundant_length / 1000.0, 3),
        "redundancy_pct": round(redundant_length / total_length * 100, 2) if total_length > 0 else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# 5. Master Evaluation Function
# ──────────────────────────────────────────────────────────────────────

def evaluate_centerlines(
    generated: gpd.GeoDataFrame,
    nav: gpd.GeoDataFrame,
    buffer_m: float = 15.0,
    hausdorff_buffer_m: float = 30.0,
    snap_tolerance: float = 1.0,
    redundancy_buffer_m: float = 5.0,
    skip_redundancy: bool = False,
) -> CenterlineMetrics:
    """
    Run all evaluation metrics on generated centerlines vs Nav Streets.

    Parameters
    ----------
    generated : GeoDataFrame of generated centerlines
    nav : GeoDataFrame of Nav Streets reference
    buffer_m : float — buffer for recovery/precision (metres)
    hausdorff_buffer_m : float — search radius for per-link Hausdorff
    snap_tolerance : float — node snapping for topology graph (metres)
    redundancy_buffer_m : float — buffer for near-duplicate detection
    skip_redundancy : bool — skip redundancy (expensive for large inputs)

    Returns
    -------
    CenterlineMetrics dataclass with all scores
    """
    logger.info("Evaluating centerlines: %d generated vs %d Nav Streets",
                len(generated), len(nav))

    gen_proj = _to_metric(generated)
    total_km = gen_proj.geometry.length.sum() / 1000.0

    # 1. Reference comparison
    logger.info("  Computing recovery & precision (buffer=%dm)...", buffer_m)
    ref = nav_recovery_and_precision(generated, nav, buffer_m)

    logger.info("  Computing per-link Hausdorff distances...")
    hd = per_link_hausdorff(generated, nav, hausdorff_buffer_m)

    # 2. Smoothness
    logger.info("  Computing smoothness metrics...")
    sm = smoothness_metrics(generated)

    # 3. Topology
    logger.info("  Computing topology metrics...")
    topo = topology_metrics(generated, snap_tolerance)

    # 4. Redundancy
    if skip_redundancy:
        red = {"redundant_length_km": 0.0, "redundancy_pct": 0.0}
        logger.info("  Redundancy: skipped.")
    else:
        logger.info("  Computing redundancy (buffer=%dm)...", redundancy_buffer_m)
        red = redundancy_metrics(generated, redundancy_buffer_m)

    metrics = CenterlineMetrics(
        # Reference
        nav_recovery_pct=ref["nav_recovery_pct"],
        nav_precision_pct=ref["nav_precision_pct"],
        new_road_km=ref["new_road_km"],
        mean_hausdorff_m=hd["mean_hausdorff_m"],
        median_hausdorff_m=hd["median_hausdorff_m"],
        # Intrinsic
        mean_angular_deflection_deg=sm["mean_angular_deflection_deg"],
        median_angular_deflection_deg=sm["median_angular_deflection_deg"],
        total_length_km=round(total_km, 3),
        num_lines=len(generated),
        # Topology
        num_components=topo["num_components"],
        num_dangling_ends=topo["num_dangling_ends"],
        num_pseudo_nodes=topo["num_pseudo_nodes"],
        num_intersections=topo["num_intersections"],
        # Redundancy
        redundant_length_km=red["redundant_length_km"],
        redundancy_pct=red["redundancy_pct"],
    )

    logger.info("  Evaluation complete.")
    return metrics


# ──────────────────────────────────────────────────────────────────────
# 6. FAST Precision / Recall (spatial-index only, no geometry intersection)
# ──────────────────────────────────────────────────────────────────────

def quick_precision_recall(
    generated: gpd.GeoDataFrame,
    nav: gpd.GeoDataFrame,
    buffer_m: float = 15.0,
) -> dict:
    """
    Fast precision & recall using Shapely 2.x STRtree.query(dwithin).

    Instead of computing expensive intersection geometries, this checks
    per-line whether it is within buffer_m of ANY line in the other set.
    A matched line's full length counts as covered.

    ~100x faster than the intersection-based approach.
    """
    from shapely import STRtree

    gen_proj = _to_metric(generated)
    nav_proj = _to_metric(nav)

    gen_geoms = gen_proj.geometry.values
    nav_geoms = nav_proj.geometry.values
    gen_lengths = gen_proj.geometry.length.values
    nav_lengths = nav_proj.geometry.length.values

    gen_total = gen_lengths.sum()
    nav_total = nav_lengths.sum()

    # --- Precision: % of generated length that is near nav ---
    nav_tree = STRtree(nav_geoms)
    hit_gen, _ = nav_tree.query(gen_geoms, predicate="dwithin", distance=buffer_m)
    covered_gen_idx = np.unique(hit_gen)
    precision_length = gen_lengths[covered_gen_idx].sum() if len(covered_gen_idx) > 0 else 0.0
    precision = (precision_length / gen_total * 100) if gen_total > 0 else 0.0

    # --- Recovery: % of nav length that is near generated ---
    gen_tree = STRtree(gen_geoms)
    hit_nav, _ = gen_tree.query(nav_geoms, predicate="dwithin", distance=buffer_m)
    covered_nav_idx = np.unique(hit_nav)
    recovery_length = nav_lengths[covered_nav_idx].sum() if len(covered_nav_idx) > 0 else 0.0
    recovery = (recovery_length / nav_total * 100) if nav_total > 0 else 0.0

    return {
        "nav_recovery_pct": round(recovery, 2),
        "nav_precision_pct": round(precision, 2),
        "new_road_km": round(max(gen_total - precision_length, 0) / 1000.0, 3),
        "total_length_km": round(gen_total / 1000.0, 3),
        "num_lines": len(generated),
    }


def print_quick_metrics(d: dict, label: str = "Centerlines") -> None:
    """Pretty-print quick metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {label}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<35} {'Value':>12}")
    print(f"  {'-' * 35} {'-' * 12}")
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>12.2f}")
        else:
            print(f"  {k:<35} {v:>12}")
    print(f"{'=' * 60}\n")


def metrics_to_dict(m: CenterlineMetrics) -> dict:
    """Convert CenterlineMetrics to a flat dict (for DataFrames / display)."""
    from dataclasses import asdict
    return asdict(m)


def print_metrics(m: CenterlineMetrics, label: str = "Centerlines") -> None:
    """Pretty-print metrics to console."""
    d = metrics_to_dict(m)
    print(f"\n{'=' * 60}")
    print(f"  Evaluation: {label}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<40} {'Value':>12}")
    print(f"  {'-' * 40} {'-' * 12}")
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:>12.3f}")
        else:
            print(f"  {k:<40} {v:>12}")
    print(f"{'=' * 60}\n")
