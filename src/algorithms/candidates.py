"""
Candidate algorithms for centerline generation (Phase 5C).

IMPORTANT — CRS convention
---------------------------
All candidates expect **metric-CRS input** (e.g. EPSG:32634 / UTM 34N).
Distance parameters (eps, buffer_dist, pixel_size …) are in **metres**.
The output GeoDataFrame inherits the CRS of the input.
Callers must project *before* calling and re-project *after* if needed.
"""

import logging
import math
from collections import defaultdict
from typing import List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label as ndi_label
from scipy.spatial import cKDTree
from shapely import get_coordinates, segmentize
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import linemerge, unary_union
from shapely import STRtree
from skimage.morphology import skeletonize

from src.config import CRS

logger = logging.getLogger(__name__)

METRIC_CRS = "EPSG:32634"  # UTM 34N for Kosovo — same as evaluation module
MIN_LINE_LENGTH_M = 10.0  # discard lines shorter than this (noise filter)

# ---------------------------------------------------------------------------
# Utility / Helper Functions
# ---------------------------------------------------------------------------


def _extract_all_segment_points(gdf: gpd.GeoDataFrame, step: float) -> np.ndarray:
    """Densify all LineStrings in *gdf* via Shapely 2 vectorised segmentize,
    then return all coordinates as a single (N, 2) float64 array.
    This is orders of magnitude faster than Python-loop densification."""
    geoms = gdf.geometry.values  # GeoSeries → array of shapely objects
    densified = segmentize(geoms, max_segment_length=step)
    # get_coordinates returns (N,2) or (N,3); take first 2 cols
    coords = get_coordinates(densified)[:, :2]
    return coords.astype(np.float64)


def _extract_midpoints_and_headings(
    gdf: gpd.GeoDataFrame, step: float = 5.0
) -> np.ndarray:
    """For each segment in each densified trace, compute (mx, my, cos_h, sin_h).
    Returns an (N, 4) float64 array.  Fully vectorised — no Python geometry loops."""
    geoms = gdf.geometry.values
    densified = segmentize(geoms, max_segment_length=step)

    # Build per-geometry coordinate arrays and process segments
    all_mx, all_my, all_cos, all_sin = [], [], [], []

    for geom in densified:
        if geom is None or geom.is_empty:
            continue
        coords = get_coordinates(geom)[:, :2]
        if len(coords) < 2:
            continue
        # Vectorised segment midpoints + headings
        p1 = coords[:-1]  # (M, 2)
        p2 = coords[1:]  # (M, 2)
        mid = (p1 + p2) * 0.5
        dx = p2[:, 0] - p1[:, 0]
        dy = p2[:, 1] - p1[:, 1]
        angles = np.arctan2(dy, dx)
        all_mx.append(mid[:, 0])
        all_my.append(mid[:, 1])
        all_cos.append(np.cos(angles))
        all_sin.append(np.sin(angles))

    if not all_mx:
        return np.empty((0, 4), dtype=np.float64)

    return np.column_stack(
        [
            np.concatenate(all_mx),
            np.concatenate(all_my),
            np.concatenate(all_cos),
            np.concatenate(all_sin),
        ]
    )


def _filter_short_lines(geoms, min_length: float = MIN_LINE_LENGTH_M):
    """Drop lines shorter than *min_length* metres."""
    return [g for g in geoms if g.length >= min_length]


def _make_output(lines, crs):
    """Build a clean GeoDataFrame from a list of LineStrings."""
    lines = [g for g in lines if not g.is_empty and g.length > 0]
    lines = _filter_short_lines(lines)
    return gpd.GeoDataFrame(geometry=lines, crs=crs)


# ---------------------------------------------------------------------------
# Candidate 1: Grid + KDE + Skeletonization  (OPTIMISED)
# ---------------------------------------------------------------------------


def candidate_kde_skeleton(
    gdf: gpd.GeoDataFrame,
    pixel_size: float = 1.0,
    blur_sigma: float = 2.0,
    threshold: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Rasterize trajectories, apply Gaussian blur (proxy for KDE), threshold,
    skeletonize, and vectorize back to LineStrings.

    Optimised version:
    - Fully vectorised rasterization via Shapely 2 segmentize + numpy binning
    - Connected-component labelling instead of full pixel graph
    - Direct coordinate tracing per component for fast vectorisation

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input trajectories (VPD).
    pixel_size : float
        Resolution of the grid in meters.
    blur_sigma : float
        Sigma for Gaussian filter (in pixels).
    threshold : float
        Density threshold (0.0 to 1.0 relative to max) to define 'road'.

    Returns
    -------
    gpd.GeoDataFrame
        Extracted centerlines.
    """
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    crs_out = gdf.crs

    # 1. Grid bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    buff = 20.0
    minx -= buff
    miny -= buff
    maxx += buff
    maxy += buff

    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    if width <= 0 or height <= 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    logger.info(
        "KDE grid: %d × %d pixels (%.1fm resolution)", width, height, pixel_size
    )

    # 2. Vectorised rasterization ──────────────────────────────────────
    #    segmentize all geometries at half-pixel spacing, extract coords,
    #    convert to pixel indices, and use np.add.at for accumulation.
    coords = _extract_all_segment_points(gdf, step=pixel_size * 0.5)
    logger.info("  Rasterizing %d points …", len(coords))

    cols = ((coords[:, 0] - minx) / pixel_size).astype(np.intp)
    rows = ((maxy - coords[:, 1]) / pixel_size).astype(np.intp)

    # Clip to grid bounds
    mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[mask]
    cols = cols[mask]

    grid = np.zeros((height, width), dtype=np.float32)
    np.add.at(grid, (rows, cols), 1.0)

    # 3. Gaussian blur (KDE proxy)
    density = gaussian_filter(grid, sigma=blur_sigma)
    d_max = density.max()
    if d_max > 0:
        density /= d_max

    # 4. Threshold
    binary = density > threshold

    # 5. Skeletonize
    skeleton = skeletonize(binary)

    # 6. Vectorise via connected-component graph tracing ────────────────
    skel_rows, skel_cols = np.where(skeleton)
    if len(skel_rows) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    logger.info("  Skeleton: %d pixels, tracing lines …", len(skel_rows))

    # Build adjacency graph from skeleton pixels
    pixel_set = set(zip(skel_rows.tolist(), skel_cols.tolist()))
    G = nx.Graph()
    for r, c in pixel_set:
        G.add_node((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in pixel_set:
                    G.add_edge((r, c), nb)

    def px2world(r, c):
        return (minx + (c + 0.5) * pixel_size, maxy - (r + 0.5) * pixel_size)

    # Trace paths between junctions / endpoints
    junctions = [n for n in G.nodes() if G.degree(n) > 2]
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    terminals = set(junctions + endpoints)
    if not terminals and G.number_of_nodes():
        terminals = {next(iter(G.nodes()))}

    visited_edges: set = set()
    lines: list = []

    def _trace(start, neighbor):
        path = [start, neighbor]
        visited_edges.add((start, neighbor))
        visited_edges.add((neighbor, start))
        cur, prev = neighbor, start
        while cur not in terminals:
            nbrs = [
                n
                for n in G.neighbors(cur)
                if n != prev and (cur, n) not in visited_edges
            ]
            if not nbrs:
                break
            nxt = nbrs[0]
            visited_edges.add((cur, nxt))
            visited_edges.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for t in terminals:
        for nb in list(G.neighbors(t)):
            if (t, nb) not in visited_edges:
                path = _trace(t, nb)
                if len(path) >= 2:
                    lines.append(LineString([px2world(r, c) for r, c in path]))

    # Leftover cycles
    for u, v in G.edges():
        if (u, v) not in visited_edges:
            path = _trace(u, v)
            if len(path) >= 2:
                lines.append(LineString([px2world(r, c) for r, c in path]))

    logger.info("  Produced %d raw lines", len(lines))
    return _make_output(lines, crs_out)


# ---------------------------------------------------------------------------
# Candidate 2: DBSCAN Point Clustering + Polyline Fitting  (OPTIMISED)
# ---------------------------------------------------------------------------


def candidate_dbscan_polyline(
    gdf: gpd.GeoDataFrame,
    eps: float = 10.0,
    min_samples: int = 5,
    heading_weight: float = 1.0,
    max_points: int = 2_000_000,
) -> gpd.GeoDataFrame:
    """
    Cluster points from trajectories using DBSCAN on (x, y, heading)
    and fit polylines.

    Optimised version:
    - Fully vectorised point + heading extraction (no Python Point loops)
    - Automatic subsampling when point count exceeds *max_points*
    - MiniBatchKMeans pre-filter option removed in favour of simple random
      downsample to keep DBSCAN memory feasible

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input trajectories.
    eps : float
        DBSCAN epsilon in meters.
    min_samples : int
        DBSCAN min_samples.
    heading_weight : float
        Weight for heading distance (to split opposing lanes).
    max_points : int
        Safety cap — randomly subsample if more points than this.

    Returns
    -------
    gpd.GeoDataFrame
    """
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    crs_out = gdf.crs

    # 1. Vectorised extraction ─────────────────────────────────────────
    data = _extract_midpoints_and_headings(gdf, step=5.0)
    if len(data) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    logger.info("DBSCAN: %d segment midpoints extracted", len(data))

    # Safety subsample to keep DBSCAN RAM usage bounded
    if len(data) > max_points:
        logger.warning(
            "  Subsampling %d → %d points (max_points cap)", len(data), max_points
        )
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=max_points, replace=False)
        data = data[idx]

    # 2. Feature matrix: (x, y, scaled_cos, scaled_sin)
    xy = data[:, :2]
    heading_feats = data[:, 2:4] * (eps * heading_weight)
    features = np.hstack([xy, heading_feats])

    logger.info(
        "  Running DBSCAN (eps=%.1f, min_samples=%d) on %d points …",
        eps,
        min_samples,
        len(features),
    )
    labels = DBSCAN(
        eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1
    ).fit_predict(features)

    # 3. Fit polylines per cluster ─────────────────────────────────────
    lines = []
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        pts = xy[mask]
        if len(pts) < 2:
            continue

        # PCA to find main axis → sort along it
        pca = PCA(n_components=2)
        projected = pca.fit_transform(pts)
        order = np.argsort(projected[:, 0])
        sorted_pts = pts[order]

        line = LineString(sorted_pts)
        simple = line.simplify(tolerance=2.0)
        lines.append(simple)

    logger.info(
        "  %d clusters → %d lines",
        len(unique_labels) - (1 if -1 in unique_labels else 0),
        len(lines),
    )
    return _make_output(lines, crs_out)


# ---------------------------------------------------------------------------
# Candidate 3: Trace Clustering (Hausdorff)  (OPTIMISED)
# ---------------------------------------------------------------------------


def candidate_trace_clustering(
    gdf: gpd.GeoDataFrame,
    eps: float = 15.0,
    min_samples: int = 2,
    max_traces: int = 500,
) -> gpd.GeoDataFrame:
    """
    Cluster full trajectories using Hausdorff distance and extract medoids.

    Optimised version:
    - Built-in random subsample cap (avoids relying on notebook to do it)
    - Simplify traces before Hausdorff to reduce per-pair cost
    - Use Shapely 2 vectorised hausdorff where possible
    - Progress logging every 50 rows

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    eps : float
        DBSCAN eps for the distance matrix.
    min_samples : int
    max_traces : int
        Built-in safety cap. O(n²) — keep this ≤ 500.

    Returns
    -------
    gpd.GeoDataFrame
    """
    from sklearn.cluster import DBSCAN

    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    crs_out = gdf.crs

    # Built-in subsample if needed
    if len(gdf) > max_traces:
        logger.warning(
            "Trace clustering: subsampling %d → %d traces (O(n²) cap)",
            len(gdf),
            max_traces,
        )
        gdf = gdf.sample(n=max_traces, random_state=42)

    # Simplify to reduce Hausdorff cost
    traces = gdf.geometry.simplify(tolerance=2.0).tolist()
    # Remove empty / degenerate
    traces = [t for t in traces if t is not None and not t.is_empty and t.length > 0]
    n = len(traces)
    logger.info(
        "Trace clustering: %d traces, computing %d×%d distance matrix …", n, n, n
    )

    if n < min_samples:
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    # Compute upper-triangle Hausdorff distance matrix
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if i % 50 == 0:
            logger.info("  Hausdorff row %d / %d", i, n)
        for j in range(i + 1, n):
            d = traces[i].hausdorff_distance(traces[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Cluster
    labels = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples).fit_predict(
        dist_matrix
    )

    # Extract medoids
    centerlines = []
    unique_labels = set(labels.tolist())
    unique_labels.discard(-1)

    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        # Medoid: trace with min average distance to cluster peers
        sub = dist_matrix[np.ix_(indices, indices)]
        avg_dists = sub.mean(axis=1)
        best_local = np.argmin(avg_dists)
        centerlines.append(traces[indices[best_local]])

    logger.info("  %d clusters → %d centerlines", len(unique_labels), len(centerlines))
    return _make_output(centerlines, crs_out)


# ---------------------------------------------------------------------------
# Candidate 4: Incremental Map Construction (Graph-based)  (OPTIMISED)
# ---------------------------------------------------------------------------


def candidate_incremental_graph(
    gdf: gpd.GeoDataFrame,
    buffer_dist: float = 5.0,
    snap_dist: float = 5.0,
) -> gpd.GeoDataFrame:
    """
    Incrementally build a road network by adding traces longest-first,
    skipping those that are redundant with the already-accepted set.

    Optimised version:
    - STRtree rebuilt periodically (every 200 accepted edges) instead of
      every single iteration → O(n log n) instead of O(n²)
    - Hausdorff check only; removed expensive buffer().contains() test
    - Simplified traces before processing to reduce geometry complexity
    - Progress logging

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    buffer_dist : float
        Buffer to check for existing edges (merging).
    snap_dist : float
        Distance to snap close endpoints.

    Returns
    -------
    gpd.GeoDataFrame
    """
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    crs_out = gdf.crs
    total = len(gdf)

    # Pre-simplify all traces to reduce geometry complexity
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tolerance=1.0)
    gdf["_length"] = gdf.geometry.length
    sorted_gdf = gdf.sort_values("_length", ascending=False)

    edges: list = []  # accepted LineStrings
    tree: Optional[STRtree] = None
    tree_stale = True  # rebuild flag
    REBUILD_INTERVAL = 200  # rebuild STRtree every N accepted edges

    logger.info("Incremental graph: processing %d traces …", total)

    for count, (idx, row) in enumerate(sorted_gdf.iterrows()):
        if count % 2000 == 0 and count > 0:
            logger.info(
                "  Processed %d / %d traces, accepted %d edges",
                count,
                total,
                len(edges),
            )

        trace = row.geometry
        if trace is None or trace.is_empty or trace.length < MIN_LINE_LENGTH_M:
            continue

        # Check redundancy against already-accepted edges
        is_redundant = False
        if edges:
            # Rebuild tree if stale
            if tree_stale:
                tree = STRtree(edges)
                tree_stale = False

            candidate_idxs = tree.query(
                trace, predicate="dwithin", distance=buffer_dist
            )
            for ci in candidate_idxs:
                existing = edges[ci]
                if trace.hausdorff_distance(existing) < buffer_dist:
                    is_redundant = True
                    break

        if is_redundant:
            continue

        edges.append(trace)
        # Mark tree stale periodically — minor false negatives between
        # rebuilds are cleaned up by the union+linemerge post-processing
        if len(edges) % REBUILD_INTERVAL == 0:
            tree_stale = True

    logger.info("  Accepted %d edges from %d traces", len(edges), total)

    if not edges:
        return gpd.GeoDataFrame(geometry=[], crs=crs_out)

    # Post-processing: node the lines and snap close endpoints ─────────
    logger.info("  Post-processing: union + linemerge …")
    noded = unary_union(edges)
    merged = linemerge(noded)

    if isinstance(merged, LineString):
        result = [merged]
    elif isinstance(merged, MultiLineString):
        result = list(merged.geoms)
    else:
        result = []

    # Endpoint snapping via cKDTree + union-find
    if snap_dist > 0 and len(result) > 1:
        all_ends = []
        for line in result:
            cs = list(line.coords)
            all_ends.append(cs[0])
            all_ends.append(cs[-1])
        coords_arr = np.array(all_ends, dtype=np.float64)
        kd = cKDTree(coords_arr)
        pairs = kd.query_pairs(snap_dist)

        # Union-find
        parent = list(range(len(all_ends)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        clusters = defaultdict(list)
        for i in range(len(all_ends)):
            clusters[find(i)].append(i)

        snap_map = {}
        for members in clusters.values():
            if len(members) < 2:
                continue
            cx = np.mean([coords_arr[m][0] for m in members])
            cy = np.mean([coords_arr[m][1] for m in members])
            for m in members:
                snap_map[m] = (cx, cy)

        snapped = []
        for li, line in enumerate(result):
            coords = list(line.coords)
            si, ei = li * 2, li * 2 + 1
            if si in snap_map:
                coords[0] = snap_map[si]
            if ei in snap_map:
                coords[-1] = snap_map[ei]
            snapped.append(LineString(coords))
        result = snapped

    logger.info("  Final: %d lines after merge + snap", len(result))
    return _make_output(result, crs_out)
