"""Shared geometric utilities for centerline generation algorithms.

These helpers are algorithm-agnostic and can be reused by any centerline
generation approach (clustering-based, ML-based, geometric, etc.).
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------


def angle_diff_deg(a: float, b: float) -> float:
    """Absolute angular difference in degrees, wrapping around 360."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing from (x1,y1) to (x2,y2) in projected coords.  0=north, CW."""
    return (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# Trace sampling
# ---------------------------------------------------------------------------


def sample_line_projected(
        line_xy: LineString,
        sample_spacing_m: float,
        max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densify a projected LineString into evenly-spaced samples.

    Returns
    -------
    dists : (N,) float32
        Distance along the line at each sample.
    coords : (N, 2) float32
        Projected (x, y) coordinates.
    headings : (N,) float32
        Bearing at each sample (central-difference of neighbours).
    """
    if line_xy.length == 0:
        arr = np.array([], dtype=np.float32)
        return arr, arr.reshape(0, 2), arr

    effective_step = max(sample_spacing_m, line_xy.length / max(max_points, 1))
    n_segments = max(int(math.ceil(line_xy.length / effective_step)), 1)
    dists = np.linspace(0.0, float(line_xy.length), n_segments + 1, dtype=np.float32)

    coords = np.zeros((len(dists), 2), dtype=np.float32)
    for i, d in enumerate(dists):
        p = line_xy.interpolate(float(d))
        coords[i, 0] = float(p.x)
        coords[i, 1] = float(p.y)

    headings = np.zeros(len(dists), dtype=np.float32)
    for i in range(len(dists)):
        i0 = max(i - 1, 0)
        i1 = min(i + 1, len(dists) - 1)
        if i0 == i1:
            headings[i] = 0.0
        else:
            headings[i] = bearing_from_xy(
                coords[i0, 0], coords[i0, 1], coords[i1, 0], coords[i1, 1]
            )
    return dists, coords, headings


# ---------------------------------------------------------------------------
# Altitude interpolation
# ---------------------------------------------------------------------------


def interpolate_altitudes(
        altitudes: List[float],
        line_xy: LineString,
        sample_dists: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a VPD altitude array to sample positions."""
    if not altitudes:
        return np.full(len(sample_dists), np.nan, dtype=np.float32)

    cleaned: List[float] = []
    for a in altitudes:
        try:
            cleaned.append(float(a))
        except Exception:
            continue
    if len(cleaned) < 2:
        return np.full(len(sample_dists), np.nan, dtype=np.float32)

    anchor_dists = np.linspace(
        0.0, float(line_xy.length), len(cleaned), dtype=np.float32
    )
    return np.interp(
        sample_dists, anchor_dists, np.asarray(cleaned, dtype=np.float32)
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Curve smoothing
# ---------------------------------------------------------------------------


def chaikin(coords: np.ndarray, iterations: int) -> np.ndarray:
    """Chaikin corner-cutting curve smoothing."""
    if len(coords) < 3 or iterations <= 0:
        return coords

    out = coords.copy()
    for _ in range(iterations):
        if len(out) < 3:
            break
        next_coords = [out[0]]
        for i in range(len(out) - 1):
            p = out[i]
            q = out[i + 1]
            q1 = 0.75 * p + 0.25 * q
            q2 = 0.25 * p + 0.75 * q
            next_coords.append(q1)
            next_coords.append(q2)
        next_coords.append(out[-1])
        out = np.asarray(next_coords, dtype=np.float64)
    return out


def turn_indices(coords: np.ndarray, turn_deg: float) -> set[int]:
    """Return vertex indices where local heading change exceeds ``turn_deg``."""
    keep: set[int] = set()
    if len(coords) == 0:
        return keep
    keep.add(0)
    keep.add(len(coords) - 1)
    if len(coords) < 3:
        return keep

    threshold = float(max(turn_deg, 0.0))
    for i in range(1, len(coords) - 1):
        h1 = bearing_from_xy(
            float(coords[i - 1, 0]),
            float(coords[i - 1, 1]),
            float(coords[i, 0]),
            float(coords[i, 1]),
        )
        h2 = bearing_from_xy(
            float(coords[i, 0]),
            float(coords[i, 1]),
            float(coords[i + 1, 0]),
            float(coords[i + 1, 1]),
        )
        if angle_diff_deg(h1, h2) >= threshold:
            keep.add(i)
    return keep


def smooth_polyline_preserve_turns(
        coords: np.ndarray,
        *,
        passes: int,
        turn_deg: float = 30.0,
        neighbor_weight: float = 0.25,
) -> np.ndarray:
    """Smooth a polyline while preserving endpoints and sharp turns."""
    out = coords.copy()
    if len(out) <= 2 or passes <= 0:
        return out

    keep_idx = turn_indices(out, turn_deg=turn_deg)
    alpha = float(np.clip(neighbor_weight, 0.0, 0.49))
    center_w = 1.0 - 2.0 * alpha
    for _ in range(passes):
        nxt = out.copy()
        for i in range(1, len(out) - 1):
            if i in keep_idx:
                continue
            nxt[i] = alpha * out[i - 1] + center_w * out[i] + alpha * out[i + 1]
        out = nxt
    return out


# ---------------------------------------------------------------------------
# Graph stitching  (polylines from edge set)
# ---------------------------------------------------------------------------


def stitch_centerline_paths(
        edge_support: Dict[Tuple[int, int], dict],
) -> List[List[int]]:
    """Walk the undirected edge graph to form maximal polyline paths."""
    undirected_adj: Dict[int, set] = defaultdict(set)
    undirected_edges = set()
    for u, v in edge_support:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        undirected_edges.add((a, b))
        undirected_adj[u].add(v)
        undirected_adj[v].add(u)

    visited: set = set()
    paths: List[List[int]] = []

    def walk(start: int, nxt: int) -> List[int]:
        path = [start, nxt]
        visited.add((min(start, nxt), max(start, nxt)))
        prev, cur = start, nxt
        while len(undirected_adj[cur]) == 2:
            candidates = [z for z in undirected_adj[cur] if z != prev]
            if not candidates:
                break
            nn = candidates[0]
            ek = (min(cur, nn), max(cur, nn))
            if ek in visited:
                break
            path.append(nn)
            visited.add(ek)
            prev, cur = cur, nn
        return path

    endpoints = [n for n, neigh in undirected_adj.items() if len(neigh) != 2]
    for s in endpoints:
        for n in list(undirected_adj[s]):
            e = (min(s, n), max(s, n))
            if e in visited:
                continue
            paths.append(walk(s, n))

    # Remaining loops.
    for a, b in list(undirected_edges):
        if (a, b) in visited:
            continue
        loop = walk(a, b)
        paths.append(loop)

    return paths


# ---------------------------------------------------------------------------
# Hop-limited Dijkstra for transitive edge pruning
# ---------------------------------------------------------------------------


def shortest_alternative_with_hop_limit(
        graph: Dict[int, List[Tuple[int, float]]],
        source: int,
        target: int,
        skip_edge: Tuple[int, int],
        max_hops: int,
        max_dist: float,
) -> float:
    """Dijkstra-like search constrained by hop count."""
    best = {(source, 0): 0.0}
    frontier: list[tuple[float, int, int]] = [(0.0, source, 0)]

    while frontier:
        dist, node, hops = heapq.heappop(frontier)
        if dist > max_dist:
            continue
        if node == target and hops > 0:
            return dist
        if hops >= max_hops:
            continue

        for nbr, w in graph.get(node, []):
            if (node, nbr) == skip_edge:
                continue
            nd = dist + w
            key = (nbr, hops + 1)
            if nd < best.get(key, float("inf")):
                best[key] = nd
                heapq.heappush(frontier, (nd, nbr, hops + 1))

    return float("inf")
