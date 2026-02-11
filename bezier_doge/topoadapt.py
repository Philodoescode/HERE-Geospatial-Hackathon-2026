"""
Module 7: TopoAdapt — Discrete Topology Refinement.

Implements the five discrete operators from the DOGE paper (Section 3.3)
that dynamically refine the graph's topology between DiffAlign steps.

Operators:
1. Road Addition     — seed new edges in uncovered road regions
2. Node Merging      — merge nearby nodes into one
3. T-Junction        — snap nodes to nearby edges
4. Collinear Merging — simplify degree-2 nodes on straight paths
5. Edge Pruning      — remove degenerate edges and isolated nodes

Reference: DOGE paper, Section 3.3 and Figure 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label as nd_label

from .bezier_graph import BezierGraph
from .diff_renderer import render


@dataclass
class TopoAdaptConfig:
    """Configuration for topology operators."""

    # Proximity threshold for merging (in pixels, ~= meters at 1m/px)
    merge_distance: float = 4.0
    # Road addition
    seg_threshold: float = 0.5  # Threshold for target segmentation
    render_threshold: float = 0.3  # Below this = uncovered
    max_new_roads: int = 10  # Max new edges per iteration
    new_road_length: float = 40.0  # Initial length of new road segments (pixels)
    new_road_width: float = 3.0  # Initial width of new road segments
    # Collinear merging angle threshold (degrees) — merge if angle > this
    collinear_threshold_deg: float = 160.0
    # Pruning
    min_edge_length: float = 5.0  # Minimum edge length in pixels
    min_edge_width: float = 1.0  # Minimum edge width in pixels


def road_addition(
    graph: BezierGraph,
    target: torch.Tensor,
    cfg: TopoAdaptConfig = TopoAdaptConfig(),
) -> int:
    """
    Add new road edges in uncovered regions (Eq 9-10).

    Finds regions where the target mask is high but the rendered graph
    has low coverage, then seeds short Bezier edges there.

    Returns
    -------
    n_added : int
        Number of edges added.
    """
    H, W = target.shape
    device = graph.device

    # Render current graph
    with torch.no_grad():
        rendered = render(graph, H, W, backend="pytorch", n_samples=32)

    # Compute unfit map (Eq 9): high target AND low coverage
    target_np = target.detach().cpu().numpy()
    rendered_np = rendered.detach().cpu().numpy()

    unfit = (
        (target_np > cfg.seg_threshold) & (rendered_np < cfg.render_threshold)
    ).astype(np.float32)

    # Connected components
    labeled, n_components = nd_label(unfit)
    if n_components == 0:
        return 0

    # Find centroids of the k largest components
    component_sizes = []
    component_centers = []
    for comp_id in range(1, n_components + 1):
        mask = labeled == comp_id
        size = mask.sum()
        if size < 10:  # Skip tiny regions
            continue
        ys, xs = np.where(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        component_sizes.append(size)
        component_centers.append((cx, cy))

    if not component_centers:
        return 0

    # Sort by size, take top k
    sorted_indices = np.argsort(component_sizes)[::-1][: cfg.max_new_roads]

    n_added = 0
    for idx in sorted_indices:
        cx, cy = component_centers[idx]

        # Check if too close to existing nodes
        if graph.n_nodes > 0:
            positions_np = graph.node_positions.detach().cpu().numpy()
            dists = np.sqrt(
                (positions_np[:, 0] - cx) ** 2 + (positions_np[:, 1] - cy) ** 2
            )
            if dists.min() < cfg.merge_distance * 2:
                continue

        # Create a short edge with random orientation
        angle = np.random.uniform(0, np.pi)
        half_len = cfg.new_road_length / 2
        dx = half_len * np.cos(angle)
        dy = half_len * np.sin(angle)

        # Clamp endpoints to canvas
        x0 = float(np.clip(cx - dx, 1, W - 2))
        y0 = float(np.clip(cy - dy, 1, H - 2))
        x1 = float(np.clip(cx + dx, 1, W - 2))
        y1 = float(np.clip(cy + dy, 1, H - 2))

        # Add two nodes
        new_positions = torch.tensor(
            [[x0, y0], [x1, y1]], device=device, dtype=torch.float32
        )
        node_ids = graph.add_nodes(new_positions)

        # Add edge between them
        graph.add_edges(
            start_nodes=node_ids[0:1],
            end_nodes=node_ids[1:2],
            widths=torch.tensor([cfg.new_road_width], device=device),
        )
        n_added += 1

    return n_added


def node_merging(
    graph: BezierGraph,
    cfg: TopoAdaptConfig = TopoAdaptConfig(),
) -> int:
    """
    Merge pairs of nodes closer than merge_distance.

    The merged node is placed at the midpoint and inherits all incident edges.

    Returns
    -------
    n_merged : int
        Number of merge operations performed.
    """
    if graph.n_nodes < 2:
        return 0

    positions = graph.node_positions.detach().cpu().numpy()
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cfg.merge_distance)

    if not pairs:
        return 0

    # Build union-find to handle transitive merges
    parent = list(range(graph.n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    # Build mapping: old node -> canonical node
    canonical = {}
    for i in range(graph.n_nodes):
        canonical[i] = find(i)

    # Check if any merging actually happens
    unique_roots = set(canonical.values())
    if len(unique_roots) == graph.n_nodes:
        return 0

    # Compute merged positions (average of group members)
    root_positions = {}
    root_members = {}
    for i in range(graph.n_nodes):
        r = canonical[i]
        if r not in root_positions:
            root_positions[r] = []
            root_members[r] = []
        root_positions[r].append(positions[i])
        root_members[r].append(i)

    # Build new node list
    root_to_new_idx = {}
    new_positions = []
    for root in sorted(root_positions.keys()):
        root_to_new_idx[root] = len(new_positions)
        avg_pos = np.mean(root_positions[root], axis=0)
        new_positions.append(avg_pos)

    new_positions = np.array(new_positions, dtype=np.float32)

    # Build old-to-new index mapping
    old_to_new = {}
    for i in range(graph.n_nodes):
        old_to_new[i] = root_to_new_idx[canonical[i]]

    # Update graph
    device = graph.device
    new_pos_tensor = torch.tensor(new_positions, device=device, dtype=torch.float32)

    # Remap edges
    if graph.n_edges > 0:
        old_edges = graph.edge_indices.detach().cpu().numpy()
        new_edges = np.array(
            [[old_to_new[int(u)], old_to_new[int(v)]] for u, v in old_edges],
            dtype=np.int64,
        )
        # Remove self-loops
        valid = new_edges[:, 0] != new_edges[:, 1]
        new_edges = new_edges[valid]

        graph.edge_indices = torch.tensor(new_edges, device=device, dtype=torch.long)
        # Also filter edge parameters
        valid_torch = torch.tensor(valid, device=device, dtype=torch.bool)
        graph.edge_alpha = torch.nn.Parameter(graph.edge_alpha.data[valid_torch])
        graph.edge_offset = torch.nn.Parameter(graph.edge_offset.data[valid_torch])
        graph.edge_width = torch.nn.Parameter(graph.edge_width.data[valid_torch])
        graph._n_edges = int(valid.sum())

    # Update nodes
    graph.node_positions = torch.nn.Parameter(new_pos_tensor)
    graph._n_nodes = len(new_positions)

    n_merged = graph.n_nodes  # Rough count
    return len(pairs)


def t_junction_creation(
    graph: BezierGraph,
    cfg: TopoAdaptConfig = TopoAdaptConfig(),
    n_samples: int = 32,
) -> int:
    """
    Create T-junctions by snapping isolated nodes to nearby edges.

    If a node is close to an edge but not connected to it, split
    the edge at the nearest point and merge the resulting vertex
    with the node.

    Returns
    -------
    n_created : int
    """
    if graph.n_nodes < 1 or graph.n_edges < 1:
        return 0

    positions = graph.node_positions.detach().cpu().numpy()
    deg = graph.node_degree().cpu().numpy()

    # Only consider degree-1 nodes (dangling endpoints)
    candidate_nodes = np.where(deg == 1)[0]
    if len(candidate_nodes) == 0:
        return 0

    # Sample points along all edges
    with torch.no_grad():
        all_pts = graph.sample_curves(n_samples=n_samples)  # (E, T, 2)
    all_pts_np = all_pts.cpu().numpy()

    n_created = 0
    edges_to_add = []  # (node1_pos, node2_pos, split_info)
    edges_to_remove = []

    for node_idx in candidate_nodes:
        node_pos = positions[node_idx]

        # Find the nearest edge (that this node is NOT already connected to)
        connected_edges = set(e for e, _ in graph.edges_at_node(node_idx))

        best_dist = float("inf")
        best_edge = -1
        best_t_idx = -1

        for e in range(graph.n_edges):
            if e in connected_edges:
                continue

            edge_pts = all_pts_np[e]  # (T, 2)
            dists = np.sqrt(np.sum((edge_pts - node_pos) ** 2, axis=1))
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            if min_dist < best_dist:
                best_dist = min_dist
                best_edge = e
                best_t_idx = min_idx

        if best_edge < 0 or best_dist > cfg.merge_distance:
            continue

        # Snap: connect this node to the nearest point on the edge
        # For simplicity, we add a new edge from this node to the nearest
        # endpoint of the target edge
        edge_start = int(graph.edge_indices[best_edge, 0].item())
        edge_end = int(graph.edge_indices[best_edge, 1].item())

        # Choose the closer endpoint
        d_start = np.linalg.norm(positions[edge_start] - node_pos)
        d_end = np.linalg.norm(positions[edge_end] - node_pos)

        target_node = edge_start if d_start <= d_end else edge_end

        # Don't create if already connected
        if target_node == node_idx:
            continue

        # Check existing edges to avoid duplicates
        already_exists = False
        for e in range(graph.n_edges):
            u, v = graph.edge_indices[e].tolist()
            if (u == node_idx and v == target_node) or (
                u == target_node and v == node_idx
            ):
                already_exists = True
                break

        if already_exists:
            continue

        # Add edge from node to nearest endpoint
        graph.add_edges(
            start_nodes=torch.tensor([node_idx], device=graph.device),
            end_nodes=torch.tensor([target_node], device=graph.device),
            widths=torch.tensor(
                [graph.edge_width[best_edge].item()], device=graph.device
            ),
        )
        n_created += 1

    return n_created


def collinear_edge_merging(
    graph: BezierGraph,
    cfg: TopoAdaptConfig = TopoAdaptConfig(),
) -> int:
    """
    Merge collinear edges at degree-2 nodes.

    If a degree-2 node lies on a nearly straight path (angle between
    incident edge tangents exceeds threshold), remove the node and
    replace its two edges with a single refitted Bezier curve.

    Returns
    -------
    n_merged : int
    """
    if graph.n_nodes == 0 or graph.n_edges < 2:
        return 0

    deg = graph.node_degree()
    deg2_nodes = torch.where(deg == 2)[0].tolist()

    if not deg2_nodes:
        return 0

    threshold_rad = cfg.collinear_threshold_deg * np.pi / 180.0

    # Get tangent vectors
    with torch.no_grad():
        start_tangents, end_tangents = graph.get_edge_tangents()

    nodes_to_remove = []
    edges_to_remove = set()
    new_edges_info = []

    for node_idx in deg2_nodes:
        edges = graph.edges_at_node(node_idx)
        if len(edges) != 2:
            continue

        e1_idx, e1_is_start = edges[0]
        e2_idx, e2_is_start = edges[1]

        # Skip if either edge is already marked for removal
        if e1_idx in edges_to_remove or e2_idx in edges_to_remove:
            continue

        # Get tangent vectors at this node
        if e1_is_start:
            t1 = start_tangents[e1_idx]
        else:
            t1 = -end_tangents[e1_idx]

        if e2_is_start:
            t2 = start_tangents[e2_idx]
        else:
            t2 = -end_tangents[e2_idx]

        # Compute angle
        t1_norm = t1 / (torch.norm(t1) + 1e-8)
        t2_norm = t2 / (torch.norm(t2) + 1e-8)
        cos_angle = torch.dot(t1_norm, t2_norm).clamp(-1, 1)
        angle = torch.acos(cos_angle).item()

        # If angle > threshold (nearly straight), merge
        if angle > threshold_rad:
            # Determine the other endpoints
            if e1_is_start:
                other1 = int(graph.edge_indices[e1_idx, 1].item())
            else:
                other1 = int(graph.edge_indices[e1_idx, 0].item())

            if e2_is_start:
                other2 = int(graph.edge_indices[e2_idx, 1].item())
            else:
                other2 = int(graph.edge_indices[e2_idx, 0].item())

            if other1 == other2:
                continue  # Would create self-loop

            edges_to_remove.add(e1_idx)
            edges_to_remove.add(e2_idx)
            nodes_to_remove.append(node_idx)

            # Average width
            avg_width = (
                graph.edge_width[e1_idx].item() + graph.edge_width[e2_idx].item()
            ) / 2
            new_edges_info.append((other1, other2, avg_width))

    if not edges_to_remove:
        return 0

    # Remove old edges
    remove_mask = torch.zeros(graph.n_edges, dtype=torch.bool, device=graph.device)
    for e in edges_to_remove:
        remove_mask[e] = True
    graph.remove_edges(remove_mask)

    # Add merged edges
    for start, end, width in new_edges_info:
        # Node indices may have shifted — use original positions to look up
        graph.add_edges(
            start_nodes=torch.tensor([start], device=graph.device),
            end_nodes=torch.tensor([end], device=graph.device),
            widths=torch.tensor([width], device=graph.device),
        )

    # Clean up isolated nodes
    graph.remove_isolated_nodes()

    return len(new_edges_info)


def edge_pruning(
    graph: BezierGraph,
    cfg: TopoAdaptConfig = TopoAdaptConfig(),
) -> int:
    """
    Prune invalid edges: too short, too thin, or geometrically degenerate.

    Returns
    -------
    n_pruned : int
    """
    if graph.n_edges == 0:
        return 0

    remove_mask = torch.zeros(graph.n_edges, dtype=torch.bool, device=graph.device)

    with torch.no_grad():
        # Check edge lengths (chord length as proxy)
        indices = graph.edge_indices
        P0 = graph.node_positions[indices[:, 0]]
        P3 = graph.node_positions[indices[:, 1]]
        chord_lengths = torch.norm(P3 - P0, dim=1)

        # Mark too-short edges
        remove_mask |= chord_lengths < cfg.min_edge_length

        # Mark too-thin edges
        remove_mask |= graph.edge_width < cfg.min_edge_width

    n_pruned = int(remove_mask.sum().item())
    if n_pruned > 0:
        graph.remove_edges(remove_mask)
        graph.remove_isolated_nodes()

    return n_pruned


def topoadapt(
    graph: BezierGraph,
    target: torch.Tensor,
    cfg: TopoAdaptConfig = TopoAdaptConfig(),
    iteration: int = 0,
) -> dict:
    """
    Run all topology operators in sequence.

    Parameters
    ----------
    graph : BezierGraph
    target : Tensor (H, W)
    cfg : TopoAdaptConfig
    iteration : int
        Current optimization iteration (controls which operators run).

    Returns
    -------
    stats : dict
        Count of operations performed.
    """
    stats = {}

    # 1. Prune invalid edges first
    stats["pruned"] = edge_pruning(graph, cfg)

    # 2. Node merging (every iteration)
    stats["merged"] = node_merging(graph, cfg)

    # 3. Road addition (mainly early iterations)
    if iteration < 200 and iteration % 5 == 0:
        stats["added"] = road_addition(graph, target, cfg)
    else:
        stats["added"] = 0

    # 4. T-junction creation (after some initial optimization)
    if iteration >= 10 and iteration % 10 == 0:
        stats["t_junctions"] = t_junction_creation(graph, cfg)
    else:
        stats["t_junctions"] = 0

    # 5. Collinear edge merging (periodically)
    if iteration >= 20 and iteration % 15 == 0:
        stats["collinear_merged"] = collinear_edge_merging(graph, cfg)
    else:
        stats["collinear_merged"] = 0

    return stats
