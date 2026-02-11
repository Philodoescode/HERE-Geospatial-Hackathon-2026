"""
Module 6: DiffAlign — Differentiable Geometric Optimization.

Implements the 5-term composite loss function from the DOGE paper (Section 3.2)
that drives continuous geometric optimization of the Bezier Graph.

Loss terms:
1. Coverage Loss (L_cover)  — data fidelity: rendered graph should match target mask
2. Overlap Loss (L_overlap) — penalize improper intersections between edges
3. G1 Continuity Loss (L_G1) — tangent alignment at degree-2 nodes
4. Offset Loss (L_offset)   — prevent excessive control point offsets
5. Spacing Loss (L_spacing) — encourage equidistant control point placement

Reference: DOGE paper, Equations 3-8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .bezier_graph import BezierGraph
from .diff_renderer import render, render_edge_pytorch


@dataclass
class DiffAlignConfig:
    """Loss weight configuration from the paper (Section 4.1)."""

    lambda_cover: float = 1.0
    lambda_overlap: float = 0.3
    lambda_g1: float = 0.012
    lambda_offset: float = 0.006
    lambda_spacing: float = 0.006
    # G1 continuity threshold angle (degrees)
    g1_threshold_deg: float = 160.0
    # Offset loss threshold ratio
    offset_tau: float = 0.5
    # Ideal alpha positions for spacing loss
    alpha_hat_0: float = 1.0 / 3.0
    alpha_hat_1: float = 2.0 / 3.0
    # Rendering
    n_samples: int = 64
    render_backend: str = "auto"


def coverage_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Coverage Loss (Eq 4): pixel-wise L2 between rendered graph and target mask.

    L_cover = || render(G) - S ||_2^2
    """
    return F.mse_loss(rendered, target)


def overlap_loss(
    graph: BezierGraph,
    canvas_h: int,
    canvas_w: int,
    n_samples: int = 64,
    max_edges_for_overlap: int = 100,
) -> torch.Tensor:
    """
    Overlap Loss (Eq 5): penalize pixels covered by more than one edge.

    Renders each edge individually, sums them, and penalizes the excess above 1.0.

    L_overlap = (1/N) * || max(0, sum_k R(e_k) - 1) ||_1
    """
    N = graph.n_edges
    if N == 0:
        return torch.tensor(0.0, device=graph.device)

    # For efficiency, limit the number of edges we compute overlap for
    if N > max_edges_for_overlap:
        # Sample a subset
        perm = torch.randperm(N, device=graph.device)[:max_edges_for_overlap]
        edge_list = perm.tolist()
    else:
        edge_list = list(range(N))

    # Sum individual edge renderings
    sum_render = torch.zeros(canvas_h, canvas_w, device=graph.device)
    for e_idx in edge_list:
        edge_render = render_edge_pytorch(
            graph, e_idx, canvas_h, canvas_w, n_samples=n_samples
        )
        sum_render = sum_render + edge_render

    # Penalize overlap (pixels where sum > 1)
    excess = torch.clamp(sum_render - 1.0, min=0.0)
    return excess.mean()


def g1_continuity_loss(
    graph: BezierGraph,
    threshold_deg: float = 160.0,
) -> torch.Tensor:
    """
    G1 Continuity Loss (Eq 6): encourage tangent alignment at degree-2 nodes.

    For degree-2 nodes, the incoming and outgoing edge tangents should be
    nearly collinear (angle close to 180 degrees). The loss penalizes
    deviations only when the angle is below the threshold (to preserve
    legitimate turns).

    L_G1 = (1/N) * sum_{deg-2 nodes} (1 - cos(theta)) * I(theta < T_G1)
    """
    if graph.n_nodes == 0 or graph.n_edges < 2:
        return torch.tensor(0.0, device=graph.device)

    deg = graph.node_degree()
    deg2_mask = deg == 2
    deg2_indices = torch.where(deg2_mask)[0]

    if len(deg2_indices) == 0:
        return torch.tensor(0.0, device=graph.device)

    threshold_rad = threshold_deg * torch.pi / 180.0
    loss_sum = torch.tensor(0.0, device=graph.device)
    count = 0

    start_tangents, end_tangents = graph.get_edge_tangents()

    for node_idx in deg2_indices.tolist():
        edges = graph.edges_at_node(node_idx)
        if len(edges) != 2:
            continue

        # Get tangent vectors at this node
        e1_idx, e1_is_start = edges[0]
        e2_idx, e2_is_start = edges[1]

        # The tangent AT the node:
        # If this node is the start of edge e, tangent = start_tangent (pointing away)
        # If this node is the end of edge e, tangent = -end_tangent (pointing toward node, negate for outgoing)
        if e1_is_start:
            t1 = start_tangents[e1_idx]
        else:
            t1 = -end_tangents[e1_idx]

        if e2_is_start:
            t2 = start_tangents[e2_idx]
        else:
            t2 = -end_tangents[e2_idx]

        # Compute angle between tangent vectors
        t1_norm = t1 / (torch.norm(t1) + 1e-8)
        t2_norm = t2 / (torch.norm(t2) + 1e-8)
        cos_angle = torch.dot(t1_norm, t2_norm).clamp(-1, 1)
        angle = torch.acos(cos_angle)

        # Penalize if angle < threshold (i.e., not straight enough)
        if angle.item() < threshold_rad:
            loss_sum = loss_sum + (1.0 - cos_angle)
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=graph.device)

    return loss_sum / count


def offset_loss(graph: BezierGraph, tau_d: float = 0.5) -> torch.Tensor:
    """
    Offset Loss (Eq 7): penalize excessive perpendicular offsets.

    L_offset = (1/N) * sum_e max(0, exp(|d_k|/L_k - tau_d) - 1)

    Where L_k is the chord length and tau_d is the threshold ratio.
    """
    if graph.n_edges == 0:
        return torch.tensor(0.0, device=graph.device)

    # Compute chord lengths
    indices = graph.edge_indices
    P0 = graph.node_positions[indices[:, 0]]  # (E, 2)
    P3 = graph.node_positions[indices[:, 1]]  # (E, 2)
    L = torch.norm(P3 - P0, dim=1).clamp(min=1e-6)  # (E,)

    # Offset ratios
    d = graph.edge_offset  # (E, 2)
    d_abs = torch.abs(d)  # (E, 2)

    # Ratio: |d| / L
    ratios = d_abs / L.unsqueeze(1)  # (E, 2)

    # Exponential penalty
    penalty = torch.clamp(torch.exp(ratios - tau_d) - 1.0, min=0.0)

    return penalty.mean()


def spacing_loss(
    graph: BezierGraph,
    alpha_hat_0: float = 1.0 / 3.0,
    alpha_hat_1: float = 2.0 / 3.0,
) -> torch.Tensor:
    """
    Spacing Loss (Eq 8): encourage equidistant control point placement.

    L_spacing = (1/N) * sum_e ((alpha_0 - 1/3)^2 + (alpha_1 - 2/3)^2)
    """
    if graph.n_edges == 0:
        return torch.tensor(0.0, device=graph.device)

    alpha = graph.edge_alpha  # (E, 2)
    loss = (alpha[:, 0] - alpha_hat_0) ** 2 + (alpha[:, 1] - alpha_hat_1) ** 2
    return loss.mean()


def compute_total_loss(
    graph: BezierGraph,
    target: torch.Tensor,
    rendered: torch.Tensor,
    cfg: DiffAlignConfig = DiffAlignConfig(),
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the composite loss (Eq 3).

    L_total = lambda_cover * L_cover
            + lambda_overlap * L_overlap
            + lambda_G1 * L_G1
            + lambda_offset * L_offset
            + lambda_spacing * L_spacing

    Parameters
    ----------
    graph : BezierGraph
    target : Tensor (H, W)
        Target segmentation mask.
    rendered : Tensor (H, W)
        Differentiably rendered graph.
    cfg : DiffAlignConfig

    Returns
    -------
    total_loss : Tensor (scalar)
    loss_dict : dict
        Individual loss values for logging.
    """
    H, W = target.shape

    # 1. Coverage loss (always computed)
    l_cover = coverage_loss(rendered, target)

    # 2. Overlap loss (expensive, skip if few edges)
    if graph.n_edges > 1 and cfg.lambda_overlap > 0:
        l_overlap = overlap_loss(graph, H, W, n_samples=cfg.n_samples)
    else:
        l_overlap = torch.tensor(0.0, device=graph.device)

    # 3. G1 continuity loss
    if cfg.lambda_g1 > 0:
        l_g1 = g1_continuity_loss(graph, threshold_deg=cfg.g1_threshold_deg)
    else:
        l_g1 = torch.tensor(0.0, device=graph.device)

    # 4. Offset loss
    if cfg.lambda_offset > 0:
        l_offset = offset_loss(graph, tau_d=cfg.offset_tau)
    else:
        l_offset = torch.tensor(0.0, device=graph.device)

    # 5. Spacing loss
    if cfg.lambda_spacing > 0:
        l_spacing = spacing_loss(graph, cfg.alpha_hat_0, cfg.alpha_hat_1)
    else:
        l_spacing = torch.tensor(0.0, device=graph.device)

    # Composite loss
    total = (
        cfg.lambda_cover * l_cover
        + cfg.lambda_overlap * l_overlap
        + cfg.lambda_g1 * l_g1
        + cfg.lambda_offset * l_offset
        + cfg.lambda_spacing * l_spacing
    )

    loss_dict = {
        "total": float(total.item()),
        "cover": float(l_cover.item()),
        "overlap": float(l_overlap.item()),
        "g1": float(l_g1.item()),
        "offset": float(l_offset.item()),
        "spacing": float(l_spacing.item()),
    }

    return total, loss_dict


def diffalign_step(
    graph: BezierGraph,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg: DiffAlignConfig = DiffAlignConfig(),
) -> Dict[str, float]:
    """
    Perform one gradient step of geometric optimization.

    Parameters
    ----------
    graph : BezierGraph
    target : Tensor (H, W)
        Target segmentation mask.
    optimizer : torch.optim.Optimizer
    cfg : DiffAlignConfig

    Returns
    -------
    loss_dict : dict
        Loss breakdown for logging.
    """
    optimizer.zero_grad()

    # Render the graph
    rendered = render(
        graph,
        canvas_h=target.shape[0],
        canvas_w=target.shape[1],
        backend=cfg.render_backend,
        n_samples=cfg.n_samples,
    )

    # Compute losses
    total_loss, loss_dict = compute_total_loss(graph, target, rendered, cfg)

    # Backprop
    total_loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(graph.parameters(), max_norm=10.0)

    optimizer.step()

    # Clamp alpha to [0, 1] and width to positive
    with torch.no_grad():
        graph.edge_alpha.data.clamp_(0.0, 1.0)
        graph.edge_width.data.clamp_(min=0.5)

        # Clamp node positions to canvas
        graph.node_positions.data[:, 0].clamp_(0, graph.canvas_w - 1)
        graph.node_positions.data[:, 1].clamp_(0, graph.canvas_h - 1)

    return loss_dict
