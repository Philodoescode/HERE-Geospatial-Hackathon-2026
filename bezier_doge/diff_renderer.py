"""
Module 5: Differentiable Rendering.

Renders a BezierGraph to a 2D image, enabling gradient flow from pixel-level
losses back to the graph's control points, node positions, and edge widths.

Two backends:
1. DiffVG (primary): Uses the DiffVG library for accurate differentiable
   vector graphics rasterization. Best quality but requires installation.
2. PyTorch fallback: Samples points along curves and applies soft Gaussian
   kernels. Fully differentiable via autograd. Less precise but always works.

Reference: DOGE paper, Section 3.2 (DiffAlign).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from .bezier_graph import BezierGraph


# ============================================================================
# DiffVG Backend
# ============================================================================

_DIFFVG_AVAILABLE = False
try:
    import pydiffvg

    _DIFFVG_AVAILABLE = True
except ImportError:
    pass


def is_diffvg_available() -> bool:
    return _DIFFVG_AVAILABLE


def render_diffvg(
    graph: BezierGraph,
    canvas_h: Optional[int] = None,
    canvas_w: Optional[int] = None,
) -> torch.Tensor:
    """
    Render the Bezier graph using DiffVG.

    Each edge becomes a stroked cubic Bezier path.

    Returns
    -------
    rendered : Tensor (H, W) float in [0, 1]
    """
    if not _DIFFVG_AVAILABLE:
        raise RuntimeError("DiffVG is not installed. Use render_pytorch instead.")

    H = canvas_h or graph.canvas_h
    W = canvas_w or graph.canvas_w

    if graph.n_edges == 0:
        return torch.zeros(H, W, device=graph.device)

    shapes = []
    shape_groups = []
    cp = graph.compute_control_points()  # (E, 4, 2)

    for e in range(graph.n_edges):
        # Control points for this curve
        points = cp[e]  # (4, 2)
        # DiffVG expects points as (x, y)
        path = pydiffvg.Path(
            num_control_points=torch.tensor([2]),  # cubic = 2 interior control points
            points=points,
            is_closed=False,
            stroke_width=torch.abs(graph.edge_width[e]) * 2,  # diameter
        )
        shapes.append(path)

        # White stroke, no fill
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([e]),
            fill_color=None,
            stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )
        shape_groups.append(path_group)

    scene_args = pydiffvg.RenderFunction.serialize_scene(W, H, shapes, shape_groups)

    render_fn = pydiffvg.RenderFunction.apply
    img = render_fn(W, H, 2, 2, 0, None, *scene_args)
    # img is (H, W, 4) RGBA
    rendered = img[:, :, 0]  # Take red channel as grayscale

    return rendered.clamp(0.0, 1.0)


# ============================================================================
# PyTorch Fallback Backend
# ============================================================================


def render_pytorch(
    graph: BezierGraph,
    canvas_h: Optional[int] = None,
    canvas_w: Optional[int] = None,
    n_samples: int = 64,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """
    Render the Bezier graph using pure PyTorch differentiable soft rendering.

    For each sampled point on each curve, we add a soft Gaussian contribution
    to the canvas. The result approximates a "stroked" rendering of each curve.

    This is fully differentiable via autograd.

    Parameters
    ----------
    graph : BezierGraph
    canvas_h, canvas_w : int, optional
    n_samples : int
        Number of sample points per curve.
    sigma_scale : float
        Multiplier for the Gaussian sigma (relative to edge width).

    Returns
    -------
    rendered : Tensor (H, W) float in [0, 1]
    """
    H = canvas_h or graph.canvas_h
    W = canvas_w or graph.canvas_w

    if graph.n_edges == 0:
        return torch.zeros(H, W, device=graph.device)

    # Sample curve points: (E, T, 2)
    curve_pts = graph.sample_curves(n_samples=n_samples)
    E = curve_pts.shape[0]

    # Edge widths as sigma: (E,)
    widths = torch.abs(graph.edge_width).clamp(min=0.5)  # (E,)
    sigmas = widths * sigma_scale  # (E,)

    # Create pixel grid
    # Using a chunked approach to avoid OOM on large canvases
    canvas = torch.zeros(H, W, device=graph.device)

    # Process edges in batches to manage memory
    batch_size = max(1, min(E, 32))

    for batch_start in range(0, E, batch_size):
        batch_end = min(batch_start + batch_size, E)
        batch_pts = curve_pts[batch_start:batch_end]  # (B, T, 2)
        batch_sigmas = sigmas[batch_start:batch_end]  # (B,)
        B = batch_pts.shape[0]

        # For each edge in batch, compute its contribution using a localized approach
        for b in range(B):
            pts = batch_pts[b]  # (T, 2) — x, y coordinates
            sig = batch_sigmas[b]  # scalar

            # Determine bounding box of this curve (with padding)
            pad = sig * 3
            x_min = max(0, int(pts[:, 0].min().item() - pad))
            x_max = min(W, int(pts[:, 0].max().item() + pad) + 1)
            y_min = max(0, int(pts[:, 1].min().item() - pad))
            y_max = min(H, int(pts[:, 1].max().item() + pad) + 1)

            if x_max <= x_min or y_max <= y_min:
                continue

            # Create local pixel grid
            local_h = y_max - y_min
            local_w = x_max - x_min

            grid_y = torch.arange(
                y_min, y_max, device=graph.device, dtype=torch.float32
            )
            grid_x = torch.arange(
                x_min, x_max, device=graph.device, dtype=torch.float32
            )
            gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")  # (lH, lW)

            # Distance from each pixel to nearest curve point
            # Reshape for broadcasting: pixels (lH*lW, 1, 2) vs curve pts (1, T, 2)
            pixels = torch.stack([gx, gy], dim=-1).reshape(-1, 1, 2)  # (P, 1, 2)
            pts_b = pts.unsqueeze(0)  # (1, T, 2)

            dist_sq = ((pixels - pts_b) ** 2).sum(dim=-1)  # (P, T)
            min_dist_sq = dist_sq.min(dim=1).values  # (P,)

            # Gaussian kernel
            sigma_sq = sig * sig
            contribution = torch.exp(-min_dist_sq / (2 * sigma_sq))
            contribution = contribution.reshape(local_h, local_w)

            # Accumulate
            canvas[y_min:y_max, x_min:x_max] = torch.max(
                canvas[y_min:y_max, x_min:x_max],
                contribution,
            )

    return canvas.clamp(0.0, 1.0)


def render_edge_pytorch(
    graph: BezierGraph,
    edge_idx: int,
    canvas_h: Optional[int] = None,
    canvas_w: Optional[int] = None,
    n_samples: int = 64,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """
    Render a single edge. Useful for per-edge overlap computation.

    Returns
    -------
    rendered : Tensor (H, W) float in [0, 1]
    """
    H = canvas_h or graph.canvas_h
    W = canvas_w or graph.canvas_w

    idx = torch.tensor([edge_idx], device=graph.device)
    pts = graph.sample_curves(n_samples=n_samples, edge_idx=idx)  # (1, T, 2)
    pts = pts[0]  # (T, 2)

    width = torch.abs(graph.edge_width[edge_idx]).clamp(min=0.5)
    sig = width * sigma_scale

    canvas = torch.zeros(H, W, device=graph.device)

    pad = sig * 3
    x_min = max(0, int(pts[:, 0].min().item() - pad.item()))
    x_max = min(W, int(pts[:, 0].max().item() + pad.item()) + 1)
    y_min = max(0, int(pts[:, 1].min().item() - pad.item()))
    y_max = min(H, int(pts[:, 1].max().item() + pad.item()) + 1)

    if x_max <= x_min or y_max <= y_min:
        return canvas

    grid_y = torch.arange(y_min, y_max, device=graph.device, dtype=torch.float32)
    grid_x = torch.arange(x_min, x_max, device=graph.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    pixels = torch.stack([gx, gy], dim=-1).reshape(-1, 1, 2)
    pts_b = pts.unsqueeze(0)

    dist_sq = ((pixels - pts_b) ** 2).sum(dim=-1)
    min_dist_sq = dist_sq.min(dim=1).values

    sigma_sq = sig * sig
    contribution = torch.exp(-min_dist_sq / (2 * sigma_sq))
    contribution = contribution.reshape(y_max - y_min, x_max - x_min)

    canvas[y_min:y_max, x_min:x_max] = contribution

    return canvas.clamp(0.0, 1.0)


# ============================================================================
# Unified Render API
# ============================================================================


def render(
    graph: BezierGraph,
    canvas_h: Optional[int] = None,
    canvas_w: Optional[int] = None,
    backend: str = "auto",
    n_samples: int = 64,
) -> torch.Tensor:
    """
    Render the Bezier graph to a 2D canvas.

    Parameters
    ----------
    graph : BezierGraph
    canvas_h, canvas_w : int, optional
    backend : str
        "diffvg", "pytorch", or "auto" (tries DiffVG first).
    n_samples : int
        Samples per curve (PyTorch backend).

    Returns
    -------
    rendered : Tensor (H, W) float in [0, 1]
    """
    if backend == "auto":
        backend = "diffvg" if _DIFFVG_AVAILABLE else "pytorch"

    if backend == "diffvg":
        return render_diffvg(graph, canvas_h, canvas_w)
    else:
        return render_pytorch(graph, canvas_h, canvas_w, n_samples=n_samples)
