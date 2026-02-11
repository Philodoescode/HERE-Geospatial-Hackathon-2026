"""
Module 4: Bezier Graph Data Structure.

Implements the Bezier Graph G = (V, E) from the DOGE paper, where:
- Nodes V have optimizable 2D positions
- Edges E are cubic Bezier curves connecting pairs of nodes

All geometric parameters are stored as PyTorch tensors to enable
gradient-based optimization via DiffAlign.

Reference: DOGE paper, Section 3.1, Equations 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class BezierGraph(nn.Module):
    """
    A parametric road network graph using cubic Bezier curves.

    Each edge is a cubic Bezier curve C(t) = sum_{r=0}^{3} B(3,r) (1-t)^{3-r} t^r P_{k,r}
    where the four control points are deterministically constructed from:
      - P0 = position of start node
      - P3 = position of end node
      - P1, P2 = reparameterized via alpha (projection along chord) and d (perpendicular offset)

    This reparameterization (Eq 2a-2d) regularizes the curve shape.
    """

    def __init__(self, canvas_h: int, canvas_w: int):
        super().__init__()
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w

        # Node positions: (N, 2) in pixel coordinates
        self.node_positions = nn.Parameter(torch.zeros(0, 2))

        # Edge connectivity: (E, 2) int tensor (start_node, end_node) — NOT a parameter
        self.register_buffer("edge_indices", torch.zeros(0, 2, dtype=torch.long))

        # Per-edge curve parameters
        # alpha: (E, 2) — projection parameter for P1, P2 along the chord [0, 1]
        self.edge_alpha = nn.Parameter(torch.zeros(0, 2))
        # offset: (E, 2) — perpendicular offset distance for P1, P2
        self.edge_offset = nn.Parameter(torch.zeros(0, 2))
        # width: (E,) — road half-width in pixels
        self.edge_width = nn.Parameter(torch.zeros(0))

        # Track counts
        self._n_nodes = 0
        self._n_edges = 0

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def n_edges(self) -> int:
        return self._n_edges

    @property
    def device(self) -> torch.device:
        return self.node_positions.device

    def add_nodes(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Add nodes to the graph.

        Parameters
        ----------
        positions : Tensor (K, 2)
            Pixel coordinates of new nodes.

        Returns
        -------
        node_ids : Tensor (K,)
            Indices of the newly added nodes.
        """
        K = positions.shape[0]
        start_idx = self._n_nodes

        if self._n_nodes == 0:
            self.node_positions = nn.Parameter(positions.clone().float())
        else:
            new_pos = torch.cat([self.node_positions.data, positions.float()], dim=0)
            self.node_positions = nn.Parameter(new_pos)

        self._n_nodes += K
        return torch.arange(start_idx, start_idx + K, device=self.device)

    def add_edges(
        self,
        start_nodes: torch.Tensor,
        end_nodes: torch.Tensor,
        alphas: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        widths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add edges to the graph.

        Parameters
        ----------
        start_nodes, end_nodes : Tensor (K,)
            Node indices for each edge's start and end.
        alphas : Tensor (K, 2), optional
            Projection parameters. Defaults to [1/3, 2/3].
        offsets : Tensor (K, 2), optional
            Perpendicular offsets. Defaults to 0.
        widths : Tensor (K,), optional
            Road half-widths. Defaults to 3.0 pixels.

        Returns
        -------
        edge_ids : Tensor (K,)
            Indices of the newly added edges.
        """
        K = start_nodes.shape[0]
        start_idx = self._n_edges

        # Default parameters
        if alphas is None:
            alphas = torch.zeros(K, 2, device=self.device)
            alphas[:, 0] = 1.0 / 3.0
            alphas[:, 1] = 2.0 / 3.0
        if offsets is None:
            offsets = torch.zeros(K, 2, device=self.device)
        if widths is None:
            widths = torch.full((K,), 3.0, device=self.device)

        new_indices = torch.stack([start_nodes.long(), end_nodes.long()], dim=1)

        if self._n_edges == 0:
            self.edge_indices = new_indices
            self.edge_alpha = nn.Parameter(alphas.clone().float())
            self.edge_offset = nn.Parameter(offsets.clone().float())
            self.edge_width = nn.Parameter(widths.clone().float())
        else:
            self.edge_indices = torch.cat([self.edge_indices, new_indices], dim=0)
            self.edge_alpha = nn.Parameter(
                torch.cat([self.edge_alpha.data, alphas.float()], dim=0)
            )
            self.edge_offset = nn.Parameter(
                torch.cat([self.edge_offset.data, offsets.float()], dim=0)
            )
            self.edge_width = nn.Parameter(
                torch.cat([self.edge_width.data, widths.float()], dim=0)
            )

        self._n_edges += K
        return torch.arange(start_idx, start_idx + K, device=self.device)

    def compute_control_points(
        self, edge_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the four control points for each edge.

        Implements Equations 2a-2d from the paper.

        Parameters
        ----------
        edge_idx : Tensor, optional
            Indices of edges to compute. Defaults to all edges.

        Returns
        -------
        control_points : Tensor (E, 4, 2)
            Four control points [P0, P1, P2, P3] per edge.
        """
        if edge_idx is None:
            edge_idx = torch.arange(self._n_edges, device=self.device)

        if len(edge_idx) == 0:
            return torch.zeros(0, 4, 2, device=self.device)

        indices = self.edge_indices[edge_idx]  # (E, 2)
        start_idx = indices[:, 0]
        end_idx = indices[:, 1]

        P0 = self.node_positions[start_idx]  # (E, 2)
        P3 = self.node_positions[end_idx]  # (E, 2)

        alpha = torch.clamp(self.edge_alpha[edge_idx], 0.0, 1.0)  # (E, 2)
        offset = self.edge_offset[edge_idx]  # (E, 2)

        # Chord vector and normal
        chord = P3 - P0  # (E, 2)
        chord_len = torch.norm(chord, dim=1, keepdim=True).clamp(min=1e-6)  # (E, 1)
        chord_dir = chord / chord_len  # (E, 2) normalized chord direction

        # Perpendicular normal (rotate 90 degrees: (dx, dy) -> (-dy, dx))
        normal = torch.stack([-chord_dir[:, 1], chord_dir[:, 0]], dim=1)  # (E, 2)

        # Eq 2c: P1 = lerp(P0, P3, alpha_0) + d_0 * normal
        alpha_0 = alpha[:, 0:1]  # (E, 1)
        d_0 = offset[:, 0:1]  # (E, 1)
        P1 = (1 - alpha_0) * P0 + alpha_0 * P3 + d_0 * normal

        # Eq 2d: P2 = lerp(P0, P3, alpha_1) + d_1 * normal
        alpha_1 = alpha[:, 1:2]  # (E, 1)
        d_1 = offset[:, 1:2]  # (E, 1)
        P2 = (1 - alpha_1) * P0 + alpha_1 * P3 + d_1 * normal

        # Stack: (E, 4, 2)
        return torch.stack([P0, P1, P2, P3], dim=1)

    def sample_curves(
        self,
        n_samples: int = 50,
        edge_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample points along Bezier curves.

        Evaluates the cubic Bezier formula (Eq 1) at uniform t values.

        Parameters
        ----------
        n_samples : int
            Number of sample points per curve.
        edge_idx : Tensor, optional
            Which edges to sample. Defaults to all.

        Returns
        -------
        points : Tensor (E, n_samples, 2)
            Sampled curve points.
        """
        cp = self.compute_control_points(edge_idx)  # (E, 4, 2)
        if cp.shape[0] == 0:
            return torch.zeros(0, n_samples, 2, device=self.device)

        t = torch.linspace(0, 1, n_samples, device=self.device)  # (T,)
        t = t.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        # De Casteljau / Bernstein basis
        P0 = cp[:, 0:1, :]  # (E, 1, 2)
        P1 = cp[:, 1:2, :]
        P2 = cp[:, 2:3, :]
        P3 = cp[:, 3:4, :]

        # C(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        omt = 1.0 - t
        points = (
            omt**3 * P0 + 3 * omt**2 * t * P1 + 3 * omt * t**2 * P2 + t**3 * P3
        )  # (E, T, 2)

        return points

    def get_edge_tangents(
        self, edge_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute start and end tangent vectors for each edge.

        For a cubic Bezier, C'(0) = 3(P1 - P0) and C'(1) = 3(P3 - P2).

        Returns
        -------
        start_tangents : Tensor (E, 2)
        end_tangents : Tensor (E, 2)
        """
        cp = self.compute_control_points(edge_idx)  # (E, 4, 2)
        if cp.shape[0] == 0:
            empty = torch.zeros(0, 2, device=self.device)
            return empty, empty

        start_tangent = 3 * (cp[:, 1] - cp[:, 0])  # (E, 2)
        end_tangent = 3 * (cp[:, 3] - cp[:, 2])  # (E, 2)
        return start_tangent, end_tangent

    def node_degree(self) -> torch.Tensor:
        """Compute the degree of each node. Returns (N,) int tensor."""
        if self._n_nodes == 0 or self._n_edges == 0:
            return torch.zeros(self._n_nodes, dtype=torch.long, device=self.device)
        deg = torch.zeros(self._n_nodes, dtype=torch.long, device=self.device)
        for i in range(2):
            idx = self.edge_indices[:, i]
            deg.scatter_add_(0, idx, torch.ones_like(idx))
        return deg

    def edges_at_node(self, node_idx: int) -> List[Tuple[int, bool]]:
        """
        Find all edges incident to a node.

        Returns list of (edge_idx, is_start) tuples.
        """
        result = []
        for e in range(self._n_edges):
            if self.edge_indices[e, 0].item() == node_idx:
                result.append((e, True))
            if self.edge_indices[e, 1].item() == node_idx:
                result.append((e, False))
        return result

    def remove_edges(self, edge_mask: torch.Tensor):
        """
        Remove edges where edge_mask is True.

        Parameters
        ----------
        edge_mask : Tensor (E,) bool
            True for edges to REMOVE.
        """
        keep = ~edge_mask
        if keep.all():
            return

        self.edge_indices = self.edge_indices[keep]
        self.edge_alpha = nn.Parameter(self.edge_alpha.data[keep])
        self.edge_offset = nn.Parameter(self.edge_offset.data[keep])
        self.edge_width = nn.Parameter(self.edge_width.data[keep])
        self._n_edges = int(keep.sum().item())

    def remove_isolated_nodes(self):
        """Remove nodes with degree 0 and re-index edges."""
        if self._n_nodes == 0:
            return

        deg = self.node_degree()
        keep = deg > 0

        if keep.all():
            return

        # Build old-to-new index mapping
        new_idx = torch.full((self._n_nodes,), -1, dtype=torch.long, device=self.device)
        new_idx[keep] = torch.arange(keep.sum().item(), device=self.device)

        # Update node positions
        self.node_positions = nn.Parameter(self.node_positions.data[keep])
        self._n_nodes = int(keep.sum().item())

        # Remap edge indices
        if self._n_edges > 0:
            self.edge_indices = new_idx[self.edge_indices]

    def to_polylines(self, samples_per_edge: int = 50) -> List[np.ndarray]:
        """
        Convert all edges to polylines (numpy arrays of coordinates).

        Parameters
        ----------
        samples_per_edge : int
            Number of sample points per Bezier curve.

        Returns
        -------
        polylines : list of ndarray (N_i, 2)
            Each array is a sequence of (x, y) points.
        """
        with torch.no_grad():
            if self._n_edges == 0:
                return []
            points = self.sample_curves(n_samples=samples_per_edge)  # (E, T, 2)
            polylines = []
            for e in range(points.shape[0]):
                coords = points[e].cpu().numpy()
                polylines.append(coords)
            return polylines

    def get_optimizable_params(self) -> List[nn.Parameter]:
        """Return the list of parameters to optimize."""
        return [self.node_positions, self.edge_alpha, self.edge_offset, self.edge_width]

    def clone_detached(self) -> BezierGraph:
        """Create a detached copy of this graph for topology operations."""
        g = BezierGraph(self.canvas_h, self.canvas_w)
        g._n_nodes = self._n_nodes
        g._n_edges = self._n_edges

        if self._n_nodes > 0:
            g.node_positions = nn.Parameter(self.node_positions.data.clone())
        if self._n_edges > 0:
            g.edge_indices = self.edge_indices.clone()
            g.edge_alpha = nn.Parameter(self.edge_alpha.data.clone())
            g.edge_offset = nn.Parameter(self.edge_offset.data.clone())
            g.edge_width = nn.Parameter(self.edge_width.data.clone())

        return g

    def __repr__(self) -> str:
        return (
            f"BezierGraph(nodes={self._n_nodes}, edges={self._n_edges}, "
            f"canvas={self.canvas_h}x{self.canvas_w})"
        )
