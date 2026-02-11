"""
Module 8: DOGE Optimizer — Main Optimization Loop.

Implements Algorithm 1 from the DOGE paper: the global dynamic optimization
loop that alternates between TopoAdapt (discrete topology refinement) and
DiffAlign (continuous geometric optimization).

Reference: DOGE paper, Section 3.4 and Algorithm 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from .bezier_graph import BezierGraph
from .diffalign import DiffAlignConfig, diffalign_step
from .topoadapt import TopoAdaptConfig, topoadapt, road_addition


@dataclass
class DOGEConfig:
    """Configuration for the full DOGE optimization loop."""

    # Optimization
    T_max: int = 150  # Maximum iterations
    lr: float = 0.5  # Learning rate for Adam
    lr_decay: float = 0.995  # LR decay per iteration
    # DiffAlign steps per TopoAdapt call
    diffalign_steps: int = 5
    # TopoAdapt frequency
    topo_interval: int = 10  # Run TopoAdapt every N iterations
    # Early stopping
    patience: int = 30  # Stop if loss doesn't improve for this many iters
    min_delta: float = 1e-5  # Minimum improvement to reset patience
    # Sub-configs
    diffalign: DiffAlignConfig = field(default_factory=DiffAlignConfig)
    topoadapt: TopoAdaptConfig = field(default_factory=TopoAdaptConfig)
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    # Logging
    log_interval: int = 10  # Print losses every N iterations
    # Graph initialization
    max_initial_seeds: int = 30  # Max initial road segments


def initialize_graph(
    target: torch.Tensor,
    cfg: DOGEConfig,
) -> BezierGraph:
    """
    Initialize the Bezier Graph by seeding edges in high-confidence
    regions of the target segmentation mask.

    This corresponds to the InitializeGraph() call in Algorithm 1.
    """
    H, W = target.shape
    device = target.device

    graph = BezierGraph(canvas_h=H, canvas_w=W).to(device)

    # Use road_addition on the empty graph to seed initial edges
    init_cfg = TopoAdaptConfig(
        seg_threshold=cfg.topoadapt.seg_threshold,
        render_threshold=0.01,  # Very low since graph is empty
        max_new_roads=cfg.max_initial_seeds,
        new_road_length=cfg.topoadapt.new_road_length,
        new_road_width=cfg.topoadapt.new_road_width,
    )

    n_added = road_addition(graph, target, init_cfg)
    print(f"[init] Seeded {n_added} initial road segments")

    return graph


def optimize_tile(
    target: np.ndarray,
    cfg: DOGEConfig = DOGEConfig(),
) -> Tuple[BezierGraph, List[Dict]]:
    """
    Optimize a Bezier Graph for a single tile.

    This implements Algorithm 1 from the paper:
        1. Initialize graph
        2. For t = 0 to T_max:
            a. TopoAdapt (every topo_interval iterations)
            b. DiffAlign (multiple gradient steps)
        3. Return optimized graph

    Parameters
    ----------
    target : ndarray (H, W), float32
        Target segmentation mask for this tile.
    cfg : DOGEConfig

    Returns
    -------
    graph : BezierGraph
        Optimized Bezier graph.
    history : list of dict
        Per-iteration loss history.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Convert target to tensor
    target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    H, W = target_tensor.shape

    print(f"[optimizer] Tile size: {H}x{W}, device: {device}")

    # Step 1: Initialize graph
    graph = initialize_graph(target_tensor, cfg)
    print(f"[optimizer] Initial graph: {graph}")

    if graph.n_edges == 0:
        print("[optimizer] No edges initialized, skipping optimization")
        return graph, []

    # Step 2: Optimization loop
    history = []
    best_loss = float("inf")
    patience_counter = 0

    for t in range(cfg.T_max):
        iter_start = time.time()

        # Step 2a: TopoAdapt (discrete topology refinement)
        if t % cfg.topo_interval == 0:
            with torch.no_grad():
                topo_stats = topoadapt(graph, target_tensor, cfg.topoadapt, iteration=t)

            if t % cfg.log_interval == 0:
                print(
                    f"[optimizer] iter {t:3d} | TopoAdapt: {topo_stats} | "
                    f"Graph: {graph.n_nodes} nodes, {graph.n_edges} edges"
                )

            if graph.n_edges == 0:
                print("[optimizer] All edges pruned, re-seeding...")
                graph = initialize_graph(target_tensor, cfg)
                if graph.n_edges == 0:
                    break

        # Rebuild optimizer after topology changes (parameters may have changed)
        if t % cfg.topo_interval == 0 or t == 0:
            params = graph.get_optimizable_params()
            lr = cfg.lr * (cfg.lr_decay**t)
            optimizer = torch.optim.Adam(params, lr=lr)

        # Step 2b: DiffAlign (continuous geometric optimization)
        for step in range(cfg.diffalign_steps):
            try:
                loss_dict = diffalign_step(
                    graph, target_tensor, optimizer, cfg.diffalign
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[optimizer] OOM at iter {t}, step {step}. Reducing graph.")
                    torch.cuda.empty_cache()
                    break
                raise

        # Log
        history.append(
            {
                "iteration": t,
                "time_s": time.time() - iter_start,
                **loss_dict,
                "n_nodes": graph.n_nodes,
                "n_edges": graph.n_edges,
                "lr": lr if "lr" in dir() else cfg.lr,
            }
        )

        if t % cfg.log_interval == 0:
            print(
                f"[optimizer] iter {t:3d} | "
                f"loss={loss_dict['total']:.4f} "
                f"(cover={loss_dict['cover']:.4f}, "
                f"overlap={loss_dict['overlap']:.4f}, "
                f"g1={loss_dict['g1']:.4f}) | "
                f"{graph.n_nodes}N {graph.n_edges}E"
            )

        # Early stopping
        current_loss = loss_dict["total"]
        if current_loss < best_loss - cfg.min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(
                f"[optimizer] Early stopping at iteration {t} "
                f"(no improvement for {cfg.patience} iterations)"
            )
            break

    print(f"[optimizer] Final graph: {graph}")
    return graph, history


def optimize_tile_fast(
    target: np.ndarray,
    cfg: DOGEConfig = DOGEConfig(),
) -> Tuple[BezierGraph, List[Dict]]:
    """
    Faster optimization variant with reduced parameters for demo use.

    Uses fewer iterations, less frequent topology updates, and
    skips expensive loss terms.
    """
    fast_cfg = DOGEConfig(
        T_max=min(cfg.T_max, 80),
        lr=cfg.lr,
        lr_decay=cfg.lr_decay,
        diffalign_steps=3,
        topo_interval=15,
        patience=20,
        min_delta=cfg.min_delta,
        diffalign=DiffAlignConfig(
            lambda_cover=1.0,
            lambda_overlap=0.1,  # Reduced overlap computation
            lambda_g1=0.01,
            lambda_offset=0.005,
            lambda_spacing=0.005,
            n_samples=32,  # Fewer samples
            render_backend=cfg.diffalign.render_backend,
        ),
        topoadapt=cfg.topoadapt,
        device=cfg.device,
        log_interval=cfg.log_interval,
        max_initial_seeds=cfg.max_initial_seeds,
    )
    return optimize_tile(target, fast_cfg)
