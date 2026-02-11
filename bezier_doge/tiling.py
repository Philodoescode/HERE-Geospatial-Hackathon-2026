"""
Module 3: Tile Management.

Splits a large raster into overlapping tiles for per-tile DOGE optimization,
and provides stitching logic to merge tile results back into a global graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TileConfig:
    """Configuration for tiling."""

    tile_size: int = 512  # Tile dimension in pixels
    overlap: int = 64  # Overlap in pixels on each side
    min_coverage: float = 0.02  # Minimum fraction of non-zero pixels to process a tile
    merge_distance_m: float = 4.0  # Distance for merging nodes across tile boundaries


@dataclass
class Tile:
    """A single tile extracted from the global raster."""

    row: int  # Tile grid row index
    col: int  # Tile grid column index
    # Pixel coordinates in the global raster
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    # The tile's segmentation mask (soft)
    soft_mask: Optional[np.ndarray] = None
    # Coverage fraction
    coverage: float = 0.0
    # Origin in UTM coords (top-left of this tile)
    origin_utm_x: float = 0.0
    origin_utm_y: float = 0.0

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    def is_empty(self, min_coverage: float = 0.02) -> bool:
        return self.coverage < min_coverage


def create_tiles(
    soft_mask: np.ndarray,
    binary_mask: np.ndarray,
    global_origin: Tuple[float, float],
    resolution_m: float,
    cfg: TileConfig = TileConfig(),
) -> List[Tile]:
    """
    Create overlapping tiles from the global raster.

    Parameters
    ----------
    soft_mask : ndarray (H, W)
        The soft segmentation mask.
    binary_mask : ndarray (H, W)
        The binary segmentation mask.
    global_origin : tuple (origin_x, origin_y)
        UTM coords of pixel (0, 0) — top-left corner.
    resolution_m : float
        Meters per pixel.
    cfg : TileConfig
        Tiling parameters.

    Returns
    -------
    tiles : list of Tile
        Non-empty tiles to process.
    """
    H, W = soft_mask.shape
    step = cfg.tile_size - 2 * cfg.overlap  # Effective step between tiles
    step = max(step, 1)

    tiles = []
    n_rows = max(1, int(np.ceil((H - cfg.overlap) / step)))
    n_cols = max(1, int(np.ceil((W - cfg.overlap) / step)))

    print(
        f"[tiling] Grid: {n_rows} x {n_cols} tiles "
        f"(tile_size={cfg.tile_size}, overlap={cfg.overlap}, step={step})"
    )

    total_tiles = 0
    kept_tiles = 0

    for r in range(n_rows):
        for c in range(n_cols):
            y_start = r * step
            x_start = c * step
            y_end = min(y_start + cfg.tile_size, H)
            x_end = min(x_start + cfg.tile_size, W)

            # Ensure minimum tile size
            if y_end - y_start < 64 or x_end - x_start < 64:
                continue

            tile_binary = binary_mask[y_start:y_end, x_start:x_end]
            tile_soft = soft_mask[y_start:y_end, x_start:x_end].copy()
            coverage = float(tile_binary.sum()) / tile_binary.size

            total_tiles += 1

            # UTM origin of this tile
            origin_x = global_origin[0] + x_start * resolution_m
            origin_y = global_origin[1] - y_start * resolution_m  # Y flipped

            tile = Tile(
                row=r,
                col=c,
                y_start=y_start,
                y_end=y_end,
                x_start=x_start,
                x_end=x_end,
                soft_mask=tile_soft,
                coverage=coverage,
                origin_utm_x=origin_x,
                origin_utm_y=origin_y,
            )

            if not tile.is_empty(cfg.min_coverage):
                tiles.append(tile)
                kept_tiles += 1

    print(
        f"[tiling] {kept_tiles}/{total_tiles} tiles above coverage threshold "
        f"({cfg.min_coverage * 100:.0f}%)"
    )

    return tiles


def tile_to_global_coords(
    local_x: np.ndarray,
    local_y: np.ndarray,
    tile: Tile,
    resolution_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tile-local pixel coordinates to global UTM coordinates.

    Parameters
    ----------
    local_x, local_y : ndarray
        Pixel coordinates within the tile.
    tile : Tile
        The tile these coordinates belong to.
    resolution_m : float
        Meters per pixel.

    Returns
    -------
    utm_x, utm_y : ndarray
        UTM coordinates.
    """
    utm_x = tile.origin_utm_x + local_x * resolution_m
    utm_y = tile.origin_utm_y - local_y * resolution_m  # Y is flipped
    return utm_x, utm_y


def global_to_tile_coords(
    utm_x: np.ndarray,
    utm_y: np.ndarray,
    tile: Tile,
    resolution_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert global UTM coordinates to tile-local pixel coordinates.
    """
    local_x = (utm_x - tile.origin_utm_x) / resolution_m
    local_y = (tile.origin_utm_y - utm_y) / resolution_m
    return local_x, local_y
