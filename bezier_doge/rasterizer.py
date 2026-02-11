"""
Module 2: Trace Rasterization.

Converts GPS trace LineStrings into a 2D density/segmentation mask
that serves as the target for the Bezier Graph optimization
(equivalent to the SAM2 segmentation mask in the DOGE paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString


@dataclass
class RasterConfig:
    """Configuration for rasterization."""

    resolution_m: float = 1.0  # Meters per pixel
    vpd_buffer_m: float = 3.0  # Buffer width for VPD traces
    hpd_buffer_m: float = 5.0  # Buffer width for HPD traces (noisier)
    vpd_weight: float = 2.0  # Weight for VPD contributions
    hpd_weight: float = 1.0  # Weight for HPD contributions
    gaussian_sigma_px: float = 2.0  # Gaussian blur sigma in pixels
    threshold: float = 0.3  # Binarization threshold
    padding_m: float = 50.0  # Padding around UTM bounds


def rasterize_traces(
    lines_utm: List[LineString],
    sources: List[str],
    utm_bounds: Tuple[float, float, float, float],
    cfg: RasterConfig = RasterConfig(),
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Rasterize GPS traces into a segmentation mask.

    Parameters
    ----------
    lines_utm : list of LineString
        Traces in UTM coordinates.
    sources : list of str
        Source tag per trace ("VPD" or "HPD").
    utm_bounds : tuple
        (minx, miny, maxx, maxy) in UTM meters.
    cfg : RasterConfig
        Rasterization parameters.

    Returns
    -------
    soft_mask : ndarray (H, W), float32 in [0, 1]
        Continuous density mask after Gaussian smoothing and normalization.
    binary_mask : ndarray (H, W), float32 in {0, 1}
        Thresholded binary version.
    origin : tuple (origin_x, origin_y)
        UTM coordinates of the top-left corner (pixel [0,0]).
    """
    minx, miny, maxx, maxy = utm_bounds
    # Add padding
    minx -= cfg.padding_m
    miny -= cfg.padding_m
    maxx += cfg.padding_m
    maxy += cfg.padding_m

    width_m = maxx - minx
    height_m = maxy - miny

    W = int(np.ceil(width_m / cfg.resolution_m))
    H = int(np.ceil(height_m / cfg.resolution_m))

    print(
        f"[rasterizer] Raster size: {W} x {H} pixels ({width_m:.0f} x {height_m:.0f} m)"
    )

    # Accumulation grid
    density = np.zeros((H, W), dtype=np.float64)

    for line, source in zip(lines_utm, sources):
        if line is None or line.is_empty:
            continue

        source_upper = source.upper()
        buffer_m = cfg.vpd_buffer_m if source_upper == "VPD" else cfg.hpd_buffer_m
        weight = cfg.vpd_weight if source_upper == "VPD" else cfg.hpd_weight
        buffer_px = max(1, int(np.ceil(buffer_m / cfg.resolution_m)))

        coords = np.array(line.coords, dtype=np.float64)
        if len(coords) < 2:
            continue

        # Rasterize the line using Bresenham-like sampling
        total_len = line.length
        n_samples = max(int(total_len / (cfg.resolution_m * 0.5)), 2)

        for i in range(n_samples):
            frac = i / (n_samples - 1) if n_samples > 1 else 0.0
            pt = line.interpolate(frac, normalized=True)
            px = int((pt.x - minx) / cfg.resolution_m)
            py = int((maxy - pt.y) / cfg.resolution_m)  # Y is flipped (image coords)

            # Draw a filled circle of radius buffer_px
            for dy in range(-buffer_px, buffer_px + 1):
                for dx in range(-buffer_px, buffer_px + 1):
                    if dx * dx + dy * dy <= buffer_px * buffer_px:
                        cx = px + dx
                        cy = py + dy
                        if 0 <= cx < W and 0 <= cy < H:
                            density[cy, cx] += weight

    # Normalize density
    if density.max() > 0:
        density /= density.max()

    # Gaussian blur for smoothing
    if cfg.gaussian_sigma_px > 0:
        density = gaussian_filter(density, sigma=cfg.gaussian_sigma_px)
        if density.max() > 0:
            density /= density.max()

    soft_mask = density.astype(np.float32)
    binary_mask = (soft_mask >= cfg.threshold).astype(np.float32)

    origin = (minx, maxy)  # Top-left corner in UTM

    coverage_pct = float(binary_mask.sum()) / (H * W) * 100
    print(f"[rasterizer] Mask coverage: {coverage_pct:.1f}% of pixels")

    return soft_mask, binary_mask, origin


def rasterize_traces_fast(
    lines_utm: List[LineString],
    sources: List[str],
    utm_bounds: Tuple[float, float, float, float],
    cfg: RasterConfig = RasterConfig(),
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Faster rasterization using buffered polygon rasterization.

    Uses Shapely buffer + rasterio-style rasterization via numpy.
    Falls back to the basic method if needed.
    """
    minx, miny, maxx, maxy = utm_bounds
    minx -= cfg.padding_m
    miny -= cfg.padding_m
    maxx += cfg.padding_m
    maxy += cfg.padding_m

    width_m = maxx - minx
    height_m = maxy - miny

    W = int(np.ceil(width_m / cfg.resolution_m))
    H = int(np.ceil(height_m / cfg.resolution_m))

    print(f"[rasterizer_fast] Raster size: {W} x {H} pixels")

    density = np.zeros((H, W), dtype=np.float64)

    for line, source in zip(lines_utm, sources):
        if line is None or line.is_empty:
            continue

        source_upper = source.upper()
        buffer_m = cfg.vpd_buffer_m if source_upper == "VPD" else cfg.hpd_buffer_m
        weight = cfg.vpd_weight if source_upper == "VPD" else cfg.hpd_weight

        # Sample points along the line densely
        total_len = line.length
        step = cfg.resolution_m * 0.5
        n_samples = max(int(total_len / step), 2)

        # Get sampled points
        fracs = np.linspace(0.0, 1.0, n_samples)
        pts = np.array(
            [
                (
                    line.interpolate(f, normalized=True).x,
                    line.interpolate(f, normalized=True).y,
                )
                for f in fracs
            ]
        )

        # Convert to pixel coords
        px = ((pts[:, 0] - minx) / cfg.resolution_m).astype(np.int32)
        py = ((maxy - pts[:, 1]) / cfg.resolution_m).astype(np.int32)

        # Mask valid pixels
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px = px[valid]
        py = py[valid]

        # Use numpy advanced indexing (faster than per-pixel loop for the main line)
        np.add.at(density, (py, px), weight)

    # Dilate to approximate buffering
    if cfg.vpd_buffer_m > 0 or cfg.hpd_buffer_m > 0:
        avg_buffer_px = max(
            1, int(np.mean([cfg.vpd_buffer_m, cfg.hpd_buffer_m]) / cfg.resolution_m)
        )
        from scipy.ndimage import maximum_filter

        density = maximum_filter(density, size=2 * avg_buffer_px + 1)

    # Normalize
    if density.max() > 0:
        density /= density.max()

    # Gaussian blur
    if cfg.gaussian_sigma_px > 0:
        density = gaussian_filter(density, sigma=cfg.gaussian_sigma_px)
        if density.max() > 0:
            density /= density.max()

    soft_mask = density.astype(np.float32)
    binary_mask = (soft_mask >= cfg.threshold).astype(np.float32)

    origin = (minx, maxy)
    coverage_pct = float(binary_mask.sum()) / (H * W) * 100
    print(f"[rasterizer_fast] Mask coverage: {coverage_pct:.1f}%")

    return soft_mask, binary_mask, origin
