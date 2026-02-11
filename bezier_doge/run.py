"""
Module 9: End-to-End Entry Point.

Orchestrates the full DOGE pipeline:
    1. Load VPD/HPD traces
    2. Rasterize to segmentation mask
    3. Tile the raster
    4. Optimize each tile
    5. Stitch tiles
    6. Export and evaluate

Usage:
    python -m bezier_doge.run --data-dir ./data --output-dir ./outputs/bezier_doge
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

# Imports from our modules
from .data_loader import LoadConfig, TraceData, load_all_traces
from .rasterizer import RasterConfig, rasterize_traces, rasterize_traces_fast
from .tiling import TileConfig, Tile, create_tiles, tile_to_global_coords
from .bezier_graph import BezierGraph
from .doge_optimizer import DOGEConfig, optimize_tile, optimize_tile_fast


def stitch_tile_graphs(
    tile_graphs: List[Tuple[Tile, BezierGraph]],
    resolution_m: float,
    merge_distance_m: float = 4.0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Stitch tile-level Bezier graphs into a global set of polylines.

    Converts each tile's graph to polylines in UTM coordinates,
    then merges endpoints that are close across tile boundaries.

    Parameters
    ----------
    tile_graphs : list of (Tile, BezierGraph)
    resolution_m : float
    merge_distance_m : float

    Returns
    -------
    all_nodes : ndarray (N, 2) — UTM positions
    all_polylines : list of ndarray (P_i, 2) — UTM polyline coordinates
    """
    all_polylines_utm = []

    for tile, graph in tile_graphs:
        if graph.n_edges == 0:
            continue

        polylines_px = graph.to_polylines(samples_per_edge=50)

        for poly_px in polylines_px:
            # Convert pixel coords to UTM
            utm_x, utm_y = tile_to_global_coords(
                poly_px[:, 0],
                poly_px[:, 1],
                tile,
                resolution_m,
            )
            poly_utm = np.column_stack([utm_x, utm_y])
            all_polylines_utm.append(poly_utm)

    if not all_polylines_utm:
        return np.zeros((0, 2)), []

    # Collect all endpoints for merging
    endpoints = []
    for i, poly in enumerate(all_polylines_utm):
        endpoints.append((i, 0, poly[0]))  # start point
        endpoints.append((i, -1, poly[-1]))  # end point

    if len(endpoints) < 2:
        return np.zeros((0, 2)), all_polylines_utm

    # Build KDTree for endpoint merging
    ep_coords = np.array([e[2] for e in endpoints])
    tree = cKDTree(ep_coords)
    pairs = tree.query_pairs(r=merge_distance_m)

    # For each pair of close endpoints, snap them to midpoint
    for i, j in pairs:
        pi_idx, pi_pos, pi_coord = endpoints[i]
        pj_idx, pj_pos, pj_coord = endpoints[j]

        # Don't merge endpoints of the same polyline
        if pi_idx == pj_idx:
            continue

        midpoint = (pi_coord + pj_coord) / 2.0

        # Snap both endpoints to midpoint
        all_polylines_utm[pi_idx][pi_pos] = midpoint
        all_polylines_utm[pj_idx][pj_pos] = midpoint

    # Collect unique nodes
    all_pts = np.vstack(all_polylines_utm) if all_polylines_utm else np.zeros((0, 2))

    return all_pts, all_polylines_utm


def polylines_to_wkt(
    polylines_utm: List[np.ndarray],
    utm_crs_epsg: int,
) -> List[str]:
    """
    Convert UTM polylines to WGS84 WKT LINESTRING format.

    Parameters
    ----------
    polylines_utm : list of ndarray (P_i, 2)
    utm_crs_epsg : int

    Returns
    -------
    wkt_strings : list of str
    """
    from pyproj import CRS, Transformer
    from shapely.geometry import LineString
    from shapely.ops import transform

    utm_crs = CRS.from_epsg(utm_crs_epsg)
    wgs84 = CRS.from_epsg(4326)
    to_wgs = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    wkt_list = []
    for poly_utm in polylines_utm:
        if len(poly_utm) < 2:
            continue
        line_utm = LineString(poly_utm)
        line_wgs = transform(to_wgs.transform, line_utm)
        if not line_wgs.is_empty:
            wkt_list.append(line_wgs.wkt)

    return wkt_list


def save_results(
    polylines_utm: List[np.ndarray],
    utm_crs_epsg: int,
    output_dir: Path,
    history: Optional[List] = None,
):
    """Save results as GPKG, CSV, and JSON."""
    import geopandas as gpd
    from pyproj import CRS, Transformer
    from shapely.geometry import LineString
    from shapely.ops import transform

    output_dir.mkdir(parents=True, exist_ok=True)

    utm_crs = CRS.from_epsg(utm_crs_epsg)
    wgs84 = CRS.from_epsg(4326)
    to_wgs = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    # Build GeoDataFrame
    geometries = []
    for poly_utm in polylines_utm:
        if len(poly_utm) < 2:
            continue
        line_utm = LineString(poly_utm)
        line_wgs = transform(to_wgs.transform, line_utm)
        if not line_wgs.is_empty:
            geometries.append(line_wgs)

    if geometries:
        gdf = gpd.GeoDataFrame(
            {"id": range(len(geometries)), "source": ["bezier_doge"] * len(geometries)},
            geometry=geometries,
            crs="EPSG:4326",
        )

        # Save in multiple formats
        gpkg_path = output_dir / "bezier_centerlines.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG", layer="centerlines")
        print(f"[save] GPKG: {gpkg_path} ({len(gdf)} lines)")

        csv_path = output_dir / "bezier_centerlines.csv"
        csv_df = gdf.copy()
        csv_df["geometry_wkt"] = csv_df.geometry.apply(lambda g: g.wkt)
        csv_df = csv_df.drop(columns=["geometry"])
        csv_df.to_csv(csv_path, index=False)
        print(f"[save] CSV: {csv_path}")
    else:
        print("[save] No geometries to save!")

    # Save optimization history
    if history:
        history_path = output_dir / "optimization_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        print(f"[save] History: {history_path}")

    # Save summary
    summary = {
        "n_centerlines": len(geometries) if geometries else 0,
        "total_length_m": sum(
            LineString(p).length for p in polylines_utm if len(p) >= 2
        ),
        "utm_crs": f"EPSG:{utm_crs_epsg}",
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Summary: {summary_path}")


def run(
    data_dir: str = "./data",
    output_dir: str = "./outputs/bezier_doge",
    device: str = "cuda",
    fast_mode: bool = False,
    max_tiles: Optional[int] = None,
    tile_size: int = 512,
    resolution_m: float = 1.0,
    T_max: int = 150,
):
    """
    Run the full DOGE-adapted Bezier centerline extraction pipeline.

    Parameters
    ----------
    data_dir : str
        Directory containing Kosovo_VPD/ and Kosovo_HPD/ folders.
    output_dir : str
        Output directory for results.
    device : str
        "cuda" or "cpu".
    fast_mode : bool
        Use faster settings (fewer iterations, lower resolution).
    max_tiles : int, optional
        Process only this many tiles (for debugging).
    tile_size : int
        Tile size in pixels.
    resolution_m : float
        Meters per pixel.
    T_max : int
        Maximum optimization iterations per tile.
    """
    total_start = time.time()
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOGE-Adapted Bezier Centerline Extraction")
    print("=" * 70)

    # ---- Step 1: Load Data ----
    print("\n--- Step 1: Loading data ---")
    load_cfg = LoadConfig()
    traces = load_all_traces(load_cfg, data_path)
    print(
        f"Loaded {len(traces.lines_utm)} traces "
        f"({traces.n_vpd} VPD, {traces.n_hpd} HPD)"
    )

    if not traces.lines_utm:
        print("ERROR: No traces loaded. Check data directory.")
        return

    # ---- Step 2: Rasterize ----
    print("\n--- Step 2: Rasterizing traces ---")
    raster_cfg = RasterConfig(resolution_m=resolution_m)
    if fast_mode:
        raster_cfg.gaussian_sigma_px = 3.0

    soft_mask, binary_mask, origin = rasterize_traces_fast(
        traces.lines_utm,
        traces.sources,
        traces.utm_bounds,
        raster_cfg,
    )

    # Save raster for visualization
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(soft_mask, cmap="hot", origin="upper")
        axes[0].set_title("Soft Density Mask")
        axes[1].imshow(binary_mask, cmap="gray", origin="upper")
        axes[1].set_title(f"Binary Mask (threshold={raster_cfg.threshold})")
        for ax in axes:
            ax.set_axis_off()
        fig.savefig(out_path / "raster_mask.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] Saved raster mask to {out_path / 'raster_mask.png'}")
    except Exception as e:
        print(f"[viz] Could not save raster visualization: {e}")

    # ---- Step 3: Create Tiles ----
    print("\n--- Step 3: Creating tiles ---")
    tile_cfg = TileConfig(tile_size=tile_size, overlap=64)
    tiles = create_tiles(soft_mask, binary_mask, origin, resolution_m, tile_cfg)

    if max_tiles is not None:
        tiles = tiles[:max_tiles]
        print(f"[tiles] Limited to {max_tiles} tiles for debugging")

    # ---- Step 4: Optimize Tiles ----
    print(f"\n--- Step 4: Optimizing {len(tiles)} tiles ---")
    doge_cfg = DOGEConfig(
        T_max=T_max if not fast_mode else min(T_max, 80),
        lr=0.5,
        device=device,
        diffalign=DiffAlignConfig(render_backend="auto"),
        topoadapt=TopoAdaptConfig(),
        log_interval=20,
    )

    tile_graphs = []
    all_history = []

    for i, tile in enumerate(tiles):
        print(
            f"\n  -- Tile {i + 1}/{len(tiles)} "
            f"(row={tile.row}, col={tile.col}, "
            f"coverage={tile.coverage:.1%}) --"
        )

        optimize_fn = optimize_tile_fast if fast_mode else optimize_tile
        try:
            graph, history = optimize_fn(tile.soft_mask, doge_cfg)
        except Exception as e:
            print(f"  ERROR on tile {i}: {e}")
            continue

        tile_graphs.append((tile, graph))
        all_history.extend(history)

        print(f"  Result: {graph.n_nodes} nodes, {graph.n_edges} edges")

    if not tile_graphs:
        print("ERROR: No tiles produced valid graphs.")
        return

    # ---- Step 5: Stitch ----
    print("\n--- Step 5: Stitching tiles ---")
    all_pts, polylines_utm = stitch_tile_graphs(
        tile_graphs,
        resolution_m,
        merge_distance_m=tile_cfg.merge_distance_m,
    )
    print(f"Stitched result: {len(polylines_utm)} polylines")

    # ---- Step 6: Save ----
    print("\n--- Step 6: Saving results ---")
    save_results(
        polylines_utm,
        utm_crs_epsg=load_cfg.target_crs_epsg,
        output_dir=out_path,
        history=all_history,
    )

    # ---- Step 7: Evaluate (if ground truth available) ----
    print("\n--- Step 7: Evaluation ---")
    gt_path = data_path / "Kosovo's nav streets" / "nav_kosovo.parquet"
    if not gt_path.exists():
        gt_path = data_path / "Kosovo's nav streets" / "nav_kosovo.csv"

    if gt_path.exists():
        try:
            from centerline.evaluation import evaluate_centerlines

            metrics = evaluate_centerlines(
                generated_centerlines=str(out_path / "bezier_centerlines.gpkg"),
                ground_truth_navstreet=str(gt_path),
                buffer_m=15.0,
                compute_topology_metrics=True,
            )
            print(f"\n  Precision: {metrics.get('length_precision', 'N/A'):.4f}")
            print(f"  Recall:    {metrics.get('length_recall', 'N/A'):.4f}")
            print(f"  F1:        {metrics.get('length_f1', 'N/A'):.4f}")

            metrics_path = out_path / "evaluation_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"  Saved metrics to {metrics_path}")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
    else:
        print(f"  Ground truth not found at {gt_path}, skipping evaluation.")

    elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Output directory: {out_path}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="DOGE-Adapted Bezier Centerline Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory with Kosovo_VPD/ and Kosovo_HPD/ data",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/bezier_doge",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Compute device"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use fast mode (fewer iterations)"
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Max tiles to process (for debugging)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512, help="Tile size in pixels"
    )
    parser.add_argument(
        "--resolution", type=float, default=1.0, help="Meters per pixel"
    )
    parser.add_argument(
        "--t-max", type=int, default=150, help="Max optimization iterations per tile"
    )

    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        fast_mode=args.fast,
        max_tiles=args.max_tiles,
        tile_size=args.tile_size,
        resolution_m=args.resolution,
        T_max=args.t_max,
    )


if __name__ == "__main__":
    main()
