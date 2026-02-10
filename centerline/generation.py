"""Centerline generation pipeline -- orchestrator layer.

This module is the public entry-point for centerline generation.  It:

1. Loads and fuses VPD + HPD trace data.
2. Infers a local metric CRS.
3. Delegates the actual algorithm work to a pluggable
   ``BaseCenterlineAlgorithm`` implementation (default: ``kharita``).
4. Serialises results to GPKG / CSV / JSON.

Backward compatibility
----------------------
``CenterlineConfig`` and ``generate_centerlines()`` continue to work
exactly as before -- they are thin wrappers that construct and invoke
the ``KharitaAlgorithm`` under the hood.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyproj import CRS, Transformer

from .algorithms import BaseCenterlineAlgorithm
from .algorithms.kharita import KharitaConfig
from .io_utils import (
    clip_line_geometries_to_bbox,
    infer_local_projected_crs,
    load_bbox_from_txt,
    load_hpd_traces,
    load_vpd_traces,
)


# ---------------------------------------------------------------------------
# Legacy config  (kept for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass
class CenterlineConfig:
    """Configuration dataclass for the legacy ``generate_centerlines()`` API.

    .. deprecated::
        Prefer using ``KharitaConfig`` directly together with
        ``KharitaAlgorithm``, or pass an ``--algorithm`` flag to the CLI.
    """

    # Kharita-style heading-aware clustering
    cluster_radius_m: float = 10.0
    heading_tolerance_deg: float = 45.0
    heading_distance_weight_m: float = 0.22
    min_cluster_points: int = 1

    # Trace sampling and edge extraction
    sample_spacing_m: float = 8.0
    max_points_per_trace: int = 120
    max_transition_distance_m: float = 50.0

    # Edge pruning
    min_edge_support: float = 2.0
    reverse_edge_ratio: float = 0.2
    transitive_max_hops: int = 4
    transitive_ratio: float = 1.03
    transitive_max_checks: int = 25000

    # Centerline smoothing/cleanup
    smooth_iterations: int = 2
    min_centerline_length_m: float = 12.0

    # Trace weighting
    vpd_base_weight: float = 1.2
    hpd_base_weight: float = 1.0

    def to_kharita_config(self) -> KharitaConfig:
        """Convert to the algorithm-specific ``KharitaConfig``."""
        return KharitaConfig(
            cluster_radius_m=self.cluster_radius_m,
            heading_tolerance_deg=self.heading_tolerance_deg,
            heading_distance_weight_m=self.heading_distance_weight_m,
            min_cluster_points=self.min_cluster_points,
            sample_spacing_m=self.sample_spacing_m,
            max_points_per_trace=self.max_points_per_trace,
            max_transition_distance_m=self.max_transition_distance_m,
            min_edge_support=self.min_edge_support,
            reverse_edge_ratio=self.reverse_edge_ratio,
            transitive_max_hops=self.transitive_max_hops,
            transitive_ratio=self.transitive_ratio,
            transitive_max_checks=self.transitive_max_checks,
            smooth_iterations=self.smooth_iterations,
            min_centerline_length_m=self.min_centerline_length_m,
            vpd_base_weight=self.vpd_base_weight,
            hpd_base_weight=self.hpd_base_weight,
        )


# ---------------------------------------------------------------------------
# Data loading + CRS inference (shared across all algorithms)
# ---------------------------------------------------------------------------


def _load_and_prepare_traces(
        vpd_csv: str | Path,
        hpd_csvs: Iterable[str | Path],
        fused_only: bool = True,
        max_vpd_rows: int | None = None,
        max_hpd_rows_per_file: int | None = None,
        bbox_file: str | Path | None = None,
        apply_bbox: bool = False,
) -> tuple[pd.DataFrame, CRS | None, Transformer | None, Transformer | None]:
    """Load VPD + HPD, fuse, infer CRS, build transformers.

    Returns (traces, projected_crs, to_proj, to_wgs).
    If no valid traces are found the CRS/transformers will be ``None``.
    """
    vpd_df = load_vpd_traces(vpd_csv, fused_only=fused_only, max_rows=max_vpd_rows)
    hpd_df = load_hpd_traces(hpd_csvs, max_rows_per_file=max_hpd_rows_per_file)
    traces = pd.concat([vpd_df, hpd_df], ignore_index=True)
    traces = traces[traces["geometry"].notnull()].reset_index(drop=True)

    if apply_bbox and bbox_file is not None:
        bbox = load_bbox_from_txt(bbox_file)
        traces = clip_line_geometries_to_bbox(traces, bbox_wgs84=bbox, geometry_col="geometry")

    if traces.empty:
        return traces, None, None, None

    projected_crs: CRS = infer_local_projected_crs(list(traces["geometry"]))
    to_proj = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    to_wgs = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    return traces, projected_crs, to_proj, to_wgs


# ---------------------------------------------------------------------------
# Public API  (algorithm-aware)
# ---------------------------------------------------------------------------


def generate_centerlines_with_algorithm(
        vpd_csv: str | Path,
        hpd_csvs: Iterable[str | Path],
        algorithm: BaseCenterlineAlgorithm,
        fused_only: bool = True,
        max_vpd_rows: int | None = None,
        max_hpd_rows_per_file: int | None = None,
        bbox_file: str | Path | None = None,
        apply_bbox: bool = False,
) -> dict:
    """Run the full pipeline with a specific *algorithm* instance.

    This is the recommended entry-point for new code.
    """
    traces, projected_crs, to_proj, to_wgs = _load_and_prepare_traces(
        vpd_csv=vpd_csv,
        hpd_csvs=hpd_csvs,
        fused_only=fused_only,
        max_vpd_rows=max_vpd_rows,
        max_hpd_rows_per_file=max_hpd_rows_per_file,
        bbox_file=bbox_file,
        apply_bbox=apply_bbox,
    )

    if traces.empty:
        return {
            "projected_crs": None,
            "nodes": pd.DataFrame(),
            "edges": pd.DataFrame(),
            "centerlines": pd.DataFrame(),
            "trace_count": 0,
            "sample_point_count": 0,
        }

    return algorithm.generate(traces, projected_crs, to_proj, to_wgs)


# ---------------------------------------------------------------------------
# Legacy public API  (backward-compatible)
# ---------------------------------------------------------------------------


def generate_centerlines(
        vpd_csv: str | Path,
        hpd_csvs: Iterable[str | Path],
        config: CenterlineConfig | None = None,
        fused_only: bool = True,
        max_vpd_rows: int | None = None,
        max_hpd_rows_per_file: int | None = None,
        bbox_file: str | Path | None = None,
        apply_bbox: bool = False,
) -> dict:
    """Generate road centerlines (backward-compatible wrapper).

    This function preserves the original signature.  Internally it
    constructs a ``KharitaAlgorithm`` from the supplied
    ``CenterlineConfig`` and delegates.
    """
    from .algorithms.kharita import KharitaAlgorithm

    config = config or CenterlineConfig()
    algo = KharitaAlgorithm(config=config.to_kharita_config())

    return generate_centerlines_with_algorithm(
        vpd_csv=vpd_csv,
        hpd_csvs=hpd_csvs,
        algorithm=algo,
        fused_only=fused_only,
        max_vpd_rows=max_vpd_rows,
        max_hpd_rows_per_file=max_hpd_rows_per_file,
        bbox_file=bbox_file,
        apply_bbox=apply_bbox,
    )


# ---------------------------------------------------------------------------
# Output serialisation  (algorithm-agnostic)
# ---------------------------------------------------------------------------


def save_centerline_outputs(
        result: dict, output_dir: str | Path, stem: str = "generated_centerlines"
) -> dict:
    import geopandas as gpd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = result["nodes"].copy()
    edges = result["edges"].copy()
    centerlines = result["centerlines"].copy()

    files = {}

    def serialize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            if c == "geometry":
                continue
            if out[c].dtype == "object":
                out[c] = out[c].map(
                    lambda v: json.dumps(sorted(v) if isinstance(v, set) else v)
                    if isinstance(v, (list, dict, set))
                    else v
                )
        return out

    if not nodes.empty:
        nodes["geometry"] = gpd.points_from_xy(
            nodes["lon"], nodes["lat"], crs="EPSG:4326"
        )
        gnodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
        node_path = output_dir / f"{stem}_nodes.gpkg"
        gnodes.to_file(node_path, layer="nodes", driver="GPKG")
        files["nodes"] = str(node_path)

    if not edges.empty:
        edges_write = serialize_object_columns(edges)
        gedges = gpd.GeoDataFrame(edges_write, geometry="geometry", crs="EPSG:4326")
        edge_path = output_dir / f"{stem}_edges.gpkg"
        gedges.to_file(edge_path, layer="edges", driver="GPKG")
        files["edges"] = str(edge_path)

    if not centerlines.empty:
        center_write = serialize_object_columns(centerlines)
        gcenter = gpd.GeoDataFrame(center_write, geometry="geometry", crs="EPSG:4326")
        center_path = output_dir / f"{stem}.gpkg"
        gcenter.to_file(center_path, layer="centerlines", driver="GPKG")
        files["centerlines"] = str(center_path)

        csv_path = output_dir / f"{stem}.csv"
        out_csv = centerlines.copy()
        out_csv["geometry_wkt"] = out_csv["geometry"].astype(str)
        out_csv = out_csv.drop(columns=["geometry"])
        out_csv.to_csv(csv_path, index=False)
        files["centerlines_csv"] = str(csv_path)

    summary_path = output_dir / f"{stem}_summary.json"
    pd.Series(
        {
            "trace_count": int(result.get("trace_count", 0)),
            "sample_point_count": int(result.get("sample_point_count", 0)),
            "node_count": int(len(nodes)),
            "edge_count": int(len(edges)),
            "centerline_count": int(len(centerlines)),
            "projected_crs": str(result.get("projected_crs")),
        }
    ).to_json(summary_path, indent=2)
    files["summary"] = str(summary_path)

    return files
