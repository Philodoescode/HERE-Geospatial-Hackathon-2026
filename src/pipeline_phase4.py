"""
Phase 4: Final Geometry Optimization (formerly Phase 5)

Lightweight final pass — runs AFTER the consolidated Phase 3 cleanup.
Only performs operations unique to this phase:
  1. Interchange zone cleanup (cloverleaf/bridge knot removal)
  2. Quality-based candidate selection (disabled by default)
  3. Geometry cleaning + export (local CRS + WGS84)

NOTE: Smoothing, parallel merge, stub pruning, and Z-level resolution
are now consolidated in Phase 3 and NOT duplicated here.
"""

import gc
import os
import sys
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString

warnings.filterwarnings("ignore")

# Import quality scoring module
try:
    from src.algorithms.quality_scoring import (
        QualityConfig,
        enhance_segments_with_quality,
    )
    QUALITY_SCORING_AVAILABLE = True
except ImportError:
    QUALITY_SCORING_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GeometryOptimizer:
    """
    Phase 4 (formerly Phase 5): Lightweight final geometry optimization.

    Only performs operations that are unique to this phase:
      1. Complex interchange zone cleanup
      2. Quality-based candidate selection (optional)
      3. Geometry cleaning & export
    """

    def __init__(
        self,
        input_path: str,
        output_dir: str = "data",
        min_length: float = 4.0,
        # Complex interchange handling
        enable_interchange_cleanup: bool = True,
        interchange_density_threshold: int = 5,
        interchange_zone_radius_m: float = 50.0,
        interchange_min_length_m: float = 15.0,
        interchange_crossing_angle_deg: float = 30.0,
        # Quality-based candidate selection (disabled by default)
        enable_quality_selection: bool = False,
        quality_selection_threshold: float = 0.30,
        quality_force_keep_support: float = 6.0,
        quality_dangling_max_length_m: float = 35.0,
        quality_dangling_min_support: float = 6.0,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.min_length = min_length

        # Interchange cleanup params
        self.enable_interchange_cleanup = enable_interchange_cleanup
        self.interchange_density_threshold = interchange_density_threshold
        self.interchange_zone_radius_m = interchange_zone_radius_m
        self.interchange_min_length_m = interchange_min_length_m
        self.interchange_crossing_angle_deg = interchange_crossing_angle_deg

        # Quality selection params
        self.enable_quality_selection = enable_quality_selection
        self.quality_selection_threshold = quality_selection_threshold
        self.quality_force_keep_support = quality_force_keep_support
        self.quality_dangling_max_length_m = quality_dangling_max_length_m
        self.quality_dangling_min_support = quality_dangling_min_support

        self.gdf = None
        self.local_crs = None
        self.final_gdf = None

    # ──────────────────────────────────────────────────────────────────
    #  DATA LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_data(self):
        print("  Loading refined skeleton...")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input not found: {self.input_path}")

        self.gdf = gpd.read_file(self.input_path)

        if self.gdf.crs is None or self.gdf.crs.is_geographic:
            bounds = self.gdf.total_bounds
            cx = (bounds[0] + bounds[2]) / 2.0
            cy = (bounds[1] + bounds[3]) / 2.0
            self.local_crs = (
                f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
                f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            )
            self.gdf = self.gdf.to_crs(self.local_crs)
        else:
            self.local_crs = self.gdf.crs

        # Explode MultiLineStrings
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)

        # Filter valid
        self.gdf = self.gdf[
            self.gdf.geometry.notna()
            & ~self.gdf.geometry.is_empty
            & (self.gdf.geom_type == "LineString")
        ].reset_index(drop=True)

        print(f"    Loaded {len(self.gdf)} line segments")

    # ──────────────────────────────────────────────────────────────────
    #  COMPLEX INTERCHANGE CLEANUP
    # ──────────────────────────────────────────────────────────────────

    def cleanup_interchange_zones(self):
        """
        Detect and clean up complex interchange zones (cloverleafs, bridges).

        At these locations, GPS noise causes a "central knot" where lines
        from different road levels get mashed together.

        Strategy:
          1. Detect high-density zones where many lines cross
          2. Remove short "crossing connector" segments that jump between flows
          3. Preserve main through-routes based on length and consistency
        """
        if not self.enable_interchange_cleanup:
            return

        print("  Cleaning up complex interchange zones...")
        before_count = len(self.gdf)

        if before_count < 5:
            return

        # Find potential interchange centroids (high line density)
        geoms = self.gdf.geometry.values

        # Collect line midpoints for density analysis
        midpoints = []
        for geom in geoms:
            if isinstance(geom, LineString) and len(geom.coords) >= 2:
                mid = geom.interpolate(0.5, normalized=True)
                midpoints.append((mid.x, mid.y))

        if len(midpoints) < 5:
            return

        midpoints_arr = np.array(midpoints)
        mid_tree = cKDTree(midpoints_arr)

        # Find dense clusters that indicate interchange zones
        interchange_zones = []
        checked: set = set()

        for i, (mx, my) in enumerate(midpoints):
            if i in checked:
                continue

            nearby_idx = mid_tree.query_ball_point(
                [mx, my], r=self.interchange_zone_radius_m
            )

            if len(nearby_idx) >= self.interchange_density_threshold:
                zone_x = np.mean([midpoints[j][0] for j in nearby_idx])
                zone_y = np.mean([midpoints[j][1] for j in nearby_idx])
                interchange_zones.append({
                    "center_x": zone_x,
                    "center_y": zone_y,
                    "line_indices": nearby_idx,
                    "density": len(nearby_idx),
                })
                for j in nearby_idx:
                    checked.add(j)

        if not interchange_zones:
            print("    No complex interchange zones detected")
            return

        print(f"    Detected {len(interchange_zones)} interchange zones")

        # Analyze lines in each interchange zone
        to_remove: set = set()

        for zone in interchange_zones:
            zone_indices = zone["line_indices"]
            zone_cx, zone_cy = zone["center_x"], zone["center_y"]

            zone_lines = []
            for idx in zone_indices:
                if idx >= len(self.gdf):
                    continue

                row = self.gdf.iloc[idx]
                geom = row.geometry
                if not isinstance(geom, LineString) or len(geom.coords) < 2:
                    continue

                coords = np.array(geom.coords)
                dx = coords[-1, 0] - coords[0, 0]
                dy = coords[-1, 1] - coords[0, 1]
                heading = np.degrees(np.arctan2(dy, dx)) % 180.0

                zone_lines.append({
                    "idx": idx,
                    "length": geom.length,
                    "heading": heading,
                    "source": row.get("source", "VPD") if "source" in row.index else "VPD",
                })

            if len(zone_lines) < 3:
                continue

            # Group lines by heading
            heading_groups: dict = {}
            for line_info in zone_lines:
                h = line_info["heading"]
                assigned = False
                for gh, group in heading_groups.items():
                    hdiff = abs(h - gh)
                    hdiff = min(hdiff, 180 - hdiff)
                    if hdiff <= self.interchange_crossing_angle_deg:
                        group.append(line_info)
                        assigned = True
                        break
                if not assigned:
                    heading_groups[h] = [line_info]

            # Main flows have multiple consistent-heading lines
            main_flow_headings = {
                gh for gh, group in heading_groups.items() if len(group) >= 2
            }

            for line_info in zone_lines:
                h = line_info["heading"]
                is_main_flow = False
                for mfh in main_flow_headings:
                    hdiff = abs(h - mfh)
                    hdiff = min(hdiff, 180 - hdiff)
                    if hdiff <= self.interchange_crossing_angle_deg:
                        is_main_flow = True
                        break

                # Remove short non-main-flow lines (noise/connectors)
                if not is_main_flow and line_info["length"] < self.interchange_min_length_m:
                    to_remove.add(line_info["idx"])
                elif line_info["length"] < self.interchange_min_length_m * 0.5:
                    to_remove.add(line_info["idx"])

        if to_remove:
            keep_mask = [i not in to_remove for i in range(len(self.gdf))]
            self.gdf = self.gdf[keep_mask].reset_index(drop=True)

        removed = before_count - len(self.gdf)
        print(f"    Removed {removed} interchange noise segments ({len(self.gdf)} remaining)")

    # ──────────────────────────────────────────────────────────────────
    #  QUALITY-BASED CANDIDATE SELECTION
    # ──────────────────────────────────────────────────────────────────

    def apply_quality_selection(self):
        """
        Apply quality-based candidate selection to filter weak segments.
        Currently disabled by default — kept for tuning.
        """
        if not self.enable_quality_selection:
            print("  Quality selection: DISABLED (skipping)")
            return

        if not QUALITY_SCORING_AVAILABLE:
            print("  Quality scoring module not available, skipping...")
            return

        print("  Applying quality-based candidate selection...")
        before = len(self.gdf)

        config = QualityConfig(
            candidate_selection_enabled=True,
            candidate_selection_threshold=self.quality_selection_threshold,
            candidate_force_keep_support=self.quality_force_keep_support,
            candidate_dangling_max_length_m=self.quality_dangling_max_length_m,
            candidate_dangling_min_support=self.quality_dangling_min_support,
            candidate_short_length_m=10.0,
            candidate_low_support=4.0,
            candidate_length_scale_m=70.0,
        )

        scored_gdf = enhance_segments_with_quality(self.gdf, config)

        if "is_candidate" in scored_gdf.columns:
            has_source = "source" in scored_gdf.columns
            if has_source:
                keep_mask = (
                    scored_gdf["is_candidate"]
                    | (scored_gdf["source"] == "VPD")
                    | (scored_gdf["source"] == "roundabout")
                )
            else:
                keep_mask = scored_gdf["is_candidate"]

            self.gdf = scored_gdf[keep_mask].reset_index(drop=True)
        else:
            self.gdf = scored_gdf

        removed = before - len(self.gdf)
        print(f"    Quality selection: {before} -> {len(self.gdf)} ({removed} removed)")

    # ──────────────────────────────────────────────────────────────────
    #  CLEANING + EXPORT
    # ──────────────────────────────────────────────────────────────────

    def clean_geometries(self):
        """Fix invalid geometries, remove short/empty segments."""
        print("  Cleaning geometries...")
        before = len(self.gdf)

        self.gdf = self.gdf[~self.gdf.geometry.is_empty].copy()

        invalid_mask = ~self.gdf.geometry.is_valid
        if invalid_mask.sum() > 0:
            self.gdf.loc[invalid_mask, "geometry"] = (
                self.gdf.loc[invalid_mask, "geometry"].buffer(0)
            )

        self.gdf = self.gdf[
            self.gdf.geom_type.isin(["LineString", "MultiLineString"])
        ].copy()

        self.gdf = self.gdf[self.gdf.geometry.length >= self.min_length].copy()
        self.gdf = self.gdf.reset_index(drop=True)

        print(f"    Cleaned: {before} -> {len(self.gdf)} segments")

    def export(self):
        self.clean_geometries()
        self.final_gdf = self.gdf.copy()

        output_path = os.path.join(self.output_dir, "final_centerline_output.gpkg")
        print(f"  Exporting to {output_path}...")

        self.final_gdf.to_file(output_path, driver="GPKG")

        output_4326 = output_path.replace(".gpkg", "_4326.gpkg")
        gdf_4326 = self.final_gdf.to_crs("EPSG:4326")
        gdf_4326.to_file(output_4326, driver="GPKG")
        print(f"  Exported: {len(self.final_gdf)} segments")
        print(f"    Local CRS: {output_path}")
        print(f"    WGS84:     {output_4326}")

    # ──────────────────────────────────────────────────────────────────
    #  MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  PHASE 4: Final Geometry Optimization")
        print("=" * 60)
        t0 = time.time()

        self.load_data()

        # Interchange zone cleanup (unique to this phase)
        self.cleanup_interchange_zones()

        # Quality selection (disabled by default, kept for tuning)
        self.apply_quality_selection()

        # Export (includes clean_geometries)
        self.export()

        elapsed = time.time() - t0
        total_km = self.final_gdf.geometry.length.sum() / 1000.0
        print(f"\nPhase 4 complete in {elapsed:.1f}s")
        print(f"  Final network: {len(self.final_gdf)} segments, {total_km:.1f} km")

        return self.final_gdf


if __name__ == "__main__":
    INPUT_FILE = os.path.join(
        PROJECT_ROOT, "data", "interim_refined_skeleton_phase3.gpkg"
    )
    optimizer = GeometryOptimizer(INPUT_FILE)
    optimizer.run()
