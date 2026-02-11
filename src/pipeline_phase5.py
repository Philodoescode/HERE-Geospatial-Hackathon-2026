"""
Phase 5: Geometry Optimization & Topology Cleanup

Final refinement of the merged network:
  1. Adaptive B-spline smoothing with tight Hausdorff guard
  2. Redundancy removal (parallel duplicate detection)
  3. Stub pruning (iterative dead-end removal)
  4. Quality-based candidate selection (road-likeness filtering)
  5. Geometry cleaning
  6. Export in both local CRS and WGS84
"""

import gc
import os
import sys
import time
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge

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

# Import curve smoothing utilities
try:
    from src.algorithms.curve_smoothing import (
        simplify_and_smooth_centerline,
        merge_nearby_points,
        fix_overlapping_segments,
    )
    CURVE_SMOOTHING_AVAILABLE = True
except ImportError:
    CURVE_SMOOTHING_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GeometryOptimizer:
    """
    Phase 5: Final geometry optimization and topology cleanup.

    Steps:
      1. Smooth lines with adaptive B-splines (tight Hausdorff guard)
      2. Remove near-duplicate parallel segments
      3. Prune dead-end stubs
      4. Quality-based candidate selection (road-likeness filtering)
      5. Clean invalid geometries
      6. Export
    """

    def __init__(
        self,
        input_path,
        output_dir="data",
        smoothing_factor=1.0,    # gentler smoothing to avoid shifting geometry
        points_per_100m=15,      # higher density than before
        max_hausdorff_dev=2.0,   # tight deviation limit to prevent drift
        stub_threshold=8.0,     # dead-end pruning threshold (relaxed)
        probe_stub_threshold=10.0,  # relaxed stub pruning for probe lines
        redundancy_buffer=8.0,   # parallel duplicate detection buffer
        heading_threshold=20.0,  # heading compatibility for dedup
        min_length=4.0,          # minimum segment length
        probe_min_length=8.0,    # relaxed minimum for probe segments
        # Quality-based candidate selection parameters
        # DISABLED: Was hurting recovery by filtering too aggressively
        enable_quality_selection=False,
        quality_selection_threshold=0.30,  # lower threshold if re-enabled
        quality_force_keep_support=6.0,    # lower threshold if re-enabled
        quality_dangling_max_length_m=35.0,
        quality_dangling_min_support=6.0,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.smoothing_factor = smoothing_factor
        self.points_per_100m = points_per_100m
        self.max_hausdorff_dev = max_hausdorff_dev
        self.stub_threshold = stub_threshold
        self.redundancy_buffer = redundancy_buffer
        self.heading_threshold = heading_threshold
        self.min_length = min_length
        self.probe_min_length = probe_min_length
        self.probe_stub_threshold = probe_stub_threshold
        
        # Quality selection parameters
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
        print("  Loading merged network...")
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
    #  SMOOTHING (B-spline with tight Hausdorff guard)
    # ──────────────────────────────────────────────────────────────────

    def smooth_geometry(self):
        """
        Adaptive B-spline smoothing with length-proportional
        output point count and tight Hausdorff deviation limit.
        Also applies curve simplification to remove overlapping clusters.
        """
        print("  Smoothing geometry (adaptive B-splines + curve cleanup)...")
        smoothed_geoms = []
        smooth_count = 0
        skip_count = 0
        curve_cleaned = 0

        for idx, geom in enumerate(self.gdf.geometry):
            if not isinstance(geom, LineString):
                smoothed_geoms.append(geom)
                skip_count += 1
                continue

            n_coords = len(geom.coords)
            
            # First, apply curve simplification to remove overlapping clusters
            if CURVE_SMOOTHING_AVAILABLE and n_coords >= 4:
                try:
                    coords = np.array(geom.coords)
                    cleaned_coords = simplify_and_smooth_centerline(
                        coords,
                        simplify_tolerance=2.0,
                        merge_distance=3.0,
                        smooth_iterations=1,
                        smooth_weight=0.2,
                    )
                    if len(cleaned_coords) >= 2:
                        geom = LineString(cleaned_coords)
                        n_coords = len(geom.coords)
                        curve_cleaned += 1
                except Exception:
                    pass
            
            if n_coords < 4:
                smoothed_geoms.append(geom)
                skip_count += 1
                continue

            try:
                x, y = geom.xy
                x = list(x)
                y = list(y)

                n_out = max(4, int(geom.length / 100.0 * self.points_per_100m))
                n_out = min(n_out, 150)

                k = min(3, n_coords - 1)
                tck, u = splprep([x, y], s=self.smoothing_factor, k=k)
                new_points = splev(np.linspace(0, 1, n_out), tck)

                new_line = LineString(zip(new_points[0], new_points[1]))

                # Tight Hausdorff guard
                if new_line.hausdorff_distance(geom) < self.max_hausdorff_dev:
                    smoothed_geoms.append(new_line)
                    smooth_count += 1
                else:
                    smoothed_geoms.append(geom)
                    skip_count += 1
            except Exception:
                smoothed_geoms.append(geom)
                skip_count += 1

        self.gdf["geometry"] = smoothed_geoms
        print(f"    Curve cleanup: {curve_cleaned} lines simplified")
        print(f"    B-spline smoothed: {smooth_count} lines, skipped: {skip_count}")

    # ──────────────────────────────────────────────────────────────────
    #  REDUNDANCY REMOVAL
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _heading_0_180(geom):
        """Direction-agnostic heading."""
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            return 0.0
        c = geom.coords
        dx = c[-1][0] - c[0][0]
        dy = c[-1][1] - c[0][1]
        return np.degrees(np.arctan2(dy, dx)) % 180.0

    def remove_redundancy(self):
        """
        Remove near-duplicate parallel segments.
        Uses buffer overlap ratio (not just Hausdorff) for robustness
        against longitudinal offsets in dual-direction skeletons.
        Confidence weighting: always keep VPD over Probe.
        """
        print(f"  Removing redundant parallel segments "
              f"(buffer={self.redundancy_buffer}m, heading<{self.heading_threshold} deg)...")
        before = len(self.gdf)

        if before < 2:
            return

        # Compute headings
        headings = np.array([self._heading_0_180(g) for g in self.gdf.geometry])
        lengths = self.gdf.geometry.length.values

        # Spatial pre-filter
        geoms = self.gdf.geometry.values
        tree = STRtree(geoms)
        left_idx, right_idx = tree.query(
            geoms, predicate="dwithin", distance=self.redundancy_buffer
        )

        # Keep only i < j pairs
        mask = left_idx < right_idx
        left_idx = left_idx[mask]
        right_idx = right_idx[mask]

        has_source = "source" in self.gdf.columns
        to_remove = set()

        for k in range(len(left_idx)):
            i, j = left_idx[k], right_idx[k]
            if i in to_remove or j in to_remove:
                continue

            # Heading compatibility
            h_diff = abs(headings[i] - headings[j])
            h_diff = min(h_diff, 180 - h_diff)
            if h_diff > self.heading_threshold:
                continue

            # Buffer overlap check (more robust than Hausdorff for offset lines)
            shorter_idx = i if lengths[i] < lengths[j] else j
            longer_idx = j if shorter_idx == i else i

            shorter_buf = geoms[longer_idx].buffer(self.redundancy_buffer)
            try:
                inside = geoms[shorter_idx].intersection(shorter_buf)
                overlap = inside.length / lengths[shorter_idx] if lengths[shorter_idx] > 0 else 0
            except Exception:
                overlap = 0

            if overlap < 0.5:
                continue  # Less than 50% overlap -> not truly parallel

            # Length similarity check: don't remove lines of very different lengths
            # as they likely cover different road segments
            len_ratio = min(lengths[i], lengths[j]) / max(lengths[i], lengths[j]) if max(lengths[i], lengths[j]) > 0 else 1.0
            if len_ratio < 0.3:
                continue  # Very different lengths -> likely different road segments

            # Confidence weighting: VPD > Probe, protect roundabouts
            src_i = self.gdf.iloc[i]["source"] if has_source else ""
            src_j = self.gdf.iloc[j]["source"] if has_source else ""

            # Never remove roundabout segments
            if src_i == "roundabout" or src_j == "roundabout":
                continue

            if src_i == "VPD" and src_j != "VPD":
                to_remove.add(j)
            elif src_j == "VPD" and src_i != "VPD":
                to_remove.add(i)
            else:
                # Same source: remove shorter
                to_remove.add(shorter_idx)

        if to_remove:
            self.gdf = self.gdf.drop(index=list(to_remove)).reset_index(drop=True)

        removed = before - len(self.gdf)
        print(f"    Removed {removed} redundant segments ({len(self.gdf)} remaining)")

    # ──────────────────────────────────────────────────────────────────
    #  STUB PRUNING
    # ──────────────────────────────────────────────────────────────────

    def prune_stubs(self):
        """
        Iteratively remove dead-end segments shorter than threshold.
        Uses different thresholds for VPD vs Probe sources.
        """
        print(f"  Pruning stubs (VPD < {self.stub_threshold}m, Probe < {self.probe_stub_threshold}m)...")

        def round_coord(c):
            return (round(c[0], 1), round(c[1], 1))

        has_source = "source" in self.gdf.columns
        total_removed = 0
        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1

            G = nx.Graph()
            for idx, row in self.gdf.iterrows():
                geom = row.geometry
                if not isinstance(geom, LineString) or len(geom.coords) < 2:
                    continue
                u = round_coord(geom.coords[0])
                v = round_coord(geom.coords[-1])
                if u == v:
                    continue
                source = row.get("source", "VPD") if has_source else "VPD"
                G.add_edge(u, v, idx=idx, length=geom.length, source=source)

            stubs_to_remove = set()
            for node in G.nodes():
                if G.degree(node) == 1:
                    neighbor = list(G.neighbors(node))[0]
                    data = G[node][neighbor]
                    length = data.get("length", float("inf"))
                    source = data.get("source", "VPD")
                    # Use different thresholds: be aggressive with probe stubs
                    threshold = self.probe_stub_threshold if source == "Probe" else self.stub_threshold
                    if length < threshold:
                        stubs_to_remove.add(data["idx"])

            if not stubs_to_remove:
                break

            self.gdf = self.gdf.drop(index=list(stubs_to_remove)).reset_index(drop=True)
            total_removed += len(stubs_to_remove)

        print(f"    Removed {total_removed} stubs over {iteration} iteration(s)")
        print(f"    Network: {len(self.gdf)} segments remaining")

    # ──────────────────────────────────────────────────────────────────
    #  QUALITY-BASED CANDIDATE SELECTION (from Kharita/Roadster algorithms)
    # ──────────────────────────────────────────────────────────────────

    def apply_quality_selection(self):
        """
        Apply quality-based candidate selection to filter weak segments.
        
        Uses road-likeness scoring from research algorithms:
          - Support density (traces per meter)
          - Network connectivity (node degrees)
          - Dangling edge handling (weak dead-ends)
          - Source-aware weighting (VPD > Probe)
        
        This improves precision by removing false positives while
        preserving high-confidence road segments.
        """
        if not self.enable_quality_selection:
            print("  Quality selection disabled, skipping...")
            return
        
        if not QUALITY_SCORING_AVAILABLE:
            print("  Quality scoring module not available, using fallback...")
            self._fallback_quality_selection()
            return
        
        print(f"  Applying quality-based candidate selection...")
        print(f"    Threshold: {self.quality_selection_threshold}")
        print(f"    Force-keep support: {self.quality_force_keep_support}")
        
        before = len(self.gdf)
        
        # Configure quality scoring
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
        
        # Apply quality scoring
        scored_gdf = enhance_segments_with_quality(self.gdf, config)
        
        # Filter by candidate selection
        if "is_candidate" in scored_gdf.columns:
            # Always keep VPD and roundabouts
            has_source = "source" in scored_gdf.columns
            if has_source:
                keep_mask = (
                    scored_gdf["is_candidate"] | 
                    (scored_gdf["source"] == "VPD") |
                    (scored_gdf["source"] == "roundabout")
                )
            else:
                keep_mask = scored_gdf["is_candidate"]
            
            self.gdf = scored_gdf[keep_mask].reset_index(drop=True)
            
            # Log selection reasons
            if "selection_reason" in scored_gdf.columns:
                removed = scored_gdf[~keep_mask]
                reason_counts = removed["selection_reason"].value_counts()
                for reason, count in reason_counts.items():
                    print(f"      Removed {count} segments: {reason}")
        else:
            self.gdf = scored_gdf
        
        removed = before - len(self.gdf)
        print(f"    Quality selection: {before} -> {len(self.gdf)} segments ({removed} removed)")
    
    def _fallback_quality_selection(self):
        """
        Fallback quality selection when module is not available.
        
        Uses simple heuristics:
          - Remove short dangling probe segments
          - Remove very short isolated segments
        """
        from collections import Counter
        
        before = len(self.gdf)
        has_source = "source" in self.gdf.columns
        
        # Compute node degrees
        def round_coord(c):
            return (round(c[0], 1), round(c[1], 1))
        
        degree = Counter()
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString) and len(geom.coords) >= 2:
                start = round_coord(geom.coords[0])
                end = round_coord(geom.coords[-1])
                degree[start] += 1
                degree[end] += 1
        
        # Filter by quality heuristics
        keep_mask = []
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            length = geom.length if isinstance(geom, LineString) else 0
            source = row.get("source", "VPD") if has_source else "VPD"
            
            # Always keep VPD and roundabouts
            if source == "VPD" or source == "roundabout":
                keep_mask.append(True)
                continue
            
            # Check if dangling
            if isinstance(geom, LineString) and len(geom.coords) >= 2:
                start = round_coord(geom.coords[0])
                end = round_coord(geom.coords[-1])
                is_dangling = degree[start] <= 1 or degree[end] <= 1
            else:
                is_dangling = True
            
            # Drop weak dangling probe segments
            if is_dangling and length < self.quality_dangling_max_length_m:
                keep_mask.append(False)
            elif length < 8.0:  # Very short segments
                keep_mask.append(False)
            else:
                keep_mask.append(True)
        
        self.gdf = self.gdf[keep_mask].reset_index(drop=True)
        removed = before - len(self.gdf)
        print(f"    Fallback quality filter: removed {removed} weak segments")

    # ──────────────────────────────────────────────────────────────────
    #  ISOLATED PROBE FILTER — remove probe lines not connected to VPD
    # ──────────────────────────────────────────────────────────────────

    def filter_isolated_probes(self):
        """
        Remove probe-sourced lines that are isolated from the VPD network.
        
        A probe line is 'isolated' if it's not within snap distance of any
        VPD segment. Isolated probes are likely GPS noise / non-road paths.
        """
        if "source" not in self.gdf.columns:
            return

        vpd_mask = self.gdf["source"] != "Probe"
        probe_mask = self.gdf["source"] == "Probe"
        n_probes_before = probe_mask.sum()

        if n_probes_before == 0 or vpd_mask.sum() == 0:
            return

        print(f"  Filtering isolated probe segments...")
        vpd_lines = self.gdf[vpd_mask]
        probe_lines = self.gdf[probe_mask]

        # Build spatial index on VPD lines
        vpd_tree = STRtree(vpd_lines.geometry.values)

        keep_probe_mask = []
        for geom in probe_lines.geometry:
            # Check if any VPD line is within 25m of this probe line
            hits = vpd_tree.query(geom, predicate="dwithin", distance=25.0)
            keep_probe_mask.append(len(hits) > 0)

        # Also require probe lines to meet minimum length
        for i, (idx, row) in enumerate(probe_lines.iterrows()):
            if keep_probe_mask[i] and row.geometry.length < self.probe_min_length:
                keep_probe_mask[i] = False

        kept_probes = probe_lines[keep_probe_mask]
        removed = n_probes_before - len(kept_probes)

        self.gdf = gpd.GeoDataFrame(
            pd.concat([vpd_lines, kept_probes], ignore_index=True),
            crs=self.local_crs,
        )
        print(f"    Removed {removed} isolated/short probe segments "
              f"({n_probes_before} -> {len(kept_probes)})")

    # ──────────────────────────────────────────────────────────────────
    #  CLEANING
    # ──────────────────────────────────────────────────────────────────

    def clean_geometries(self):
        """Fix invalid geometries, remove short/empty segments."""
        print("  Cleaning geometries...")
        before = len(self.gdf)

        self.gdf = self.gdf[~self.gdf.geometry.is_empty].copy()

        # Fix invalid
        invalid_mask = ~self.gdf.geometry.is_valid
        if invalid_mask.sum() > 0:
            self.gdf.loc[invalid_mask, "geometry"] = (
                self.gdf.loc[invalid_mask, "geometry"].buffer(0)
            )

        # Keep LineStrings only
        self.gdf = self.gdf[
            self.gdf.geom_type.isin(["LineString", "MultiLineString"])
        ].copy()

        # Remove very short
        self.gdf = self.gdf[self.gdf.geometry.length >= self.min_length].copy()
        self.gdf = self.gdf.reset_index(drop=True)

        after = len(self.gdf)
        print(f"    Cleaned: {before} -> {after} segments")

    # ──────────────────────────────────────────────────────────────────
    #  EXPORT
    # ──────────────────────────────────────────────────────────────────

    def export(self):
        self.clean_geometries()
        self.final_gdf = self.gdf.copy()

        output_path = os.path.join(self.output_dir, "final_centerline_output.gpkg")
        print(f"  Exporting to {output_path}...")

        # Save in local CRS
        self.final_gdf.to_file(output_path, driver="GPKG")

        # Also save as WGS84
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
        print("  PHASE 5: Geometry Optimization & Topology Cleanup")
        print("=" * 60)
        t0 = time.time()

        self.load_data()
        # DISABLED: filter_isolated_probes() was destroying valid gap-filling segments
        # self.filter_isolated_probes()
        self.smooth_geometry()
        self.remove_redundancy()
        self.prune_stubs()
        
        # NEW: Quality-based candidate selection (from Kharita/Roadster algorithms)
        self.apply_quality_selection()
        
        self.export()

        elapsed = time.time() - t0
        total_km = self.final_gdf.geometry.length.sum() / 1000.0
        print(f"\nPhase 5 complete in {elapsed:.1f}s")
        print(f"  Final network: {len(self.final_gdf)} segments, {total_km:.1f} km")

        return self.final_gdf


if __name__ == "__main__":
    INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "merged_network_phase4.gpkg")
    optimizer = GeometryOptimizer(INPUT_FILE)
    optimizer.run()
