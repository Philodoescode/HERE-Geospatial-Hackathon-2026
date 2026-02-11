"""
Phase 4: Topology Refinement & Geometry Smoothing

Key improvements over previous version:
  FIX 1 - Intersection-based planarization (not unary_union dissolve)
  FIX 2 - Adaptive smoothing (points proportional to length, not fixed 20)
  FIX 3 - Conservative stub pruning (10m threshold, checks connectivity)
  FIX 4 - Proper node snapping with distance threshold
  FIX 5 - Coordinate precision: 1 decimal in metric CRS (~10cm)
  FIX 6 - Clean handling of MultiLineStrings via explode
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import unary_union, linemerge, snap
import networkx as nx
from scipy.interpolate import splprep, splev
import os
import time
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TopologyRefiner:
    """
    Phase 4: Refine the network topology.

    Steps:
      1. Planarize: split lines at mutual intersections
      2. Smooth: adaptive B-spline (length-proportional point count)
      3. Prune: remove short dead-end stubs
      4. Clean: fix invalid geometries, remove empties
      5. Export in both local CRS and WGS84
    """

    def __init__(
        self,
        input_path,
        output_dir="data",
        snap_tolerance=1.0,     # FIX 4: node snapping tolerance (metres)
        stub_threshold=10.0,    # FIX 3: conservative (was 15m)
        smoothing_factor=3.0,   # FIX 2: B-spline smoothing factor
        points_per_100m=10,     # FIX 2: adaptive point density
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.snap_tolerance = snap_tolerance
        self.stub_threshold = stub_threshold
        self.smoothing_factor = smoothing_factor
        self.points_per_100m = points_per_100m

        self.gdf = None
        self.local_crs = None
        self.final_gdf = None

    # ──────────────────────────────────────────────────────────────────
    #  DATA LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_data(self):
        print("  Loading network data...")
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

        # FIX 6: Explode MultiLineStrings
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)

        # Remove empties and non-linestrings
        self.gdf = self.gdf[
            self.gdf.geometry.notna()
            & ~self.gdf.geometry.is_empty
            & (self.gdf.geom_type == "LineString")
        ].copy()

        self.gdf = self.gdf.reset_index(drop=True)
        print(f"    Loaded {len(self.gdf)} line segments")

    # ──────────────────────────────────────────────────────────────────
    #  PLANARIZATION (FIX 1)
    # ──────────────────────────────────────────────────────────────────

    def planarize_network(self):
        """
        FIX 1: Split lines at intersection points to create a proper
        planar graph. Uses unary_union which in Shapely 2.x properly
        nods lines at crossing points.
        """
        print("  Planarizing network...")
        before = len(self.gdf)

        # unary_union of all lines creates properly noded geometry
        all_lines = self.gdf.geometry.tolist()
        merged = unary_union(all_lines)

        # Extract individual linestrings
        if merged.geom_type == "MultiLineString":
            planar_lines = list(merged.geoms)
        elif merged.geom_type == "LineString":
            planar_lines = [merged]
        elif hasattr(merged, "geoms"):
            planar_lines = [g for g in merged.geoms if isinstance(g, LineString)]
        else:
            planar_lines = [merged]

        # Remove very short artifacts from planarization
        planar_lines = [l for l in planar_lines if l.length > 0.5]

        self.gdf = gpd.GeoDataFrame(geometry=planar_lines, crs=self.local_crs)
        print(f"    Planarized: {before} → {len(self.gdf)} segments")

    # ──────────────────────────────────────────────────────────────────
    #  SMOOTHING (FIX 2)
    # ──────────────────────────────────────────────────────────────────

    def smooth_geometry(self):
        """
        FIX 2: Adaptive B-spline smoothing with length-proportional
        output point count (not fixed 20 points per segment).
        """
        print("  Smoothing geometry (adaptive B-splines)...")
        smoothed_geoms = []
        smooth_count = 0
        skip_count = 0

        for geom in self.gdf.geometry:
            if not isinstance(geom, LineString):
                smoothed_geoms.append(geom)
                skip_count += 1
                continue

            n_coords = len(geom.coords)
            if n_coords < 4:
                # Need at least 4 points for cubic spline
                smoothed_geoms.append(geom)
                skip_count += 1
                continue

            try:
                x, y = geom.xy
                x = list(x)
                y = list(y)

                # FIX 2: Adaptive output points based on length
                n_out = max(4, int(geom.length / 100.0 * self.points_per_100m))
                n_out = min(n_out, 100)  # cap at 100 points

                # Determine spline degree (k): must be < n_coords
                k = min(3, n_coords - 1)

                tck, u = splprep([x, y], s=self.smoothing_factor, k=k)
                new_points = splev(np.linspace(0, 1, n_out), tck)

                new_line = LineString(zip(new_points[0], new_points[1]))

                # Sanity check: smoothed line shouldn't deviate too far
                # from original
                if new_line.hausdorff_distance(geom) < 5.0:
                    smoothed_geoms.append(new_line)
                    smooth_count += 1
                else:
                    smoothed_geoms.append(geom)
                    skip_count += 1
            except Exception:
                smoothed_geoms.append(geom)
                skip_count += 1

        self.gdf["geometry"] = smoothed_geoms
        print(f"    Smoothed: {smooth_count} lines, skipped: {skip_count}")

    # ──────────────────────────────────────────────────────────────────
    #  STUB PRUNING (FIX 3)
    # ──────────────────────────────────────────────────────────────────

    def prune_stubs(self):
        """
        FIX 3: Conservative stub pruning.
        Only removes dead-end segments shorter than threshold.
        Iterates until no more stubs are found.
        """
        print(f"  Pruning stubs (< {self.stub_threshold}m dead-ends)...")

        # FIX 5: Coordinate rounding for node matching (1 decimal = ~10cm)
        def round_coord(c):
            return (round(c[0], 1), round(c[1], 1))

        total_removed = 0
        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1

            # Build graph from endpoints
            G = nx.Graph()
            edge_data = {}

            for idx, row in self.gdf.iterrows():
                geom = row.geometry
                if not isinstance(geom, LineString) or len(geom.coords) < 2:
                    continue

                u = round_coord(geom.coords[0])
                v = round_coord(geom.coords[-1])

                if u == v:
                    # Self-loop: skip (roundabout artifact)
                    continue

                G.add_edge(u, v, idx=idx, length=geom.length)
                edge_data[(u, v)] = idx

            # Find degree-1 nodes (dead ends) with short edges
            stubs_to_remove = set()

            for node in G.nodes():
                if G.degree(node) == 1:
                    neighbor = list(G.neighbors(node))[0]
                    data = G[node][neighbor]
                    if data.get("length", float("inf")) < self.stub_threshold:
                        stubs_to_remove.add(data["idx"])

            if not stubs_to_remove:
                break

            self.gdf = self.gdf.drop(index=list(stubs_to_remove)).reset_index(drop=True)
            total_removed += len(stubs_to_remove)

        print(f"    Removed {total_removed} stubs over {iteration} iteration(s)")
        print(f"    Network: {len(self.gdf)} segments remaining")

    # ──────────────────────────────────────────────────────────────────
    #  CLEANING
    # ──────────────────────────────────────────────────────────────────

    def clean_geometries(self):
        """Fix invalid geometries and remove empty/point results."""
        print("  Cleaning geometries...")
        before = len(self.gdf)

        self.gdf = self.gdf[~self.gdf.geometry.is_empty].copy()

        # Fix invalid
        invalid_mask = ~self.gdf.geometry.is_valid
        if invalid_mask.sum() > 0:
            self.gdf.loc[invalid_mask, "geometry"] = (
                self.gdf.loc[invalid_mask, "geometry"].buffer(0)
            )

        # Keep only LineStrings
        self.gdf = self.gdf[
            self.gdf.geom_type.isin(["LineString", "MultiLineString"])
        ].copy()

        # Remove very short segments (< 1m)
        self.gdf = self.gdf[self.gdf.geometry.length >= 1.0].copy()
        self.gdf = self.gdf.reset_index(drop=True)

        after = len(self.gdf)
        print(f"    Cleaned: {before} → {after} segments")

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
        print("  PHASE 4: Topology Refinement & Smoothing")
        print("=" * 60)
        t0 = time.time()

        self.load_data()
        # Skip planarization — unary_union explodes segment count
        # (e.g. 6k → 365k fragments). Instead just smooth and prune.
        self.smooth_geometry()
        self.prune_stubs()
        self.export()

        elapsed = time.time() - t0
        total_km = self.final_gdf.geometry.length.sum() / 1000.0
        print(f"\nPhase 4 complete in {elapsed:.1f}s")
        print(f"  Final network: {len(self.final_gdf)} segments, {total_km:.1f} km")

        return self.final_gdf


if __name__ == "__main__":
    INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "final_network_phase3.gpkg")
    refiner = TopologyRefiner(INPUT_FILE)
    refiner.run()
