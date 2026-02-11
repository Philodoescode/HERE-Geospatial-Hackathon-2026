"""
Phase 3: Probe Recovery via KDE Rasterization + Transition Enhancement

Replaces DBSCAN midpoint clustering with rasterization + transition
line-drawing + KDE + skeletonization. Recovers residential/sparse roads
that the VPD skeleton misses.

Algorithm:
  1. Load VPD skeleton and HPD probes
  2. Exclude probe portions already covered by VPD (buffer-based difference)
  3. Rasterize residual probe segments onto a density grid
  4. Transition enhancement: Bresenham line drawing between consecutive
     probe points to "connect the dots" for sparse traces
  5. Speed-based weighting (higher speed -> more likely a real road)
  6. Gaussian blur + threshold + skeletonize + vectorize
  7. Filter short/noisy segments
"""

import gc
import math
import os
import sys
import time
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, binary_closing
from shapely import get_coordinates, segmentize as shp_segmentize
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, substring
from skimage.morphology import skeletonize, disk
from skimage.draw import line as bresenham_line

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProbeRecovery:
    """
    Phase 3: Recover road segments from HPD probe data that the VPD
    skeleton missed, using KDE rasterization with transition enhancement.
    """

    def __init__(
        self,
        skeleton_path,
        hpd_path,
        output_dir="data",
        skeleton_buffer_m=15.0,      # moderate buffer (was 8 original, 20 too aggressive)
        pixel_size=3.0,              # coarser than VPD (GPS noise tolerance)
        blur_sigma=4.0,              # moderate blur
        threshold_frac=0.005,        # moderate threshold (was 0.003 original, 0.008 too strict)
        min_segment_length=20.0,     # moderate minimum (was 5 original, 30 too strict)
        grid_buffer=30.0,
        transition_weight=1.5,       # reduced transition weight (was 2.0)
        min_probe_count=2,           # require at least 2 probe traces (was 3)
    ):
        self.skeleton_path = skeleton_path
        self.hpd_path = hpd_path
        self.output_dir = output_dir
        self.skeleton_buffer_m = skeleton_buffer_m
        self.pixel_size = pixel_size
        self.blur_sigma = blur_sigma
        self.threshold_frac = threshold_frac
        self.min_segment_length = min_segment_length
        self.grid_buffer = grid_buffer
        self.transition_weight = transition_weight
        self.min_probe_count = min_probe_count

        self.local_crs = None
        self.skeleton = None
        self.probes = None
        self.probe_skeleton_gdf = None

    # ──────────────────────────────────────────────────────────────────
    #  DATA LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_data(self):
        """Load VPD skeleton and HPD probes."""
        print("  Loading data...")

        # Skeleton
        if not os.path.exists(self.skeleton_path):
            raise FileNotFoundError(f"Skeleton not found: {self.skeleton_path}")
        self.skeleton = gpd.read_file(self.skeleton_path)

        if self.skeleton.crs is None or self.skeleton.crs.is_geographic:
            bounds = self.skeleton.total_bounds
            cx = (bounds[0] + bounds[2]) / 2.0
            cy = (bounds[1] + bounds[3]) / 2.0
            self.local_crs = (
                f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
                f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            )
            self.skeleton = self.skeleton.to_crs(self.local_crs)
        else:
            self.local_crs = self.skeleton.crs

        print(f"    Skeleton: {len(self.skeleton)} segments")

        # HPD probes
        if not os.path.exists(self.hpd_path):
            print(f"    WARNING: HPD not found at {self.hpd_path}")
            self._load_hpd_from_csvs()
        else:
            self.probes = gpd.read_file(self.hpd_path)
            if self.probes.crs != self.local_crs:
                self.probes = self.probes.to_crs(self.local_crs)
            print(f"    Probes: {len(self.probes)} traces")

    def _load_hpd_from_csvs(self):
        """Fallback: load HPD from raw CSVs."""
        hpd_dir = os.path.join(PROJECT_ROOT, "data", "Kosovo_HPD")
        files = [
            os.path.join(hpd_dir, "XKO_HPD_week_1.csv"),
            os.path.join(hpd_dir, "XKO_HPD_week_2.csv"),
        ]
        all_probes = []
        for probe_path in files:
            if not os.path.exists(probe_path):
                continue
            df = pd.read_csv(probe_path)
            df.sort_values(by=["traceid", "time"], inplace=True)
            if "speed" in df.columns:
                trace_avg = df.groupby("traceid")["speed"].mean()
                fast = trace_avg[trace_avg >= 5.0].index
                df = df[df["traceid"].isin(fast)]
            for tid, group in df.groupby("traceid"):
                if len(group) < 2:
                    continue
                coords = list(zip(group["longitude"], group["latitude"]))
                geom = LineString(coords)
                if not geom.is_empty:
                    avg_spd = group["speed"].mean() if "speed" in group.columns else None
                    all_probes.append({"traceid": tid, "geometry": geom, "avg_speed": avg_spd})
        self.probes = gpd.GeoDataFrame(all_probes, crs="EPSG:4326").to_crs(self.local_crs)
        print(f"    Probes (CSV fallback): {len(self.probes)} traces")

    # ──────────────────────────────────────────────────────────────────
    #  GAP EXTRACTION (reused from old Phase 3 — efficient)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _explode_to_linestrings(geom):
        """Extract LineStrings from any geometry type."""
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, LineString):
            return [geom]
        if isinstance(geom, MultiLineString):
            return list(geom.geoms)
        return [g for g in geom.geoms if isinstance(g, LineString)]

    def extract_residual_probes(self):
        """
        Remove probe portions already covered by VPD skeleton.
        Uses spatial-index-based approach for memory efficiency.
        """
        print(f"  Extracting probe residuals (excluding {self.skeleton_buffer_m}m VPD buffer)...")

        skel_buffered = self.skeleton.geometry.buffer(self.skeleton_buffer_m)
        skel_buf_gdf = gpd.GeoDataFrame(geometry=skel_buffered.values, crs=self.local_crs)
        skel_sindex = skel_buf_gdf.sindex

        residual_traces = []   # (geometry, avg_speed) tuples
        n_probes = len(self.probes)

        for i in range(n_probes):
            row = self.probes.iloc[i]
            probe_geom = row.geometry
            if not isinstance(probe_geom, (LineString, MultiLineString)):
                continue

            avg_speed = row.get("avg_speed", None)

            # Speed gate: reject very slow probes (likely pedestrian/parking)
            if avg_speed is not None and not np.isnan(avg_speed) and avg_speed < 10:
                continue

            # Find nearby skeleton buffers
            candidate_idxs = list(skel_sindex.intersection(probe_geom.bounds))
            if not candidate_idxs:
                # No nearby skeleton — keep entire probe but require minimum length
                for line in self._explode_to_linestrings(probe_geom):
                    if line.length >= 20.0:
                        residual_traces.append((line, avg_speed))
                continue

            try:
                local_buffer = unary_union(skel_buffered.iloc[candidate_idxs].values)
                diff = probe_geom.difference(local_buffer)
            except Exception:
                continue

            for line in self._explode_to_linestrings(diff):
                if line.length >= 20.0:  # require longer residual segments
                    residual_traces.append((line, avg_speed))

            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{n_probes} probes, "
                      f"residuals: {len(residual_traces)}")

        print(f"    Extracted {len(residual_traces)} residual segments from "
              f"{n_probes} probes")
        return residual_traces

    # ──────────────────────────────────────────────────────────────────
    #  RASTERIZATION + TRANSITION ENHANCEMENT
    # ──────────────────────────────────────────────────────────────────

    def rasterize_probes(self, residual_traces):
        """
        Rasterize residual probe segments with transition enhancement.
        
        Two passes:
          1. Point density: accumulate segment midpoints (like VPD Phase 2)
          2. Transition lines: Bresenham line drawing between consecutive
             points within each trace to "connect the dots"
        """
        print("  Rasterizing probe traces with transition enhancement...")
        t0 = time.time()

        if not residual_traces:
            return None, 0, 0, 0, 0

        # Compute grid extent from residual geometries
        all_geoms = [t[0] for t in residual_traces]
        all_bounds = np.array([g.bounds for g in all_geoms])
        minx = all_bounds[:, 0].min() - self.grid_buffer
        miny = all_bounds[:, 1].min() - self.grid_buffer
        maxx = all_bounds[:, 2].max() + self.grid_buffer
        maxy = all_bounds[:, 3].max() + self.grid_buffer

        width = int(np.ceil((maxx - minx) / self.pixel_size))
        height = int(np.ceil((maxy - miny) / self.pixel_size))
        print(f"    Grid: {width} x {height} pixels ({self.pixel_size}m resolution)")

        # Safety check
        if width * height > 50_000_000:
            print("    WARNING: Grid too large, increasing pixel size")
            self.pixel_size = 5.0
            width = int(np.ceil((maxx - minx) / self.pixel_size))
            height = int(np.ceil((maxy - miny) / self.pixel_size))

        grid = np.zeros((height, width), dtype=np.float32)

        # Pass 1: Point density + transition lines
        for geom, avg_speed in residual_traces:
            coords = np.array(geom.coords)
            if len(coords) < 2:
                continue

            # Speed-based weight
            weight = 1.0
            if avg_speed is not None and not np.isnan(avg_speed):
                if avg_speed >= 30:
                    weight = 2.0  # likely a real road
                elif avg_speed < 10:
                    weight = 0.5  # possibly walking/parking

            # Point accumulation
            cols = ((coords[:, 0] - minx) / self.pixel_size).astype(np.intp)
            rows = ((maxy - coords[:, 1]) / self.pixel_size).astype(np.intp)

            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            r_valid = rows[valid]
            c_valid = cols[valid]
            np.add.at(grid, (r_valid, c_valid), weight)

            # Transition line drawing (Bresenham) between consecutive points
            # This "connects the dots" for sparse traces
            for j in range(len(coords) - 1):
                r0, c0 = rows[j], cols[j]
                r1, c1 = rows[j + 1], cols[j + 1]
                if (0 <= r0 < height and 0 <= c0 < width and
                    0 <= r1 < height and 0 <= c1 < width):
                    rr, cc = bresenham_line(r0, c0, r1, c1)
                    # Clip to grid bounds
                    mask = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                    np.add.at(grid, (rr[mask], cc[mask]),
                              weight * self.transition_weight)

        elapsed = time.time() - t0
        print(f"    Rasterized {len(residual_traces)} traces in {elapsed:.1f}s")

        return grid, width, height, minx, maxy

    def kde_and_skeletonize(self, grid, width, height):
        """Gaussian blur, threshold, morphological cleanup, skeletonize."""
        print("  Applying KDE + skeletonization on probe grid...")

        # Require minimum density: at least min_probe_count hits per pixel
        # This eliminates single-pass noise before blurring
        grid[grid < self.min_probe_count] = 0
        print(f"    Pixels above min count ({self.min_probe_count}): {(grid > 0).sum()}")

        # Gaussian blur
        density = gaussian_filter(grid, sigma=self.blur_sigma)
        del grid
        gc.collect()

        # Normalize
        d_max = density.max()
        if d_max > 0:
            density /= d_max

        # Threshold
        binary = density > self.threshold_frac
        print(f"    Binary road pixels: {binary.sum()} / {width * height}")

        # Morphological cleanup
        struct = disk(2)
        binary = binary_closing(binary, structure=struct)

        # Skeletonize
        skeleton = skeletonize(binary)
        n_skel = skeleton.sum()
        print(f"    Skeleton pixels: {n_skel}")

        del density, binary
        gc.collect()

        return skeleton

    def vectorize_skeleton(self, skeleton, minx, maxy):
        """Convert skeleton pixel mask to LineStrings via graph tracing."""
        print("  Vectorizing probe skeleton...")
        t0 = time.time()

        skel_rows, skel_cols = np.where(skeleton)
        if len(skel_rows) == 0:
            return []
        print(f"    {len(skel_rows)} skeleton pixels to trace")

        pixel_set = set(zip(skel_rows.tolist(), skel_cols.tolist()))
        G = nx.Graph()
        for r, c in pixel_set:
            G.add_node((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in pixel_set:
                        G.add_edge((r, c), nb)

        ps = self.pixel_size

        def px2world(r, c):
            return (minx + (c + 0.5) * ps, maxy - (r + 0.5) * ps)

        junctions = [n for n in G.nodes() if G.degree(n) > 2]
        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        terminals = set(junctions + endpoints)
        if not terminals and G.number_of_nodes() > 0:
            terminals = {next(iter(G.nodes()))}

        visited_edges = set()
        lines = []

        def _trace(start, neighbor):
            path = [start, neighbor]
            visited_edges.add((start, neighbor))
            visited_edges.add((neighbor, start))
            cur, prev = neighbor, start
            while cur not in terminals:
                nbrs = [
                    n for n in G.neighbors(cur)  # noqa: F821
                    if n != prev and (cur, n) not in visited_edges
                ]
                if not nbrs:
                    break
                nxt = nbrs[0]
                visited_edges.add((cur, nxt))
                visited_edges.add((nxt, cur))
                path.append(nxt)
                prev, cur = cur, nxt
            return path

        for t in terminals:
            for nb in list(G.neighbors(t)):
                if (t, nb) not in visited_edges:
                    path = _trace(t, nb)
                    if len(path) >= 2:
                        coords = [px2world(r, c) for r, c in path]
                        lines.append(LineString(coords))

        for u, v in list(G.edges()):
            if (u, v) not in visited_edges:
                path = _trace(u, v)
                if len(path) >= 2:
                    coords = [px2world(r, c) for r, c in path]
                    lines.append(LineString(coords))

        elapsed = time.time() - t0
        print(f"    Traced {len(lines)} raw lines in {elapsed:.1f}s")

        del G, pixel_set
        gc.collect()

        return lines

    def filter_lines(self, lines):
        """Remove short/noisy segments."""
        filtered = []
        for line in lines:
            if line.length >= self.min_segment_length:
                simple = line.simplify(self.pixel_size * 0.5, preserve_topology=True)
                if not simple.is_empty and simple.length >= self.min_segment_length:
                    filtered.append(simple)
        print(f"    Filtered: {len(lines)} -> {len(filtered)} lines "
              f"(min length {self.min_segment_length}m)")
        return filtered

    # ──────────────────────────────────────────────────────────────────
    #  EXPORT / MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  PHASE 3: Probe Recovery via KDE + Transition Enhancement")
        print("=" * 60)
        t0 = time.time()

        self.load_data()

        # Extract residual (uncovered) probe segments
        residual_traces = self.extract_residual_probes()

        if not residual_traces:
            print("  No residual probe segments found.")
            # Just output skeleton as-is
            output_path = os.path.join(self.output_dir, "interim_probe_skeleton_phase3.gpkg")
            self.skeleton.to_crs("EPSG:4326").to_file(output_path, driver="GPKG")
            self.probe_skeleton_gdf = gpd.GeoDataFrame(
                columns=["geometry"], crs=self.local_crs
            )
            return self.probe_skeleton_gdf

        # Rasterize with transition enhancement
        result = self.rasterize_probes(residual_traces)
        del residual_traces
        gc.collect()

        grid, width, height, minx, maxy = result
        if grid is None:
            self.probe_skeleton_gdf = gpd.GeoDataFrame(
                columns=["geometry"], crs=self.local_crs
            )
            return self.probe_skeleton_gdf

        # KDE + skeletonize
        skeleton = self.kde_and_skeletonize(grid, width, height)

        # Vectorize
        lines = self.vectorize_skeleton(skeleton, minx, maxy)
        del skeleton
        gc.collect()

        # Filter
        lines = self.filter_lines(lines)

        # Export probe skeleton
        output_path = os.path.join(self.output_dir, "interim_probe_skeleton_phase3.gpkg")
        print(f"  Exporting probe skeleton to {output_path}...")

        if lines:
            gdf = gpd.GeoDataFrame(geometry=lines, crs=self.local_crs)
            gdf["source"] = "Probe"
            gdf["length_m"] = gdf.geometry.length
            out_gdf = gdf.to_crs("EPSG:4326")
            out_gdf.to_file(output_path, driver="GPKG")
            self.probe_skeleton_gdf = out_gdf
        else:
            self.probe_skeleton_gdf = gpd.GeoDataFrame(
                columns=["geometry", "source"], crs="EPSG:4326"
            )
            self.probe_skeleton_gdf.to_file(output_path, driver="GPKG")

        elapsed = time.time() - t0
        probe_km = sum(l.length for l in lines) / 1000.0 if lines else 0
        skel_km = self.skeleton.geometry.length.sum() / 1000.0

        print(f"\nPhase 3 complete in {elapsed:.1f}s")
        print(f"  VPD skeleton: {len(self.skeleton)} segments ({skel_km:.1f} km)")
        print(f"  Probe recovery: {len(lines)} segments ({probe_km:.1f} km)")

        return self.probe_skeleton_gdf


if __name__ == "__main__":
    SKELETON_FILE = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
    HPD_FILE = os.path.join(PROJECT_ROOT, "data", "interim_hpd_phase1.gpkg")

    recovery = ProbeRecovery(SKELETON_FILE, HPD_FILE)
    recovery.run()
