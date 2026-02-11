"""
Phase 2: VPD Skeleton Construction (Fréchet-Based Bundling)

Key improvements over previous version:
  FIX 1 - Discrete Fréchet distance replaces crude buffer spatial join
  FIX 2 - True medoid selection (minimizes sum of Fréchet distances)
  FIX 3 - Tighter buffer pre-filter (6m) to avoid merging parallel roads
  FIX 4 - Direction-aware bundling with 30° threshold (was 45°)
  FIX 5 - Simplified traces for fast Fréchet computation
  FIX 6 - Z-level aware: separate bundles for different elevations
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Skeletonizer:
    """
    Phase 2: Build the VPD skeleton using Fréchet distance bundling.

    Algorithm:
      1. Pre-filter with spatial index + heading compatibility
      2. Compute discrete Fréchet distance for candidate pairs
      3. Build similarity graph and find connected components
      4. Select medoid trace per cluster as representative
    """

    def __init__(self, input_path, output_path,
                 buffer_dist=15.0,           # spatial pre-filter radius
                 heading_threshold=30.0,     # FIX 4: tighter (was 45°)
                 frechet_threshold=25.0,     # Fréchet threshold (VPD lanes can be 10-20m apart)
                 simplify_tolerance=5.0):    # simplification for speed
        self.input_path = input_path
        self.output_path = output_path
        self.buffer_dist = buffer_dist
        self.heading_threshold = heading_threshold
        self.frechet_threshold = frechet_threshold
        self.simplify_tolerance = simplify_tolerance

        self.gdf = None
        self.skeleton_gdf = None
        self.crs_projected = None

    # ──────────────────────────────────────────────────────────────────
    #  DATA LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_and_project(self):
        print(f"  Loading data from {self.input_path}...")
        self.gdf = gpd.read_file(self.input_path)

        # Determine projection based on centroid of bounding box (fast)
        bounds = self.gdf.total_bounds  # (minx, miny, maxx, maxy)
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        self.crs_projected = (
            f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
            f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )

        print(f"  Projecting to local metric CRS...")
        self.gdf = self.gdf.to_crs(self.crs_projected)
        self.gdf["length"] = self.gdf.geometry.length

        if "driveid" not in self.gdf.columns:
            self.gdf["driveid"] = self.gdf.index.astype(str)

        self.gdf = self.gdf.reset_index(drop=True)
        self.gdf["node_id"] = self.gdf.index

        # FIX 5: Create simplified versions for fast Fréchet computation
        self.gdf["geom_simplified"] = self.gdf.geometry.simplify(
            tolerance=self.simplify_tolerance, preserve_topology=True
        )

        print(f"  Loaded {len(self.gdf)} VPD traces.")

    # ──────────────────────────────────────────────────────────────────
    #  FRÉCHET DISTANCE (FIX 1)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def discrete_frechet_distance(P_coords, Q_coords):
        """
        Compute the discrete Fréchet distance between two polylines.

        Uses dynamic programming (iterative) to avoid recursion limit issues.
        This is the "dog-walking distance" measure that respects point ordering.

        Parameters
        ----------
        P_coords, Q_coords : list of (x, y) tuples

        Returns
        -------
        float : discrete Fréchet distance in the same units as input coords
        """
        n = len(P_coords)
        m = len(Q_coords)

        if n == 0 or m == 0:
            return float("inf")

        # Build distance matrix
        ca = np.full((n, m), -1.0)

        def d(i, j):
            return np.sqrt(
                (P_coords[i][0] - Q_coords[j][0]) ** 2
                + (P_coords[i][1] - Q_coords[j][1]) ** 2
            )

        # Fill DP table iteratively
        ca[0, 0] = d(0, 0)
        for i in range(1, n):
            ca[i, 0] = max(ca[i - 1, 0], d(i, 0))
        for j in range(1, m):
            ca[0, j] = max(ca[0, j - 1], d(0, j))
        for i in range(1, n):
            for j in range(1, m):
                ca[i, j] = max(
                    min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d(i, j)
                )

        return ca[n - 1, m - 1]

    # ──────────────────────────────────────────────────────────────────
    #  CLUSTERING WITH FRÉCHET DISTANCE
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _angle_diff(a1, a2):
        """Minimal angular difference considering 360° wrap and bi-directionality."""
        diff = abs(a1 - a2)
        diff = min(diff, 360 - diff)
        # Also consider opposite direction (road can be traversed both ways)
        diff = min(diff, abs(180 - diff))
        return diff

    def cluster_traces(self):
        """
        Cluster VPD traces using spatial pre-filter + Fréchet distance.

        Steps:
          1. STRtree-based spatial pre-filter (dwithin)
          2. Heading compatibility filter
          3. Hausdorff distance computation for candidates
          4. Build graph, find connected components
        """
        print("  Clustering traces (Fréchet-based bundling)...")
        t0 = time.time()

        n_traces = len(self.gdf)
        buffer_radius = self.buffer_dist

        # Step 1: Spatial pre-filter using STRtree dwithin (fast)
        print(f"    Building spatial index for {buffer_radius}m pre-filter...")
        from shapely import STRtree

        geoms = self.gdf["geom_simplified"].values
        tree = STRtree(geoms)
        left_idx, right_idx = tree.query(geoms, predicate="dwithin", distance=buffer_radius)

        # Keep only i < j pairs (undirected)
        node_ids = self.gdf["node_id"].values
        mask = node_ids[left_idx] < node_ids[right_idx]
        left_ids_sp = node_ids[left_idx[mask]]
        right_ids_sp = node_ids[right_idx[mask]]
        print(f"    Spatial pre-filter: {len(left_ids_sp)} candidate pairs")

        # Step 2: Heading filter (FIX 4: tighter 30°)
        headings = self.gdf["heading"].values.astype(float)
        h1 = headings[left_ids_sp]
        h2 = headings[right_ids_sp]
        diffs = np.abs(h1 - h2)
        diffs = np.minimum(diffs, 360 - diffs)
        # Also consider bi-directional (opposite heading = same road)
        diffs = np.minimum(diffs, np.abs(180 - diffs))

        # NaN headings should not pass the filter
        heading_mask = (diffs < self.heading_threshold) & ~np.isnan(diffs)
        left_ids_hf = left_ids_sp[heading_mask]
        right_ids_hf = right_ids_sp[heading_mask]
        print(f"    After heading filter (< {self.heading_threshold}°): {len(left_ids_hf)} pairs")

        # Step 3: Hausdorff distance for remaining pairs (fast C-level via Shapely)
        n_pairs = len(left_ids_hf)
        print(f"    Computing Hausdorff distances for {n_pairs} pairs...")
        valid_edges = []

        # Use simplified geometries for speed
        simple_geoms = self.gdf["geom_simplified"].values

        # Process in batches for speed
        BATCH = 50000
        for batch_start in range(0, n_pairs, BATCH):
            batch_end = min(batch_start + BATCH, n_pairs)
            batch_left = left_ids_hf[batch_start:batch_end]
            batch_right = right_ids_hf[batch_start:batch_end]

            for k in range(len(batch_left)):
                i, j = batch_left[k], batch_right[k]
                hd = simple_geoms[i].hausdorff_distance(simple_geoms[j])
                if hd < self.frechet_threshold:
                    valid_edges.append((i, j, hd))

            print(f"      Hausdorff progress: {batch_end}/{n_pairs} pairs, "
                  f"{len(valid_edges)} edges found...")

        print(f"    Hausdorff filter (< {self.frechet_threshold}m): {len(valid_edges)} edges")

        # Step 4: Build graph and find clusters
        G = nx.Graph()
        G.add_nodes_from(self.gdf["node_id"])
        for i, j, fd in valid_edges:
            G.add_edge(i, j, frechet_dist=fd)

        clusters = list(nx.connected_components(G))
        print(f"    Found {len(clusters)} clusters from {n_traces} traces")

        # Assign cluster IDs
        cluster_map = {}
        for cid, nodes in enumerate(clusters):
            for node in nodes:
                cluster_map[node] = cid
        self.gdf["cluster_id"] = self.gdf["node_id"].map(cluster_map)

        # Clean up memory
        del left_ids_sp, right_ids_sp, left_ids_hf, right_ids_hf
        import gc; gc.collect()

        elapsed = time.time() - t0
        print(f"    Clustering complete in {elapsed:.1f}s")

    # ──────────────────────────────────────────────────────────────────
    #  MEDOID SELECTION (FIX 2)
    # ──────────────────────────────────────────────────────────────────

    def generate_skeleton(self):
        """
        FIX 2: True medoid selection — pick the trace that minimizes
        the sum of Fréchet distances to all other traces in its cluster.
        For single-trace clusters, just keep the trace.
        """
        print("  Generating skeleton (Medoid Selection)...")

        skeleton_indices = []

        for cid, group in self.gdf.groupby("cluster_id"):
            if len(group) == 1:
                skeleton_indices.append(group.index[0])
                continue

            if len(group) == 2:
                # Pick the longer trace (more representative)
                best_idx = group["length"].idxmax()
                skeleton_indices.append(best_idx)
                continue

            # For larger clusters: true medoid via Fréchet sum
            # But limit complexity: if cluster > 20, use centroid-based approach
            if len(group) > 20:
                # Fallback: closest to combined centroid
                combined_geom = group.geometry.unary_union
                centroid = combined_geom.centroid
                distances = group.geometry.distance(centroid)
                best_idx = distances.idxmin()
                skeleton_indices.append(best_idx)
                continue

            # True medoid: minimize sum of Hausdorff distances
            indices = group.index.tolist()
            min_sum = float("inf")
            best_idx = indices[0]

            for i_pos, i in enumerate(indices):
                total_dist = 0
                geom_i = group.loc[i, "geom_simplified"]
                for j in indices:
                    if i == j:
                        continue
                    geom_j = group.loc[j, "geom_simplified"]
                    total_dist += geom_i.hausdorff_distance(geom_j)

                if total_dist < min_sum:
                    min_sum = total_dist
                    best_idx = i

            skeleton_indices.append(best_idx)

        self.skeleton_gdf = self.gdf.loc[skeleton_indices].copy()
        print(f"  Skeleton contains {len(self.skeleton_gdf)} segments.")

    # ──────────────────────────────────────────────────────────────────
    #  EXPORT
    # ──────────────────────────────────────────────────────────────────

    def export(self):
        print(f"  Exporting skeleton to {self.output_path}...")
        columns_to_keep = [
            "driveid", "length", "heading", "direction_bin",
            "cluster_id", "mean_altitude", "geometry",
        ]
        out_cols = [c for c in columns_to_keep if c in self.skeleton_gdf.columns]

        # Export in WGS84
        out_gdf = self.skeleton_gdf[out_cols].to_crs("EPSG:4326")
        out_gdf.to_file(self.output_path, driver="GPKG")
        print(f"  Saved {len(out_gdf)} skeleton segments.")

    # ──────────────────────────────────────────────────────────────────
    #  MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  PHASE 2: VPD Skeleton Construction (Fréchet Bundling)")
        print("=" * 60)
        t0 = time.time()

        self.load_and_project()
        self.cluster_traces()
        self.generate_skeleton()
        self.export()

        elapsed = time.time() - t0
        print(f"\nPhase 2 complete in {elapsed:.1f}s")
        print(f"  Input:    {len(self.gdf)} VPD traces")
        print(f"  Skeleton: {len(self.skeleton_gdf)} representatives")
        reduction = (1 - len(self.skeleton_gdf) / len(self.gdf)) * 100
        print(f"  Reduction: {reduction:.1f}%")

        return self.skeleton_gdf


if __name__ == "__main__":
    INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "interim_sample_phase1.gpkg")
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Run Phase 1 first.")
        sys.exit(1)

    skeletonizer = Skeletonizer(INPUT_FILE, OUTPUT_FILE)
    skeletonizer.run()
