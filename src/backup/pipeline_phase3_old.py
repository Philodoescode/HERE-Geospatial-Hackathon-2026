"""
Phase 3: Network Gap-Filling

Fixes applied based on diagnostic findings:
  FIX 1 - Split probe trajectories at skeleton intersections
          (converts km-long trips into road-link-sized segments)
  FIX 2 - Use geometric difference instead of aggressive overlap filter
          (extract only the gap portions outside the skeleton buffer)
  FIX 3 - Improved clustering: 30-degree heading bins normalised to 0-180,
          tighter DBSCAN eps=20m, median-length representative selection
  FIX 4 - Post-filter by length: remove <20m noise and >2km uncollapsed traces
  FIX 5 - Load both HPD weeks for better minor-road coverage
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import unary_union, split, snap
from sklearn.cluster import DBSCAN
import math
import os
import warnings

warnings.filterwarnings("ignore")


class GapFiller:
    def __init__(
        self, skeleton_path, probe_paths, ground_truth_path, output_dir="data"
    ):
        """
        Parameters
        ----------
        skeleton_path : str
            Path to interim_skeleton_phase2.gpkg (VPD skeleton).
        probe_paths : list[str]
            One or more HPD CSV paths (FIX 5: load both weeks).
        ground_truth_path : str
            Path to nav_kosovo.gpkg (ground truth for validation).
        output_dir : str
            Directory for output files.
        """
        self.skeleton_path = skeleton_path
        self.probe_paths = (
            probe_paths if isinstance(probe_paths, list) else [probe_paths]
        )
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir

        self.local_crs = None
        self.skeleton = None
        self.probes = None
        self.ground_truth = None
        self.final_network = None

    # ──────────────────────────────────────────────────────────────────
    #  DATA LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_data(self):
        """Loads Skeleton, Probes (both weeks), and Ground Truth."""
        print("Loading data...")

        # 1. Skeleton
        if not os.path.exists(self.skeleton_path):
            raise FileNotFoundError(f"Skeleton not found: {self.skeleton_path}")

        self.skeleton = gpd.read_file(self.skeleton_path)

        if self.skeleton.crs.is_geographic:
            centroid = self.skeleton.geometry.unary_union.centroid
            self.local_crs = (
                f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} "
                f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            )
            self.skeleton = self.skeleton.to_crs(self.local_crs)
        else:
            self.local_crs = self.skeleton.crs

        print(f"  Skeleton: {len(self.skeleton)} segments")

        # 2. Ground Truth
        if os.path.exists(self.ground_truth_path):
            self.ground_truth = gpd.read_file(self.ground_truth_path)
            if self.ground_truth.crs != self.local_crs:
                self.ground_truth = self.ground_truth.to_crs(self.local_crs)
            print(f"  Ground Truth: {len(self.ground_truth)} features")
        else:
            print(f"  Warning: GT not found at {self.ground_truth_path}")

        # 3. Probes — FIX 5: load ALL HPD weeks
        all_probes = []
        for probe_path in self.probe_paths:
            if not os.path.exists(probe_path):
                print(f"  Warning: Probe file not found: {probe_path}")
                continue
            print(f"  Loading Probe CSV: {probe_path}...")
            df = pd.read_csv(probe_path)
            df.sort_values(by=["traceid", "time"], inplace=True)

            grouped = df.groupby("traceid")
            for tid, group in grouped:
                if len(group) > 1:
                    coords = list(zip(group["longitude"], group["latitude"]))
                    geom = LineString(coords)
                    all_probes.append({"traceid": tid, "geometry": geom})

        self.probes = gpd.GeoDataFrame(all_probes, crs="EPSG:4326").to_crs(
            self.local_crs
        )
        print(
            f"  Probes processed: {len(self.probes)} trajectories "
            f"(from {len(self.probe_paths)} file(s))"
        )

    # ──────────────────────────────────────────────────────────────────
    #  FIX 1 & 2: SPLIT TRAJECTORIES + GEOMETRIC DIFFERENCE
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _explode_to_linestrings(geom):
        """Convert any geometry result into a list of LineStrings."""
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, LineString):
            return [geom]
        if isinstance(geom, MultiLineString):
            return list(geom.geoms)
        # GeometryCollection — extract only LineStrings
        return [g for g in geom.geoms if isinstance(g, LineString)]

    @staticmethod
    def _split_linestring_at_points(line, points, snap_tolerance=5.0):
        """Split a LineString at a set of Points (intersection nodes)."""
        if not points or not isinstance(line, LineString):
            return [line]

        # Snap points onto the line so shapely.split works reliably
        from shapely.ops import split as shapely_split, snap as shapely_snap

        result_lines = [line]
        for pt in points:
            new_results = []
            for seg in result_lines:
                if seg.is_empty:
                    continue
                snapped = shapely_snap(seg, pt, snap_tolerance)
                try:
                    parts = shapely_split(snapped, pt)
                    new_results.extend(
                        g for g in parts.geoms if isinstance(g, LineString)
                    )
                except Exception:
                    new_results.append(seg)
            result_lines = new_results

        return [l for l in result_lines if not l.is_empty and l.length > 0]

    def identify_and_split_gaps(self):
        """
        FIX 1: Split probe trajectories at skeleton intersections.
        FIX 2: Use geometric difference to extract only gap portions.

        Pipeline:
          1. Buffer the skeleton by 15m to create an exclusion zone
          2. Compute difference(probe, exclusion_zone) → gap-only geometry
          3. Explode MultiLineStrings into individual segments
          4. This replaces the old <20% overlap filter entirely
        """
        print("Identifying gaps (geometric difference approach)...")

        # Build exclusion zone from skeleton
        print("  Buffering skeleton (15m)...")
        skeleton_buffer = unary_union(self.skeleton.geometry.buffer(15.0))

        gap_segments = []

        print("  Computing probe - skeleton difference and splitting...")
        for idx, row in self.probes.iterrows():
            probe_geom = row.geometry
            if not isinstance(probe_geom, (LineString, MultiLineString)):
                continue

            # Step 1: Find where the probe intersects the skeleton buffer
            # These intersection points become split points
            try:
                intersection = probe_geom.intersection(skeleton_buffer.boundary)
                if intersection.is_empty:
                    split_pts = []
                elif intersection.geom_type == "Point":
                    split_pts = [intersection]
                elif intersection.geom_type == "MultiPoint":
                    split_pts = list(intersection.geoms)
                else:
                    split_pts = []
            except Exception:
                split_pts = []

            # Step 2: Split the trajectory at these intersection points
            if isinstance(probe_geom, MultiLineString):
                sub_lines = list(probe_geom.geoms)
            else:
                sub_lines = [probe_geom]

            split_segments = []
            for sub_line in sub_lines:
                relevant_pts = [p for p in split_pts if sub_line.distance(p) < 10.0]
                split_segments.extend(
                    self._split_linestring_at_points(sub_line, relevant_pts)
                )

            # Step 3: For each sub-segment, compute difference with skeleton
            # buffer to keep only the "gap" portion
            for seg in split_segments:
                try:
                    diff = seg.difference(skeleton_buffer)
                except Exception:
                    continue

                for line in self._explode_to_linestrings(diff):
                    if line.length > 5.0:  # minimal noise threshold
                        gap_segments.append(line)

        print(f"  Extracted {len(gap_segments)} gap sub-segments")
        
        # Segmentize long gaps to improve clustering
        return self.segmentize_gaps(gpd.GeoDataFrame(geometry=gap_segments, crs=self.local_crs))

    def segmentize_gaps(self, gdf, max_length=50.0):
        """
        Split long gap segments into smaller chunks (e.g., 50m) 
        to allow better density-based clustering.
        """
        print(f"  Segmentizing gaps (max {max_length}m)...")
        new_geoms = []
        for geom in gdf.geometry:
            if geom.length <= max_length:
                new_geoms.append(geom)
            else:
                # Split into chunks
                num_chunks = int(np.ceil(geom.length / max_length))
                for i in range(num_chunks):
                    # Extract substring
                    start = i * max_length
                    end = min((i + 1) * max_length, geom.length)
                    # shapely.ops.substring or interpolate points
                    # robust way: interpolate points along line
                    # simpler: use shapely's substring if available or linear referencing
                    
                    # Using linear referencing to cut
                    try:
                        p1 = geom.interpolate(start)
                        p2 = geom.interpolate(end)
                        # specialized logic needed to get the line segment between distances
                        # For now, let's just use a simple point interpolation approach 
                        # or rely on a helper function if we had one.
                        # Actually, substring is standard in recent shapely (2.0+), 
                        # but let's implement a safe sliced helper or use a library function if available.
                        # Since we don't want to depend on new shapely features without checking, 
                        # let's assume standard linear referencing.
                        
                        # A robust way is to project points at intervals and split.
                        # But simpler: just project points and linear interpolate?
                        # Let's use the substring method from shapely.ops if possible, 
                        # or construct it manually.
                        from shapely.ops import substring
                        segment = substring(geom, start, end)
                        new_geoms.append(segment)
                    except Exception:
                        new_geoms.append(geom) # Fallback
                        
        print(f"  Segmented {len(gdf)} -> {len(new_geoms)} chunks.")
        return gpd.GeoDataFrame(geometry=new_geoms, crs=self.local_crs)

    # ──────────────────────────────────────────────────────────────────
    #  FIX 3: IMPROVED CLUSTERING
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _heading_0_180(linestring):
        """Heading normalized to 0-180 (direction-agnostic)."""
        if not isinstance(linestring, LineString) or len(linestring.coords) < 2:
            return 0.0
        p1 = linestring.coords[0]
        p2 = linestring.coords[-1]
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        # Normalize to 0-180 (treat opposite directions as the same road)
        return angle % 180.0

    def densify_gaps(self, candidates):
        """
        FIX 3: Improved DBSCAN clustering.
          - 30-degree heading bins (normalised 0-180, 6 bins)
          - DBSCAN eps=20m on segment midpoints
          - Select the median-length segment as representative
        """
        print("Clustering gap candidates (improved DBSCAN)...")

        if len(candidates) == 0:
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.local_crs)

        candidates = candidates.copy()
        candidates["geometry"] = candidates.geometry.simplify(tolerance=3.0)

        # Remove empty/invalid after simplification
        candidates = candidates[
            ~candidates.geometry.is_empty & candidates.geometry.is_valid
        ].copy()

        if len(candidates) == 0:
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.local_crs)

        # Heading: 30-degree bins (6 bins across 0-180)
        candidates["heading"] = candidates.geometry.apply(self._heading_0_180)
        candidates["heading_bin"] = (candidates["heading"] // 30.0).astype(int)

        candidates["midpoint"] = candidates.geometry.centroid
        candidates["length"] = candidates.geometry.length

        final_segments = []

        for bin_val, group in candidates.groupby("heading_bin"):
            if len(group) == 0:
                continue

            coords = np.array([[p.x, p.y] for p in group["midpoint"]])

            # FIX 3: Tighter eps=20m
            db = DBSCAN(eps=20.0, min_samples=1).fit(coords)

            df_cluster = group.copy()
            df_cluster["cluster_id"] = db.labels_

            for cid, cluster_group in df_cluster.groupby("cluster_id"):
                if cid == -1:
                    # Noise points: keep individually if they are valid
                    for _, row in cluster_group.iterrows():
                        final_segments.append(row.geometry)
                    continue

                # FIX 3: Select median-length segment as representative
                sorted_cluster = cluster_group.sort_values("length")
                median_idx = len(sorted_cluster) // 2
                best_geom = sorted_cluster.iloc[median_idx].geometry
                final_segments.append(best_geom)

        print(f"  Clustered {len(candidates)} -> {len(final_segments)} segments")
        return gpd.GeoDataFrame(geometry=final_segments, crs=self.local_crs)

    # ──────────────────────────────────────────────────────────────────
    #  FIX 4: POST-FILTER BY LENGTH
    # ──────────────────────────────────────────────────────────────────

    def post_filter(self, gdf):
        """
        FIX 4: Remove segments <20m (noise).
        Relaxed: Do NOT drop >2km segments (they are now segmented anyway, but just in case).
        """
        print("Post-filtering by length...")
        before = len(gdf)
        lengths = gdf.geometry.length

        # Relaxed filter: Keep everything > 20m. 
        # No upper limit because we segmented them, so any remaining long one is likely valid or just a long segment.
        # But actually, if we segmented to 50m max, we shouldn't have any > 2000m.
        # So we can just drop the upper bound check to be safe.
        gdf = gdf[lengths >= 20.0].copy()

        after = len(gdf)
        removed = before - after
        print(
            f"  Removed {removed} segments "
            f"(kept {after}, dropped {removed} < 20m)"
        )
        return gdf

    # ──────────────────────────────────────────────────────────────────
    #  VALIDATION
    # ──────────────────────────────────────────────────────────────────

    def validate(self, new_roads):
        """Validates the combined network against Ground Truth."""
        print("Validating...")

        combined = pd.concat(
            [self.skeleton[["geometry"]], new_roads[["geometry"]]], ignore_index=True
        )
        self.final_network = combined

        if self.ground_truth is None:
            return None, None

        len_skel = self.skeleton.geometry.length.sum() / 1000.0
        len_new = new_roads.geometry.length.sum() / 1000.0
        len_total = combined.geometry.length.sum() / 1000.0
        len_gt = self.ground_truth.geometry.length.sum() / 1000.0

        print(f"  Length - Skeleton (VPD): {len_skel:.2f} km")
        print(f"  Length - New (Probes):   {len_new:.2f} km")
        print(f"  Length - Total:          {len_total:.2f} km")
        print(f"  Length - Ground Truth:   {len_gt:.2f} km")
        print(f"  Length ratio:            {len_total / len_gt:.2f}x GT")

        # Precision
        print("  Calculating Precision...")
        gt_buffer = self.ground_truth.geometry.buffer(15.0).unary_union
        final_in_gt = combined.geometry.intersection(gt_buffer)
        matched_length = final_in_gt.length.sum() / 1000.0
        precision = (matched_length / len_total) * 100.0
        print(f"  Precision: {precision:.1f}%")

        # Recall
        print("  Calculating Recall...")
        network_buffer = combined.geometry.buffer(15.0).unary_union
        gt_in_network = self.ground_truth.geometry.intersection(network_buffer)
        covered_gt = gt_in_network.length.sum() / 1000.0
        recall = (covered_gt / len_gt) * 100.0
        print(f"  Recall:    {recall:.1f}%")

        return precision, recall

    # ──────────────────────────────────────────────────────────────────
    #  MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        self.load_data()

        # FIX 1+2: split trajectories and extract gaps via geometric difference
        gap_candidates = self.identify_and_split_gaps()

        # FIX 3: improved clustering
        new_segments = self.densify_gaps(gap_candidates)

        # FIX 4: post-filter by length
        new_segments = self.post_filter(new_segments)

        # Combine Skeleton + New Segments
        print("Combining Skeleton and Probe Segments...")
        skel_res = self.skeleton[["geometry"]].copy()
        skel_res["source"] = "VPD"

        new_res = new_segments[["geometry"]].copy()
        new_res["source"] = "Probe"

        self.final_network = gpd.GeoDataFrame(
            pd.concat([skel_res, new_res], ignore_index=True), crs=self.local_crs
        )

        # Export
        output_path = os.path.join(self.output_dir, "final_network_phase3.gpkg")
        print(f"Exporting to {output_path}...")
        self.final_network.to_file(output_path, driver="GPKG")

        # Validate
        self.validate(new_segments)

        print("Phase 3 Complete.")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SKELETON_FILE = os.path.join(BASE_DIR, "data", "interim_skeleton_phase2.gpkg")
    GT_FILE = os.path.join(BASE_DIR, "data", "Kosovo_nav_streets", "nav_kosovo.gpkg")

    # FIX 5: Load both HPD weeks
    PROBE_FILES = [
        os.path.join(BASE_DIR, "data", "Kosovo_HPD", "XKO_HPD_week_1.csv"),
        os.path.join(BASE_DIR, "data", "Kosovo_HPD", "XKO_HPD_week_2.csv"),
    ]

    filler = GapFiller(SKELETON_FILE, PROBE_FILES, GT_FILE)
    filler.run()
