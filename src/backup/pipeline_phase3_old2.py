"""
Phase 3: Probe-based Gap Filling

Key improvements over previous version:
  FIX 1 - Uses Phase 1 HPD output (pre-filtered by speed) instead of raw CSVs
  FIX 2 - Reduced skeleton exclusion buffer (10m, was 15m) to avoid hiding
          parallel roads
  FIX 3 - DBSCAN min_samples=2 (was 1!) — actual noise rejection now works
  FIX 4 - Noise points (cid == -1) are DISCARDED, not kept as roads
  FIX 5 - Heading-aware clustering with 20° bins (6 bins over 0-180°)
  FIX 6 - Medoid-based representative selection (closest to cluster centroid)
  FIX 7 - Length post-filter: 15m minimum (was 20m — too aggressive)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, split, snap, substring
from sklearn.cluster import DBSCAN
import math
import os
import time
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GapFiller:
    """
    Phase 3: Fill gaps in the VPD skeleton using HPD probe data.

    Algorithm:
      1. Buffer skeleton to create exclusion zone (10m)
      2. Compute geometric difference: probe - skeleton_buffer
      3. Segmentize long gap portions into manageable chunks
      4. Heading-aware DBSCAN clustering with min_samples=2
      5. Select medoid representative per cluster
      6. Post-filter by length
    """

    def __init__(
        self,
        skeleton_path,
        hpd_path,
        output_dir="data",
        skeleton_buffer_m=10.0,     # FIX 2: tighter (was 15m)
        dbscan_eps=15.0,            # tighter eps
        dbscan_min_samples=2,       # FIX 3: actual noise rejection
        heading_bin_width=30.0,     # FIX 5: 30° bins
        min_gap_length=5.0,         # minimum gap segment length
        min_output_length=15.0,     # FIX 7: post-filter (was 20m)
        max_segment_length=80.0,    # segmentization max
    ):
        self.skeleton_path = skeleton_path
        self.hpd_path = hpd_path
        self.output_dir = output_dir

        self.skeleton_buffer_m = skeleton_buffer_m
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.heading_bin_width = heading_bin_width
        self.min_gap_length = min_gap_length
        self.min_output_length = min_output_length
        self.max_segment_length = max_segment_length

        self.local_crs = None
        self.skeleton = None
        self.probes = None
        self.final_network = None

    # ──────────────────────────────────────────────────────────────────
    #  DATA LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_data(self):
        """Load skeleton and HPD probes (from Phase 1 output)."""
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

        # FIX 1: Load HPD from Phase 1 output (already speed-filtered)
        if not os.path.exists(self.hpd_path):
            # Fallback: try loading raw CSVs
            print(f"    Warning: HPD GPKG not found at {self.hpd_path}")
            print("    Falling back to raw CSV loading...")
            self._load_hpd_from_csvs()
        else:
            self.probes = gpd.read_file(self.hpd_path)
            if self.probes.crs != self.local_crs:
                self.probes = self.probes.to_crs(self.local_crs)
            print(f"    Probes: {len(self.probes)} traces (pre-filtered)")

    def _load_hpd_from_csvs(self):
        """Fallback: load HPD from raw CSVs with speed filtering."""
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

            # Speed filter: remove traces with avg < 5 km/h
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
                    all_probes.append({"traceid": tid, "geometry": geom})

        self.probes = gpd.GeoDataFrame(all_probes, crs="EPSG:4326").to_crs(
            self.local_crs
        )
        print(f"    Probes (CSV fallback): {len(self.probes)} traces")

    # ──────────────────────────────────────────────────────────────────
    #  GAP EXTRACTION
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

    def extract_gaps(self):
        """
        FIX 2: Use 10m skeleton buffer (was 15m) for exclusion zone.
        Uses spatial-index-based approach instead of monolithic unary_union
        to avoid memory/performance issues with large skeleton sets.

        For each probe trace:
          1. Find nearby skeleton segments via spatial index
          2. Build local buffer from only those nearby segments
          3. Compute geometric difference
          4. Explode results to individual LineStrings
          5. Keep segments > min_gap_length
        """
        print("  Extracting gap segments...")

        # Build spatial index on skeleton buffers (chunked for speed)
        print(f"    Building {self.skeleton_buffer_m}m skeleton buffers (chunked)...")
        skel_buffered = self.skeleton.geometry.buffer(self.skeleton_buffer_m)
        skel_buf_gdf = gpd.GeoDataFrame(geometry=skel_buffered.values, crs=self.local_crs)
        skel_sindex = skel_buf_gdf.sindex

        gap_segments = []
        n_probes = len(self.probes)

        probe_geoms = self.probes.geometry.values

        for i in range(n_probes):
            probe_geom = probe_geoms[i]
            if not isinstance(probe_geom, (LineString, MultiLineString)):
                continue

            # Find skeleton buffers that intersect this probe's bounding box
            candidate_idxs = list(skel_sindex.intersection(probe_geom.bounds))
            if not candidate_idxs:
                # No nearby skeleton — keep entire probe as gap
                for line in self._explode_to_linestrings(probe_geom):
                    if line.length >= self.min_gap_length:
                        gap_segments.append(line)
                continue

            # Build local buffer union from only nearby skeleton segments
            try:
                local_buffer = unary_union(skel_buffered.iloc[candidate_idxs].values)
                diff = probe_geom.difference(local_buffer)
            except Exception:
                continue

            for line in self._explode_to_linestrings(diff):
                if line.length >= self.min_gap_length:
                    gap_segments.append(line)

            if (i + 1) % 1000 == 0:
                print(f"    Processed {i + 1}/{n_probes} probes, "
                      f"gaps found: {len(gap_segments)}")

        print(f"    Extracted {len(gap_segments)} gap segments from "
              f"{n_probes} probes")

        if not gap_segments:
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.local_crs)

        return gpd.GeoDataFrame(geometry=gap_segments, crs=self.local_crs)

    def segmentize_gaps(self, gdf):
        """Split long gap segments into smaller chunks for better clustering."""
        print(f"    Segmentizing gaps (max {self.max_segment_length}m)...")
        new_geoms = []

        for geom in gdf.geometry:
            if geom.length <= self.max_segment_length:
                new_geoms.append(geom)
            else:
                n_chunks = int(np.ceil(geom.length / self.max_segment_length))
                for i in range(n_chunks):
                    start_d = i * self.max_segment_length
                    end_d = min((i + 1) * self.max_segment_length, geom.length)
                    try:
                        seg = substring(geom, start_d, end_d)
                        if not seg.is_empty and seg.length > 0:
                            new_geoms.append(seg)
                    except Exception:
                        new_geoms.append(geom)
                        break

        print(f"    Segmented: {len(gdf)} → {len(new_geoms)} chunks")
        return gpd.GeoDataFrame(geometry=new_geoms, crs=self.local_crs)

    # ──────────────────────────────────────────────────────────────────
    #  CLUSTERING (FIX 3, 4, 5, 6)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _heading_0_180(linestring):
        """Heading normalized to 0-180° (direction-agnostic)."""
        if not isinstance(linestring, LineString) or len(linestring.coords) < 2:
            return 0.0
        c = linestring.coords
        p1, p2 = c[0], c[-1]
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        return angle % 180.0

    def cluster_gaps(self, candidates):
        """
        Heading-aware DBSCAN clustering with proper noise rejection.

        FIX 3: min_samples=2 (was 1 — every point was a cluster)
        FIX 4: Noise points discarded (were kept as roads!)
        FIX 5: 30° heading bins
        FIX 6: Medoid-based representative selection
        """
        print("  Clustering gap candidates...")

        if len(candidates) == 0:
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.local_crs)

        candidates = candidates.copy()

        # Simplify for clustering speed
        candidates["geometry"] = candidates.geometry.simplify(2.0)
        candidates = candidates[
            ~candidates.geometry.is_empty & candidates.geometry.is_valid
        ].copy()

        if len(candidates) == 0:
            return gpd.GeoDataFrame(columns=["geometry"], crs=self.local_crs)

        # Heading bins
        candidates["heading"] = candidates.geometry.apply(self._heading_0_180)
        n_bins = int(180.0 / self.heading_bin_width)
        candidates["heading_bin"] = (
            candidates["heading"] // self.heading_bin_width
        ).astype(int).clip(0, n_bins - 1)

        candidates["midpoint"] = candidates.geometry.centroid
        candidates["length"] = candidates.geometry.length

        final_segments = []
        noise_count = 0

        for bin_val, group in candidates.groupby("heading_bin"):
            if len(group) == 0:
                continue

            coords = np.array([[p.x, p.y] for p in group["midpoint"]])

            # FIX 3: min_samples=2 for real noise rejection
            db = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
            ).fit(coords)

            group = group.copy()
            group["cluster_id"] = db.labels_

            for cid, cgroup in group.groupby("cluster_id"):
                # FIX 4: DISCARD noise points
                if cid == -1:
                    noise_count += len(cgroup)
                    continue

                # FIX 6: Medoid selection — pick segment closest to cluster centroid
                centroid_x = cgroup["midpoint"].apply(lambda p: p.x).mean()
                centroid_y = cgroup["midpoint"].apply(lambda p: p.y).mean()
                cluster_centroid = Point(centroid_x, centroid_y)

                dists = cgroup["midpoint"].apply(lambda p: p.distance(cluster_centroid))

                # Pick the one closest to centroid (medoid)
                best_idx = dists.idxmin()
                final_segments.append(cgroup.loc[best_idx, "geometry"])

        print(f"    Clustered: {len(candidates)} → {len(final_segments)} segments "
              f"({noise_count} noise discarded)")
        return gpd.GeoDataFrame(geometry=final_segments, crs=self.local_crs)

    # ──────────────────────────────────────────────────────────────────
    #  POST-FILTER (FIX 7)
    # ──────────────────────────────────────────────────────────────────

    def post_filter(self, gdf):
        """Remove segments shorter than min_output_length."""
        if len(gdf) == 0:
            return gdf
        before = len(gdf)
        gdf = gdf[gdf.geometry.length >= self.min_output_length].copy()
        removed = before - len(gdf)
        print(f"    Post-filter: removed {removed} segments "
              f"< {self.min_output_length}m (kept {len(gdf)})")
        return gdf

    # ──────────────────────────────────────────────────────────────────
    #  EXPORT / MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  PHASE 3: Probe-based Gap Filling")
        print("=" * 60)
        t0 = time.time()

        self.load_data()

        # Extract gap portions from probes
        gap_candidates = self.extract_gaps()

        # Segmentize long gaps
        if len(gap_candidates) > 0:
            gap_candidates = self.segmentize_gaps(gap_candidates)

        # Cluster and select representatives
        new_segments = self.cluster_gaps(gap_candidates)

        # Post-filter by length
        new_segments = self.post_filter(new_segments)

        # Combine skeleton + new segments
        print("  Combining skeleton + probe segments...")
        skel_res = self.skeleton[["geometry"]].copy()
        skel_res["source"] = "VPD"

        new_res = new_segments[["geometry"]].copy()
        new_res["source"] = "Probe"

        self.final_network = gpd.GeoDataFrame(
            pd.concat([skel_res, new_res], ignore_index=True), crs=self.local_crs
        )

        # Export in WGS84
        output_path = os.path.join(self.output_dir, "final_network_phase3.gpkg")
        print(f"  Exporting to {output_path}...")
        out_gdf = self.final_network.to_crs("EPSG:4326")
        out_gdf.to_file(output_path, driver="GPKG")

        elapsed = time.time() - t0
        skel_km = self.skeleton.geometry.length.sum() / 1000.0
        new_km = new_segments.geometry.length.sum() / 1000.0 if len(new_segments) > 0 else 0

        print(f"\nPhase 3 complete in {elapsed:.1f}s")
        print(f"  Skeleton:      {len(self.skeleton)} segments ({skel_km:.1f} km)")
        print(f"  Gap segments:  {len(new_segments)} ({new_km:.1f} km)")
        print(f"  Total output:  {len(self.final_network)} segments")

        return self.final_network


if __name__ == "__main__":
    SKELETON_FILE = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
    HPD_FILE = os.path.join(PROJECT_ROOT, "data", "interim_hpd_phase1.gpkg")

    filler = GapFiller(SKELETON_FILE, HPD_FILE)
    filler.run()
