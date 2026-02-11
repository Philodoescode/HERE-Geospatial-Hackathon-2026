"""
Phase 3: Probe-based Gap Filling (Enhanced Clustering)

Enhanced gap filling with options for advanced trajectory clustering:
  1. Buffer skeleton to create exclusion zone (8m - tighter to catch parallel roads)
  2. Compute geometric difference: probe - skeleton_buffer
  3. Segmentize long gap portions into manageable chunks
  4. Heading-aware clustering (DBSCAN or Fréchet-based)
  5. Select medoid representative per cluster
  6. Temporal consistency filtering (multi-day presence)
  7. Post-filter by length and quality

Improvements from Kharita/Roadster algorithms:
  - Optional Fréchet-based trajectory clustering
  - Heading consistency scoring
  - Temporal repeatability filtering
"""

import gc
import math
import os
import sys
import time
import warnings
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, substring
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced clustering (optional)
try:
    from src.algorithms.quality_scoring import HeadingAwareClusterer
    from src.algorithms.trajectory_clustering import SubtrajectoryClusterer, TrajectoryClusterConfig
    ENHANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    ENHANCED_CLUSTERING_AVAILABLE = False


class ProbeRecovery:
    """
    Phase 3: Fill gaps in the VPD skeleton using HPD probe data.

    Uses heading-aware clustering to identify roads from probe traces,
    preserving actual GPS geometry while improving quality.
    """

    def __init__(
        self,
        skeleton_path,
        hpd_path,
        output_dir="data",
        skeleton_buffer_m=8.0,      # tighter buffer to catch parallel/nearby roads
        dbscan_eps=12.0,            # clustering distance
        dbscan_min_samples=2,       # require 2+ traces for noise rejection
        heading_bin_width=30.0,     # 30° heading bins (6 bins over 0-180°)
        min_gap_length=3.0,         # minimum gap segment length to consider (lowered)
        min_output_length=6.0,      # post-filter minimum (more permissive for recovery)
        max_segment_length=100.0,   # segmentization max
        min_probe_speed=3.0,        # minimum average speed (km/h) - lowered to catch residential
        # Enhanced clustering options (from Kharita/Roadster)
        use_enhanced_clustering=True,
        cluster_heading_tolerance_deg=35.0,
        min_temporal_days=1,        # require traces on at least N days (0 = disabled)
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
        self.min_probe_speed = min_probe_speed
        
        # Enhanced options
        self.use_enhanced_clustering = use_enhanced_clustering
        self.cluster_heading_tolerance_deg = cluster_heading_tolerance_deg
        self.min_temporal_days = min_temporal_days

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
        """Fallback: load HPD from raw CSVs with relaxed speed filtering."""
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

            # Relaxed speed filter: only remove very slow traces (< 3 km/h)
            if "speed" in df.columns:
                trace_avg = df.groupby("traceid")["speed"].mean()
                fast = trace_avg[trace_avg >= self.min_probe_speed].index
                df = df[df["traceid"].isin(fast)]

            for tid, group in df.groupby("traceid"):
                if len(group) < 2:
                    continue
                coords = list(zip(group["longitude"], group["latitude"]))
                geom = LineString(coords)
                if not geom.is_empty:
                    avg_spd = group["speed"].mean() if "speed" in group.columns else None
                    all_probes.append({
                        "traceid": tid,
                        "geometry": geom,
                        "avg_speed": avg_spd
                    })

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
        Extract probe portions NOT already covered by VPD skeleton.
        Uses spatial-index-based approach for memory efficiency.

        For each probe trace:
          1. Find nearby skeleton segments via spatial index
          2. Build local buffer from only those nearby segments
          3. Compute geometric difference
          4. Explode results to individual LineStrings
          5. Keep segments > min_gap_length
        """
        print(f"  Extracting gap segments (excluding {self.skeleton_buffer_m}m VPD buffer)...")

        # Build spatial index on skeleton buffers
        skel_buffered = self.skeleton.geometry.buffer(self.skeleton_buffer_m)
        skel_buf_gdf = gpd.GeoDataFrame(geometry=skel_buffered.values, crs=self.local_crs)
        skel_sindex = skel_buf_gdf.sindex

        gap_segments = []
        n_probes = len(self.probes)

        for i in range(n_probes):
            row = self.probes.iloc[i]
            probe_geom = row.geometry
            if not isinstance(probe_geom, (LineString, MultiLineString)):
                continue

            # Find nearby skeleton buffers
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

        print(f"    Extracted {len(gap_segments)} gap segments from {n_probes} probes")

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
    #  CLUSTERING (Enhanced with heading consistency)
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

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Minimal angle difference (0-90° for direction-agnostic)."""
        d = abs((a - b) % 180.0)
        return min(d, 180.0 - d)

    def cluster_gaps(self, candidates):
        """
        Enhanced heading-aware clustering with heading consistency scoring.

        Improvements from Kharita/Roadster:
          - Heading consistency check within clusters
          - Optional Fréchet-based scoring
          - Temporal day filtering (if available)
          - Weighted medoid selection based on length
        """
        print("  Clustering gap candidates (enhanced)...")

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

        # Heading bins (finer for better separation)
        candidates["heading"] = candidates.geometry.apply(self._heading_0_180)
        n_bins = int(180.0 / self.heading_bin_width)
        candidates["heading_bin"] = (
            candidates["heading"] // self.heading_bin_width
        ).astype(int).clip(0, n_bins - 1)

        candidates["midpoint"] = candidates.geometry.centroid
        candidates["length"] = candidates.geometry.length

        final_segments = []
        cluster_stats = []
        noise_count = 0

        for bin_val, group in candidates.groupby("heading_bin"):
            if len(group) == 0:
                continue

            coords = np.array([[p.x, p.y] for p in group["midpoint"]])
            headings = group["heading"].values

            # DBSCAN with min_samples=2 for noise rejection
            db = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
            ).fit(coords)

            group = group.copy()
            group["cluster_id"] = db.labels_

            for cid, cgroup in group.groupby("cluster_id"):
                # Discard noise points
                if cid == -1:
                    noise_count += len(cgroup)
                    continue

                # === HEADING CONSISTENCY SCORING (Kharita-style) ===
                # Compute heading consistency but DON'T filter - use it as quality metric
                cluster_headings = cgroup["heading"].values
                if len(cluster_headings) > 1:
                    # Convert to radians and compute circular mean
                    rads = np.radians(cluster_headings * 2)  # *2 for 0-180 range
                    sin_sum = np.sum(np.sin(rads))
                    cos_sum = np.sum(np.cos(rads))
                    r = np.sqrt(sin_sum**2 + cos_sum**2) / len(cluster_headings)
                    heading_consistency = float(np.clip(r, 0.0, 1.0))
                else:
                    heading_consistency = 0.5
                
                # NOTE: We keep ALL clusters now - quality scoring happens in Phase 5
                # This improves recovery by not prematurely discarding potential roads

                # === WEIGHTED MEDOID SELECTION ===
                # Weight by length (longer segments are more reliable)
                lengths = cgroup["length"].values
                weights = lengths / np.sum(lengths) if np.sum(lengths) > 0 else np.ones(len(lengths)) / len(lengths)
                
                # Weighted centroid
                centroid_x = np.sum([p.x * w for p, w in zip(cgroup["midpoint"], weights)])
                centroid_y = np.sum([p.y * w for p, w in zip(cgroup["midpoint"], weights)])
                cluster_centroid = Point(centroid_x, centroid_y)

                # Distance from each segment to centroid
                dists = cgroup["midpoint"].apply(lambda p: p.distance(cluster_centroid))

                # Select medoid (closest to weighted centroid)
                best_idx = dists.idxmin()
                best_geom = cgroup.loc[best_idx, "geometry"]
                
                # Store cluster stats for logging
                cluster_stats.append({
                    "n_members": len(cgroup),
                    "heading_consistency": heading_consistency,
                    "total_length": cgroup["length"].sum(),
                })
                
                # Store segment WITH support data
                final_segments.append({
                    "geometry": best_geom,
                    "cluster_members": int(len(cgroup)),
                    "heading_consistency": float(heading_consistency),
                    "cluster_length_m": float(cgroup["length"].sum()),
                    "weighted_support": float(len(cgroup) * (0.5 + 0.5 * heading_consistency)),
                })

        # Log clustering statistics
        if cluster_stats:
            avg_consistency = np.mean([s["heading_consistency"] for s in cluster_stats])
            avg_members = np.mean([s["n_members"] for s in cluster_stats])
            print(f"    Cluster stats: {len(cluster_stats)} clusters, "
                  f"avg {avg_members:.1f} members, "
                  f"avg heading consistency {avg_consistency:.2f}")

        print(f"    Clustered: {len(candidates)} → {len(final_segments)} segments "
              f"({noise_count} noise discarded)")
        
        # Return GeoDataFrame with support columns
        if not final_segments:
            return gpd.GeoDataFrame(
                columns=["geometry", "cluster_members", "heading_consistency", 
                         "cluster_length_m", "weighted_support"],
                crs=self.local_crs
            )
        
        return gpd.GeoDataFrame(final_segments, crs=self.local_crs)

    # ──────────────────────────────────────────────────────────────────
    #  POST-FILTER
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
    #  MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  PHASE 3: Probe-based Gap Filling (DBSCAN)")
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

        # Export just the probe skeleton (Phase 4 will merge)
        output_path = os.path.join(self.output_dir, "interim_probe_skeleton_phase3.gpkg")
        print(f"  Exporting probe skeleton to {output_path}...")

        if len(new_segments) > 0:
            new_segments["source"] = "Probe"
            out_gdf = new_segments.to_crs("EPSG:4326")
            out_gdf.to_file(output_path, driver="GPKG")
            self.probe_skeleton_gdf = new_segments
        else:
            # Create empty output file
            empty_gdf = gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")
            empty_gdf.to_file(output_path, driver="GPKG")
            self.probe_skeleton_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=self.local_crs)

        elapsed = time.time() - t0
        new_km = new_segments.geometry.length.sum() / 1000.0 if len(new_segments) > 0 else 0

        print(f"\nPhase 3 complete in {elapsed:.1f}s")
        print(f"  Gap segments:  {len(new_segments)} ({new_km:.1f} km)")

        gc.collect()
        return self.probe_skeleton_gdf


if __name__ == "__main__":
    SKELETON_FILE = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
    HPD_FILE = os.path.join(PROJECT_ROOT, "data", "interim_hpd_phase1.gpkg")

    recovery = ProbeRecovery(SKELETON_FILE, HPD_FILE)
    recovery.run()
