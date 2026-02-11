"""
Phase 4: Graph Merging — Combine VPD + Probe Skeletons

Merges the high-precision VPD skeleton (Phase 2) with the high-recall
Probe skeleton (Phase 3) into a single network without degrading precision.

Algorithm:
  1. Load both skeletons
  2. Buffer deduplication: remove Probe edges already covered by VPD
  3. Endpoint snapping via cKDTree + union-find
  4. Line merge for collinear adjacent segments
  5. Export combined network
"""

import gc
import os
import sys
import time
import warnings
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, nearest_points, snap, unary_union

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GraphMerger:
    """
    Phase 4: Merge VPD skeleton + Probe skeleton into a unified network.

    VPD skeleton is high-precision (kept as-is).
    Probe skeleton fills gaps but may overlap — deduplication is critical.
    """

    def __init__(
        self,
        vpd_skeleton_path,
        probe_skeleton_path,
        output_dir="data",
        dedup_buffer_m=10.0,      # tighter dedup zone to preserve parallel roads (was 18)
        snap_dist_m=10.0,         # endpoint snapping distance
        min_length_m=6.0,         # relaxed minimum (was 8)
        probe_min_length_m=8.0,   # relaxed minimum for probe lines (was 12)
    ):
        self.vpd_skeleton_path = vpd_skeleton_path
        self.probe_skeleton_path = probe_skeleton_path
        self.output_dir = output_dir
        self.dedup_buffer_m = dedup_buffer_m
        self.snap_dist_m = snap_dist_m
        self.min_length_m = min_length_m
        self.probe_min_length_m = probe_min_length_m

        self.local_crs = None
        self.vpd_skel = None
        self.probe_skel = None
        self.merged_gdf = None

    # ──────────────────────────────────────────────────────────────────
    #  DATA LOADING
    # ──────────────────────────────────────────────────────────────────

    def load_data(self):
        print("  Loading VPD and Probe skeletons...")

        # VPD skeleton
        self.vpd_skel = gpd.read_file(self.vpd_skeleton_path)
        if self.vpd_skel.crs is None or self.vpd_skel.crs.is_geographic:
            bounds = self.vpd_skel.total_bounds
            cx = (bounds[0] + bounds[2]) / 2.0
            cy = (bounds[1] + bounds[3]) / 2.0
            self.local_crs = (
                f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
                f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            )
            self.vpd_skel = self.vpd_skel.to_crs(self.local_crs)
        else:
            self.local_crs = self.vpd_skel.crs

        # Filter valid geometries
        self.vpd_skel = self.vpd_skel[
            self.vpd_skel.geometry.notna()
            & ~self.vpd_skel.geometry.is_empty
        ].reset_index(drop=True)

        print(f"    VPD skeleton: {len(self.vpd_skel)} segments")

        # Probe skeleton
        if os.path.exists(self.probe_skeleton_path):
            self.probe_skel = gpd.read_file(self.probe_skeleton_path)
            if len(self.probe_skel) > 0:
                self.probe_skel = self.probe_skel.to_crs(self.local_crs)
                self.probe_skel = self.probe_skel[
                    self.probe_skel.geometry.notna()
                    & ~self.probe_skel.geometry.is_empty
                ].reset_index(drop=True)
            print(f"    Probe skeleton: {len(self.probe_skel)} segments")
        else:
            print(f"    Probe skeleton not found at {self.probe_skeleton_path}")
            self.probe_skel = gpd.GeoDataFrame(columns=["geometry"], crs=self.local_crs)

    # ──────────────────────────────────────────────────────────────────
    #  BUFFER DEDUPLICATION
    # ──────────────────────────────────────────────────────────────────

    def deduplicate_probes(self):
        """
        Remove probe segments that fall entirely within the VPD buffer.
        A probe segment is kept only if a significant portion lies
        outside the VPD coverage zone.
        """
        if len(self.probe_skel) == 0:
            print("  No probe segments to deduplicate.")
            return

        print(f"  Deduplicating probes (removing within {self.dedup_buffer_m}m of VPD)...")
        before = len(self.probe_skel)

        # Build VPD buffer
        vpd_buffered = self.vpd_skel.geometry.buffer(self.dedup_buffer_m)
        vpd_buf_union = unary_union(vpd_buffered.values)

        # For each probe segment, compute what fraction lies within VPD buffer
        keep_mask = []
        for geom in self.probe_skel.geometry:
            try:
                inside = geom.intersection(vpd_buf_union)
                frac_inside = inside.length / geom.length if geom.length > 0 else 1.0
                # Keep if >40% is outside the VPD buffer (relaxed from 50%)
                # AND the segment is long enough to be a real road
                keep = (frac_inside < 0.6) and (geom.length >= self.probe_min_length_m)
                keep_mask.append(keep)
            except Exception:
                keep_mask.append(False)  # on error, discard probe (conservative)

        self.probe_skel = self.probe_skel[keep_mask].reset_index(drop=True)
        removed = before - len(self.probe_skel)
        print(f"    Dedup: {before} -> {len(self.probe_skel)} probe segments "
              f"({removed} removed as VPD-covered)")

        del vpd_buf_union, vpd_buffered
        gc.collect()

    # ──────────────────────────────────────────────────────────────────
    #  ENDPOINT SNAPPING
    # ──────────────────────────────────────────────────────────────────

    def snap_endpoints(self, lines):
        """
        Snap nearby endpoints together using cKDTree + union-find.
        This ensures connectivity at junctions.
        
        Returns list of same length as input - invalid geometries become None.
        """
        if len(lines) < 2:
            return lines

        print(f"  Snapping endpoints (within {self.snap_dist_m}m)...")

        # Collect all endpoints
        all_ends = []
        for line in lines:
            cs = list(line.coords)
            all_ends.append(cs[0])
            all_ends.append(cs[-1])

        coords_arr = np.array(all_ends, dtype=np.float64)
        kd = cKDTree(coords_arr)
        pairs = kd.query_pairs(self.snap_dist_m)

        if not pairs:
            print("    No endpoints to snap.")
            return lines

        # Union-find
        parent = list(range(len(all_ends)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        clusters = defaultdict(list)
        for i in range(len(all_ends)):
            clusters[find(i)].append(i)

        snap_map = {}
        for members in clusters.values():
            if len(members) < 2:
                continue
            cx = np.mean([coords_arr[m][0] for m in members])
            cy = np.mean([coords_arr[m][1] for m in members])
            for m in members:
                snap_map[m] = (cx, cy)

        # Return same length as input - None for invalid geometries
        snapped = []
        for li, line in enumerate(lines):
            coords = list(line.coords)
            si, ei = li * 2, li * 2 + 1
            if si in snap_map:
                coords[0] = snap_map[si]
            if ei in snap_map:
                coords[-1] = snap_map[ei]
            new_line = LineString(coords)
            if not new_line.is_empty and new_line.length > 0:
                snapped.append(new_line)
            else:
                snapped.append(None)  # Preserve index alignment

        n_snapped = len(snap_map)
        print(f"    Snapped {n_snapped} endpoints across {len(clusters)} groups")
        return snapped

    # ──────────────────────────────────────────────────────────────────
    #  MERGE + CLEAN
    # ──────────────────────────────────────────────────────────────────

    def merge_networks(self):
        """
        Combine VPD + deduplicated Probe segments while preserving attributes.
        
        Unlike the old version, we do NOT use linemerge (which destroys attributes).
        Instead, we concatenate GeoDataFrames and preserve support data.
        """
        print("  Merging VPD + Probe networks (attribute-preserving)...")

        # Prepare VPD with default high support
        vpd_records = []
        for idx, row in self.vpd_skel.iterrows():
            geom = row.geometry
            geoms_to_add = []
            if isinstance(geom, MultiLineString):
                geoms_to_add = list(geom.geoms)
            elif isinstance(geom, LineString):
                geoms_to_add = [geom]
            
            for g in geoms_to_add:
                vpd_records.append({
                    "geometry": g,
                    "source": "VPD",
                    "weighted_support": 50.0,  # VPD segments get high default support
                    "heading_consistency": 0.9,
                    "cluster_members": 10,
                })
        
        # Prepare Probe with support from Phase 3 output
        probe_records = []
        for idx, row in self.probe_skel.iterrows():
            geom = row.geometry
            geoms_to_add = []
            if isinstance(geom, MultiLineString):
                geoms_to_add = list(geom.geoms)
            elif isinstance(geom, LineString):
                geoms_to_add = [geom]
            
            # Get support data if available (from Phase 3)
            weighted_support = row.get("weighted_support", 3.0) if hasattr(row, "get") else getattr(row, "weighted_support", 3.0)
            heading_consistency = row.get("heading_consistency", 0.5) if hasattr(row, "get") else getattr(row, "heading_consistency", 0.5)
            cluster_members = row.get("cluster_members", 2) if hasattr(row, "get") else getattr(row, "cluster_members", 2)
            
            for g in geoms_to_add:
                probe_records.append({
                    "geometry": g,
                    "source": "Probe",
                    "weighted_support": float(weighted_support) if weighted_support is not None else 3.0,
                    "heading_consistency": float(heading_consistency) if heading_consistency is not None else 0.5,
                    "cluster_members": int(cluster_members) if cluster_members is not None else 2,
                })
        
        print(f"    Combined: {len(vpd_records)} VPD + {len(probe_records)} Probe = {len(vpd_records) + len(probe_records)} total")

        # Create combined GeoDataFrame
        all_records = vpd_records + probe_records
        if not all_records:
            self.merged_gdf = gpd.GeoDataFrame(
                columns=["geometry", "source", "weighted_support", "heading_consistency", "cluster_members"],
                crs=self.local_crs
            )
            return self.merged_gdf
        
        combined_gdf = gpd.GeoDataFrame(all_records, crs=self.local_crs)
        
        # Filter short segments (different thresholds for VPD vs Probe)
        keep_mask = []
        for idx, row in combined_gdf.iterrows():
            if row["source"] == "VPD":
                keep_mask.append(row.geometry.length >= self.min_length_m)
            else:
                keep_mask.append(row.geometry.length >= self.probe_min_length_m)
        
        combined_gdf = combined_gdf[keep_mask].reset_index(drop=True)
        print(f"    After length filter: {len(combined_gdf)} segments")

        # Snap endpoints - need to preserve attribute alignment
        # First snap all geometries
        original_geoms = list(combined_gdf.geometry)
        snapped_geoms = self.snap_endpoints(original_geoms)
        
        # Build new records keeping only valid snapped geometries
        valid_records = []
        for i, geom in enumerate(snapped_geoms):
            if geom is not None and not geom.is_empty and geom.length > 0:
                row = combined_gdf.iloc[i].to_dict()
                row["geometry"] = geom
                valid_records.append(row)
        
        if valid_records:
            combined_gdf = gpd.GeoDataFrame(valid_records, crs=self.local_crs)
        else:
            combined_gdf = gpd.GeoDataFrame(
                columns=["geometry", "source", "weighted_support", "heading_consistency", "cluster_members"],
                crs=self.local_crs
            )
        
        print(f"    After snapping: {len(combined_gdf)} segments")
        
        # Log support stats
        probe_mask = combined_gdf["source"] == "Probe"
        if probe_mask.any():
            probe_support = combined_gdf.loc[probe_mask, "weighted_support"]
            print(f"    Probe support stats: min={probe_support.min():.1f}, "
                  f"max={probe_support.max():.1f}, mean={probe_support.mean():.1f}")

        self.merged_gdf = combined_gdf
        return self.merged_gdf

    # ──────────────────────────────────────────────────────────────────
    #  EXPORT / MAIN PIPELINE
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("  PHASE 4: Graph Merging (VPD + Probe)")
        print("=" * 60)
        t0 = time.time()

        self.load_data()
        self.deduplicate_probes()
        self.merge_networks()

        # Export
        output_path = os.path.join(self.output_dir, "merged_network_phase4.gpkg")
        print(f"  Exporting to {output_path}...")
        out_gdf = self.merged_gdf.to_crs("EPSG:4326")
        out_gdf.to_file(output_path, driver="GPKG")

        elapsed = time.time() - t0
        total_km = self.merged_gdf.geometry.length.sum() / 1000.0
        n_vpd = (self.merged_gdf["source"] == "VPD").sum()
        n_probe = (self.merged_gdf["source"] == "Probe").sum()

        print(f"\nPhase 4 complete in {elapsed:.1f}s")
        print(f"  Merged network: {len(self.merged_gdf)} segments ({total_km:.1f} km)")
        print(f"    VPD: {n_vpd}, Probe: {n_probe}")

        return self.merged_gdf


if __name__ == "__main__":
    VPD_SKEL = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
    PROBE_SKEL = os.path.join(PROJECT_ROOT, "data", "interim_probe_skeleton_phase3.gpkg")

    merger = GraphMerger(VPD_SKEL, PROBE_SKEL)
    merger.run()
