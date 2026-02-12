"""
Pipeline Diagnostics — Find exactly where recovery is lost and why segments are disconnected.

This script analyzes:
1. Phase-by-phase segment statistics
2. Where segments are being removed and why
3. Connectivity analysis (why are there disconnected lines?)
4. Parameter sensitivity analysis
5. Specific recommendations with tunable parameters
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter, defaultdict
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely import STRtree

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def load_gdf(path, name):
    """Load a GeoPackage file."""
    if not os.path.exists(path):
        print(f"  {name}: FILE NOT FOUND")
        return None
    gdf = gpd.read_file(path)
    print(f"  {name}: {len(gdf)} features")
    return gdf


def bearing_from_xy(x1, y1, x2, y2):
    """Compute bearing in degrees."""
    import math
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy))
    return (angle + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Minimum angle difference."""
    d = abs((a - b) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    return d


def analyze_connectivity(gdf, name, snap_radius=10.0):
    """
    Analyze network connectivity - find disconnected components and dead-ends.
    """
    print(f"\n{'='*60}")
    print(f"  CONNECTIVITY ANALYSIS: {name}")
    print(f"{'='*60}")
    
    if gdf is None or len(gdf) == 0:
        print("  No data to analyze")
        return {}
    
    # Ensure projected CRS for distance calculations
    if gdf.crs is None or gdf.crs.is_geographic:
        bounds = gdf.total_bounds
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        local_crs = f"+proj=aeqd +lat_0={cy} +lon_0={cx} +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
        gdf = gdf.to_crs(local_crs)
    
    # Collect endpoints
    endpoints = []
    segment_endpoints = {}  # seg_idx -> (start_coord, end_coord)
    
    for i, row in gdf.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            continue
        coords = list(geom.coords)
        start = (coords[0][0], coords[0][1])
        end = (coords[-1][0], coords[-1][1])
        endpoints.append(start)
        endpoints.append(end)
        segment_endpoints[i] = (start, end)
    
    if not endpoints:
        print("  No valid endpoints found")
        return {}
    
    # Cluster endpoints
    ep_arr = np.array(endpoints)
    tree = cKDTree(ep_arr)
    
    # Count endpoint occurrences at each location (rounded)
    ep_counts = Counter()
    for x, y in endpoints:
        key = (round(x, 1), round(y, 1))
        ep_counts[key] += 1
    
    # Identify dead-ends (degree 1)
    dead_ends = [k for k, v in ep_counts.items() if v == 1]
    junctions = [k for k, v in ep_counts.items() if v >= 3]
    
    print(f"\n  Endpoint Statistics:")
    print(f"    Total segments: {len(segment_endpoints)}")
    print(f"    Total endpoints: {len(endpoints)}")
    print(f"    Unique locations: {len(ep_counts)}")
    print(f"    Dead-ends (degree 1): {len(dead_ends)}")
    print(f"    Junctions (degree 3+): {len(junctions)}")
    
    # Analyze dead-end distances to nearest other endpoint
    dead_end_distances = []
    for (dx, dy) in dead_ends:
        # Find nearest endpoint that isn't at this same location
        dists, idxs = tree.query([dx, dy], k=10)
        for d, idx in zip(dists, idxs):
            other_x, other_y = ep_arr[idx]
            other_key = (round(other_x, 1), round(other_y, 1))
            if other_key != (round(dx, 1), round(dy, 1)):
                dead_end_distances.append(d)
                break
    
    if dead_end_distances:
        de_arr = np.array(dead_end_distances)
        print(f"\n  Dead-end Gap Analysis (distance to nearest other endpoint):")
        print(f"    Min gap: {de_arr.min():.1f}m")
        print(f"    Max gap: {de_arr.max():.1f}m")
        print(f"    Mean gap: {de_arr.mean():.1f}m")
        print(f"    Median gap: {np.median(de_arr):.1f}m")
        print(f"    Gaps < 5m: {np.sum(de_arr < 5)}")
        print(f"    Gaps 5-10m: {np.sum((de_arr >= 5) & (de_arr < 10))}")
        print(f"    Gaps 10-15m: {np.sum((de_arr >= 10) & (de_arr < 15))}")
        print(f"    Gaps 15-25m: {np.sum((de_arr >= 15) & (de_arr < 25))}")
        print(f"    Gaps 25-50m: {np.sum((de_arr >= 25) & (de_arr < 50))}")
        print(f"    Gaps > 50m: {np.sum(de_arr >= 50)}")
    
    # Build connectivity graph using Union-Find
    parent = list(range(len(segment_endpoints)))
    seg_idx_list = list(segment_endpoints.keys())
    seg_to_idx = {seg: i for i, seg in enumerate(seg_idx_list)}
    
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # Connect segments that share endpoints (within snap radius)
    ep_to_segs = defaultdict(list)
    for seg_idx, (start, end) in segment_endpoints.items():
        start_key = (round(start[0], 1), round(start[1], 1))
        end_key = (round(end[0], 1), round(end[1], 1))
        ep_to_segs[start_key].append(seg_idx)
        ep_to_segs[end_key].append(seg_idx)
    
    for ep_key, segs in ep_to_segs.items():
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if segs[i] in seg_to_idx and segs[j] in seg_to_idx:
                    union(seg_to_idx[segs[i]], seg_to_idx[segs[j]])
    
    # Count connected components
    components = defaultdict(list)
    for i, seg_idx in enumerate(seg_idx_list):
        components[find(i)].append(seg_idx)
    
    component_sizes = sorted([len(v) for v in components.values()], reverse=True)
    
    print(f"\n  Connected Components:")
    print(f"    Total components: {len(components)}")
    if component_sizes:
        print(f"    Largest component: {component_sizes[0]} segments")
        if len(component_sizes) > 1:
            print(f"    2nd largest: {component_sizes[1]} segments")
        print(f"    Single-segment components: {sum(1 for s in component_sizes if s == 1)}")
        print(f"    Components with 2-5 segments: {sum(1 for s in component_sizes if 2 <= s <= 5)}")
        print(f"    Components with 6-20 segments: {sum(1 for s in component_sizes if 6 <= s <= 20)}")
    
    return {
        "total_segments": len(segment_endpoints),
        "dead_ends": len(dead_ends),
        "junctions": len(junctions),
        "components": len(components),
        "largest_component": component_sizes[0] if component_sizes else 0,
        "single_segment_components": sum(1 for s in component_sizes if s == 1),
        "dead_end_distances": dead_end_distances,
    }


def analyze_segment_lengths(gdf, name):
    """Analyze segment length distribution."""
    print(f"\n{'='*60}")
    print(f"  SEGMENT LENGTH ANALYSIS: {name}")
    print(f"{'='*60}")
    
    if gdf is None or len(gdf) == 0:
        return {}
    
    # Get projected lengths
    if gdf.crs is None or gdf.crs.is_geographic:
        bounds = gdf.total_bounds
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        local_crs = f"+proj=aeqd +lat_0={cy} +lon_0={cx} +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
        gdf = gdf.to_crs(local_crs)
    
    lengths = gdf.geometry.length.values
    
    print(f"\n  Length Distribution:")
    print(f"    Total segments: {len(lengths)}")
    print(f"    Total length: {lengths.sum()/1000:.1f} km")
    print(f"    Min: {lengths.min():.1f}m")
    print(f"    Max: {lengths.max():.1f}m")
    print(f"    Mean: {lengths.mean():.1f}m")
    print(f"    Median: {np.median(lengths):.1f}m")
    print(f"\n  Length Buckets:")
    print(f"    < 5m: {np.sum(lengths < 5)} ({100*np.sum(lengths < 5)/len(lengths):.1f}%)")
    print(f"    5-10m: {np.sum((lengths >= 5) & (lengths < 10))} ({100*np.sum((lengths >= 5) & (lengths < 10))/len(lengths):.1f}%)")
    print(f"    10-25m: {np.sum((lengths >= 10) & (lengths < 25))} ({100*np.sum((lengths >= 10) & (lengths < 25))/len(lengths):.1f}%)")
    print(f"    25-50m: {np.sum((lengths >= 25) & (lengths < 50))} ({100*np.sum((lengths >= 25) & (lengths < 50))/len(lengths):.1f}%)")
    print(f"    50-100m: {np.sum((lengths >= 50) & (lengths < 100))} ({100*np.sum((lengths >= 50) & (lengths < 100))/len(lengths):.1f}%)")
    print(f"    100-500m: {np.sum((lengths >= 100) & (lengths < 500))} ({100*np.sum((lengths >= 100) & (lengths < 500))/len(lengths):.1f}%)")
    print(f"    > 500m: {np.sum(lengths >= 500)} ({100*np.sum(lengths >= 500)/len(lengths):.1f}%)")
    
    return {
        "total_length_km": lengths.sum() / 1000,
        "mean_length": lengths.mean(),
        "median_length": np.median(lengths),
        "short_segments": np.sum(lengths < 10),
    }


def compare_with_ground_truth(generated_gdf, nav_gdf, buffer_m=15.0):
    """
    Detailed comparison with ground truth to find what's missing.
    """
    print(f"\n{'='*60}")
    print(f"  GROUND TRUTH COMPARISON")
    print(f"{'='*60}")
    
    if generated_gdf is None or nav_gdf is None:
        print("  Missing data for comparison")
        return {}
    
    # Project both to same CRS
    bounds = nav_gdf.total_bounds
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    local_crs = f"+proj=aeqd +lat_0={cy} +lon_0={cx} +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
    
    gen_proj = generated_gdf.to_crs(local_crs) if generated_gdf.crs != local_crs else generated_gdf
    nav_proj = nav_gdf.to_crs(local_crs) if nav_gdf.crs != local_crs else nav_gdf
    
    # Sample nav streets and check coverage
    nav_total_length = nav_proj.geometry.length.sum()
    gen_total_length = gen_proj.geometry.length.sum()
    
    print(f"\n  Total Lengths:")
    print(f"    Ground truth: {nav_total_length/1000:.1f} km")
    print(f"    Generated: {gen_total_length/1000:.1f} km")
    print(f"    Ratio: {100*gen_total_length/nav_total_length:.1f}%")
    
    # Sample points along nav streets
    sample_spacing = 20.0
    nav_samples = []
    
    for i, row in nav_proj.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString) or geom.is_empty:
            continue
        length = geom.length
        n_samples = max(1, int(length / sample_spacing))
        for j in range(n_samples):
            d = (j + 0.5) * length / n_samples
            pt = geom.interpolate(d)
            nav_samples.append((pt.x, pt.y, i))
    
    if not nav_samples:
        print("  No valid nav samples")
        return {}
    
    nav_sample_arr = np.array([(s[0], s[1]) for s in nav_samples])
    
    # Build spatial index of generated segments
    gen_geoms = list(gen_proj.geometry)
    gen_tree = STRtree(gen_geoms)
    
    # Check how many nav samples are covered
    covered = 0
    uncovered_by_distance = defaultdict(int)
    
    for (x, y, nav_idx) in nav_samples:
        pt = Point(x, y)
        nearby = gen_tree.query(pt.buffer(buffer_m))
        
        min_dist = float('inf')
        for idx in nearby:
            if idx < len(gen_geoms):
                dist = pt.distance(gen_geoms[idx])
                min_dist = min(min_dist, dist)
        
        if min_dist <= buffer_m:
            covered += 1
        else:
            # Bucket by distance
            if min_dist < 25:
                uncovered_by_distance["15-25m"] += 1
            elif min_dist < 50:
                uncovered_by_distance["25-50m"] += 1
            elif min_dist < 100:
                uncovered_by_distance["50-100m"] += 1
            else:
                uncovered_by_distance[">100m"] += 1
    
    coverage = 100 * covered / len(nav_samples)
    
    print(f"\n  Sample-based Coverage (buffer={buffer_m}m):")
    print(f"    Nav samples: {len(nav_samples)}")
    print(f"    Covered: {covered} ({coverage:.1f}%)")
    print(f"    Uncovered: {len(nav_samples) - covered} ({100-coverage:.1f}%)")
    
    if uncovered_by_distance:
        print(f"\n  Uncovered samples by nearest generated segment:")
        for k, v in sorted(uncovered_by_distance.items()):
            print(f"    {k}: {v}")
    
    return {
        "nav_total_km": nav_total_length / 1000,
        "gen_total_km": gen_total_length / 1000,
        "coverage_pct": coverage,
    }


def analyze_phase2_losses(skeleton_gdf):
    """
    Analyze what Phase 2 produced - check for issues in centerline generation.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2 OUTPUT ANALYSIS")
    print(f"{'='*60}")
    
    if skeleton_gdf is None:
        return {}
    
    # Check for attributes
    cols = skeleton_gdf.columns.tolist()
    print(f"\n  Available columns: {cols}")
    
    # Check support distribution if available
    if "weighted_support" in cols:
        supports = skeleton_gdf["weighted_support"].values
        print(f"\n  Weighted Support Distribution:")
        print(f"    Min: {supports.min():.2f}")
        print(f"    Max: {supports.max():.2f}")
        print(f"    Mean: {supports.mean():.2f}")
        print(f"    Support < 1: {np.sum(supports < 1)}")
        print(f"    Support 1-2: {np.sum((supports >= 1) & (supports < 2))}")
        print(f"    Support 2-5: {np.sum((supports >= 2) & (supports < 5))}")
        print(f"    Support > 5: {np.sum(supports >= 5)}")
    
    if "source" in cols:
        sources = skeleton_gdf["source"].value_counts()
        print(f"\n  Source Distribution:")
        for src, cnt in sources.items():
            print(f"    {src}: {cnt}")
    
    return {}


def analyze_phase3_losses(phase2_gdf, phase3_gdf):
    """
    Analyze what's lost between Phase 2 and Phase 3.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2 → PHASE 3 LOSS ANALYSIS")
    print(f"{'='*60}")
    
    if phase2_gdf is None or phase3_gdf is None:
        print("  Missing data")
        return {}
    
    # Project both
    bounds = phase2_gdf.total_bounds
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    local_crs = f"+proj=aeqd +lat_0={cy} +lon_0={cx} +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
    
    p2 = phase2_gdf.to_crs(local_crs)
    p3 = phase3_gdf.to_crs(local_crs)
    
    p2_length = p2.geometry.length.sum()
    p3_length = p3.geometry.length.sum()
    
    print(f"\n  Segment Counts:")
    print(f"    Phase 2: {len(p2)} segments")
    print(f"    Phase 3: {len(p3)} segments")
    print(f"    Lost: {len(p2) - len(p3)} segments ({100*(len(p2)-len(p3))/len(p2):.1f}%)")
    
    print(f"\n  Total Length:")
    print(f"    Phase 2: {p2_length/1000:.1f} km")
    print(f"    Phase 3: {p3_length/1000:.1f} km")
    print(f"    Lost: {(p2_length-p3_length)/1000:.1f} km ({100*(p2_length-p3_length)/p2_length:.1f}%)")
    
    # Sample P2 and check what's NOT in P3
    p2_samples = []
    for i, row in p2.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString) or geom.is_empty:
            continue
        length = geom.length
        if length < 1:
            continue
        n = max(1, int(length / 10))
        for j in range(n):
            d = (j + 0.5) * length / n
            pt = geom.interpolate(d)
            support = row.get("weighted_support", 1.0) if "weighted_support" in p2.columns else 1.0
            seg_len = length
            p2_samples.append((pt.x, pt.y, i, support, seg_len))
    
    # Check coverage in P3
    p3_geoms = list(p3.geometry)
    p3_tree = STRtree(p3_geoms)
    buffer = 10.0
    
    lost_samples = []
    for (x, y, seg_idx, support, seg_len) in p2_samples:
        pt = Point(x, y)
        nearby = p3_tree.query(pt.buffer(buffer))
        
        found = False
        for idx in nearby:
            if idx < len(p3_geoms) and pt.distance(p3_geoms[idx]) <= buffer:
                found = True
                break
        
        if not found:
            lost_samples.append((support, seg_len))
    
    if lost_samples:
        lost_supports = [s[0] for s in lost_samples]
        lost_lengths = [s[1] for s in lost_samples]
        
        print(f"\n  Lost Samples Analysis ({len(lost_samples)} of {len(p2_samples)} samples lost):")
        print(f"\n  Lost by Support:")
        print(f"    Support < 1: {sum(1 for s in lost_supports if s < 1)}")
        print(f"    Support 1-2: {sum(1 for s in lost_supports if 1 <= s < 2)}")
        print(f"    Support 2-5: {sum(1 for s in lost_supports if 2 <= s < 5)}")
        print(f"    Support > 5: {sum(1 for s in lost_supports if s >= 5)}")
        
        print(f"\n  Lost by Segment Length:")
        print(f"    < 10m: {sum(1 for l in lost_lengths if l < 10)}")
        print(f"    10-25m: {sum(1 for l in lost_lengths if 10 <= l < 25)}")
        print(f"    25-50m: {sum(1 for l in lost_lengths if 25 <= l < 50)}")
        print(f"    > 50m: {sum(1 for l in lost_lengths if l >= 50)}")
    
    return {
        "p2_segments": len(p2),
        "p3_segments": len(p3),
        "segments_lost": len(p2) - len(p3),
        "length_lost_km": (p2_length - p3_length) / 1000,
    }


def print_recommendations():
    """Print actionable recommendations with specific parameters."""
    print(f"\n{'='*60}")
    print(f"  TUNABLE PARAMETERS & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 PARAMETERS (src/pipeline_phase2.py - KharitaConfig)                │
├─────────────────────────────────────────────────────────────────────────────┤
│ RECOVERY IMPROVEMENT:                                                       │
│   min_edge_support: float = 0.5  → Try 0.3 for more roads                  │
│   candidate_selection_threshold: float = 0.18 → Try 0.10 for more coverage │
│   candidate_dangling_max_length_m: float = 45.0 → Try 80.0                 │
│   candidate_dangling_min_weighted_support: float = 3.0 → Try 1.5           │
│   min_centerline_length_m: float = 6.0 → Try 3.0                           │
│                                                                             │
│ CONNECTIVITY:                                                               │
│   cluster_radius_m: float = 12.0 → Try 15.0 for better endpoint clustering │
│   heading_tolerance_deg: float = 60.0 → OK                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3 PARAMETERS (src/pipeline_phase3.py - RefinementConfig)             │
├─────────────────────────────────────────────────────────────────────────────┤
│ LESS AGGRESSIVE PRUNING:                                                    │
│   spur_max_length_m: float = 15.0 → Try 8.0 (removes less)                 │
│   stub_threshold_m: float = 4.0 → Try 3.0                                  │
│   stub_max_iterations: int = 2 → Try 1                                     │
│   parallel_min_overlap: float = 0.75 → Try 0.85 (merge less aggressively)  │
│                                                                             │
│ BETTER STITCHING:                                                           │
│   endpoint_snap_radius_m: float = 8.0 → Try 12.0                           │
│   stitch_snap_radius_m: float = 15.0 → Try 20.0                            │
│   stitch_max_angle_deg: float = 70.0 → Try 90.0                            │
│   stitch_min_length_m: float = 5.0 → Try 3.0                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ ROOT CAUSE ANALYSIS:                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ IF dead-end gaps are mostly 10-25m:                                         │
│   → Increase stitch_snap_radius_m to 25m                                   │
│   → The stitching is too conservative                                       │
│                                                                             │
│ IF many segments lost from P2→P3:                                           │
│   → Reduce spur_max_length_m and stub_threshold_m                          │
│   → The cleanup is removing valid roads                                     │
│                                                                             │
│ IF P2 already has low coverage:                                             │
│   → Reduce min_edge_support and candidate_selection_threshold              │
│   → Phase 2 is filtering out valid roads too early                         │
│                                                                             │
│ IF many single-segment components:                                          │
│   → Increase endpoint_snap_radius_m                                        │
│   → Endpoints aren't being connected properly                               │
└─────────────────────────────────────────────────────────────────────────────┘
""")


def main():
    print("=" * 60)
    print("  PIPELINE DIAGNOSTICS")
    print("=" * 60)
    
    # Load all data files
    print("\n  Loading data files...")
    
    nav_gdf = load_gdf(
        os.path.join(PROJECT_ROOT, "data", "Kosovo_nav_streets", "nav_kosovo.gpkg"),
        "Nav Streets (Ground Truth)"
    )
    
    phase2_gdf = load_gdf(
        os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg"),
        "Phase 2 Output"
    )
    
    phase3_gdf = load_gdf(
        os.path.join(PROJECT_ROOT, "data", "interim_refined_skeleton_phase3.gpkg"),
        "Phase 3 Output"
    )
    
    phase4_gdf = load_gdf(
        os.path.join(PROJECT_ROOT, "data", "final_centerline_output_4326.gpkg"),
        "Phase 4 Output (Final)"
    )
    
    # Run analyses
    p2_conn = analyze_connectivity(phase2_gdf, "Phase 2")
    p3_conn = analyze_connectivity(phase3_gdf, "Phase 3")
    p4_conn = analyze_connectivity(phase4_gdf, "Phase 4 (Final)")
    
    analyze_segment_lengths(phase2_gdf, "Phase 2")
    analyze_segment_lengths(phase3_gdf, "Phase 3")
    
    analyze_phase2_losses(phase2_gdf)
    analyze_phase3_losses(phase2_gdf, phase3_gdf)
    
    if nav_gdf is not None:
        compare_with_ground_truth(phase2_gdf, nav_gdf, buffer_m=15.0)
        compare_with_ground_truth(phase4_gdf, nav_gdf, buffer_m=15.0)
    
    # Summary and recommendations
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    if p2_conn and p3_conn and p4_conn:
        print(f"\n  Connectivity Progression:")
        print(f"    {'Phase':<15} {'Segments':>10} {'Dead-ends':>12} {'Components':>12}")
        print(f"    {'-'*15} {'-'*10} {'-'*12} {'-'*12}")
        print(f"    {'Phase 2':<15} {p2_conn.get('total_segments', 0):>10} {p2_conn.get('dead_ends', 0):>12} {p2_conn.get('components', 0):>12}")
        print(f"    {'Phase 3':<15} {p3_conn.get('total_segments', 0):>10} {p3_conn.get('dead_ends', 0):>12} {p3_conn.get('components', 0):>12}")
        print(f"    {'Phase 4':<15} {p4_conn.get('total_segments', 0):>10} {p4_conn.get('dead_ends', 0):>12} {p4_conn.get('components', 0):>12}")
    
    print_recommendations()
    
    print("\n  Diagnostics complete!")


if __name__ == "__main__":
    main()
