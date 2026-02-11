"""
Diagnostic script to understand why roundabout detection is failing.
"""

import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from src.algorithms.roundabout_detection import (
    RoundaboutConfig,
    RoundaboutDetector,
)


def diagnose_roundabout_detection():
    """Check each step of roundabout detection."""
    print("=" * 60)
    print("  ROUNDABOUT DETECTION DIAGNOSTIC")
    print("=" * 60)
    
    # Load VPD data
    input_path = "data/interim_sample_phase1.gpkg"
    print(f"\n1. Loading VPD from {input_path}...")
    gdf = gpd.read_file(input_path)
    print(f"   Total traces: {len(gdf)}")
    
    # Filter valid LineStrings
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & (gdf.geom_type == "LineString")
    ].reset_index(drop=True)
    print(f"   Valid LineString traces: {len(gdf)}")
    
    # Set up projection
    bounds = gdf.total_bounds
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0
    
    crs_projected = CRS.from_proj4(
        f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
        f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )
    
    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, crs_projected, always_xy=True)
    
    # Prepare traces
    print("\n2. Preparing traces for detection...")
    traces = []
    trace_lengths = []
    trace_points = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        coords_wgs = np.array(geom.coords)
        coords_xy = np.column_stack(
            to_proj.transform(coords_wgs[:, 0], coords_wgs[:, 1])
        )
        
        if len(coords_xy) < 4:
            continue
        
        # Compute trace length
        dx = np.diff(coords_xy[:, 0])
        dy = np.diff(coords_xy[:, 1])
        length = np.sum(np.sqrt(dx**2 + dy**2))
        
        trace_lengths.append(length)
        trace_points.append(len(coords_xy))
        traces.append({"coords": coords_xy})
    
    print(f"   Valid traces for detection: {len(traces)}")
    print(f"   Avg points per trace: {np.mean(trace_points):.1f}")
    print(f"   Avg trace length: {np.mean(trace_lengths):.1f}m")
    print(f"   Max trace length: {np.max(trace_lengths):.1f}m")
    
    # Configure detector with more lenient settings for diagnosis
    print("\n3. Running arc extraction analysis...")
    ra_config = RoundaboutConfig(
        min_radius_m=6.0,
        max_radius_m=60.0,
        min_unique_traces=2,
        cluster_radius_m=30.0,
        # Relaxed arc detection for fine-sampled GPS data
        min_arc_heading_deg=60.0,
        min_arc_segments=3,
        max_radius_std_ratio=0.35,
        min_arc_bbox_m=5.0,
        arc_zero_threshold_rad=0.015,
        arc_min_segment_length_m=0.2,
        arc_max_zero_run=5,
        # Relaxed validation thresholds
        min_avg_heading_deg=60.0,
        min_entry_directions=1,
        direction_separation_deg=45.0,
        min_arcs_for_radius_only=4,
    )
    detector = RoundaboutDetector(ra_config)
    
    # Manually extract arcs to understand filtering
    all_arcs = []
    arc_heading_changes = []
    arc_lengths = []
    
    for trace in traces:
        coords = trace["coords"]
        arcs = detector._extract_curved_arcs(coords)
        
        for arc_coords, heading_change in arcs:
            all_arcs.append((arc_coords, heading_change))
            arc_heading_changes.append(np.degrees(abs(heading_change)))
            
            dx = np.diff(arc_coords[:, 0])
            dy = np.diff(arc_coords[:, 1])
            arc_lengths.append(np.sum(np.sqrt(dx**2 + dy**2)))
    
    print(f"   Total arcs extracted: {len(all_arcs)}")
    if all_arcs:
        print(f"   Avg heading change: {np.mean(arc_heading_changes):.1f}°")
        print(f"   Max heading change: {np.max(arc_heading_changes):.1f}°")
        print(f"   Arcs with heading > 90°: {sum(1 for h in arc_heading_changes if h >= 90)}")
        print(f"   Arcs with heading > 45°: {sum(1 for h in arc_heading_changes if h >= 45)}")
        print(f"   Avg arc length: {np.mean(arc_lengths):.1f}m")
    
    # Check candidate roundabouts before filtering
    print("\n4. Checking circle fitting candidates (with detailed failure analysis)...")
    candidates = []
    
    # Track failure reasons
    fail_heading = 0
    fail_points = 0
    fail_circle_fit = 0
    fail_radius_small = 0
    fail_radius_large = 0
    fail_radius_consistency = 0
    fail_bbox = 0
    
    fitted_radii = []
    arc_bboxes = []
    
    # Use config value for heading threshold
    min_heading_rad = np.radians(ra_config.min_arc_heading_deg)
    
    for arc_coords, arc_heading_change in all_arcs:
        # Skip arcs with insufficient heading change
        if abs(arc_heading_change) < min_heading_rad:
            fail_heading += 1
            continue
        
        n = len(arc_coords)
        if n < 3:
            fail_points += 1
            continue
        
        p1 = arc_coords[0]
        p2 = arc_coords[n // 2]
        p3 = arc_coords[-1]
        
        result = detector._fit_circle_3pt(p1, p2, p3)
        if result is None:
            fail_circle_fit += 1
            continue
        
        cx, cy, radius = result
        fitted_radii.append(radius)
        
        # Check radius
        if radius < ra_config.min_radius_m:
            fail_radius_small += 1
            continue
        if radius > ra_config.max_radius_m:
            fail_radius_large += 1
            continue
        
        # Verify arc points lie on this circle
        dists = np.sqrt(
            (arc_coords[:, 0] - cx)**2 + (arc_coords[:, 1] - cy)**2
        )
        dist_std = np.std(dists)
        
        if dist_std > ra_config.max_radius_std_ratio * radius:
            fail_radius_consistency += 1
            continue
        
        # Arc bounding box check (use config value)
        min_bbox = getattr(ra_config, 'min_arc_bbox_m', 5.0)
        bbox_w = np.ptp(arc_coords[:, 0])
        bbox_h = np.ptp(arc_coords[:, 1])
        arc_bboxes.append(max(bbox_w, bbox_h))
        if max(bbox_w, bbox_h) < min_bbox:
            fail_bbox += 1
            continue
        
        candidates.append({
            "center": (cx, cy),
            "radius": radius,
            "heading_change": np.degrees(abs(arc_heading_change)),
        })
    
    print(f"   Failure breakdown:")
    print(f"     - Heading < {ra_config.min_arc_heading_deg}°:  {fail_heading}")
    print(f"     - Too few points:       {fail_points}")
    print(f"     - Circle fit failed:    {fail_circle_fit}")
    print(f"     - Radius too small:     {fail_radius_small}")
    print(f"     - Radius too large:     {fail_radius_large}")
    print(f"     - Radius inconsistent:  {fail_radius_consistency}")
    print(f"     - Bbox too small (<{min_bbox}m): {fail_bbox}")
    print(f"   Passed all filters: {len(candidates)}")
    
    if fitted_radii:
        print(f"\n   Fitted radii stats (before radius filter):")
        print(f"     - Mean: {np.mean(fitted_radii):.1f}m")
        print(f"     - Median: {np.median(fitted_radii):.1f}m")
        print(f"     - Min: {np.min(fitted_radii):.1f}m")
        print(f"     - Max: {np.max(fitted_radii):.1f}m")
        print(f"     - In range ({ra_config.min_radius_m}-{ra_config.max_radius_m}m): {sum(1 for r in fitted_radii if ra_config.min_radius_m <= r <= ra_config.max_radius_m)}")
    
    if arc_bboxes:
        min_bbox = getattr(ra_config, 'min_arc_bbox_m', 5.0)
        print(f"\n   Arc bbox stats (arcs that passed radius check):")
        print(f"     - Mean bbox size: {np.mean(arc_bboxes):.1f}m")
        print(f"     - Max bbox size: {np.max(arc_bboxes):.1f}m")
        print(f"     - Bboxes >= {min_bbox}m: {sum(1 for b in arc_bboxes if b >= min_bbox)}")
    
    print(f"   Circle fit candidates (>={ra_config.min_arc_heading_deg}° heading, good radius): {len(candidates)}")
    if candidates:
        for i, c in enumerate(candidates[:10]):
            print(f"     #{i+1}: center=({c['center'][0]:.1f}, {c['center'][1]:.1f}), "
                  f"radius={c['radius']:.1f}m, heading={c['heading_change']:.1f}°")
    
    # Run full detection
    print("\n5. Running full detection...")
    roundabouts = detector.detect(traces, use_curl=False, use_arc=True)
    print(f"   Detected roundabouts: {len(roundabouts)}")
    
    # Now test with relaxed settings
    print("\n6. Testing with relaxed settings...")
    relaxed_config = RoundaboutConfig(
        min_radius_m=5.0,
        max_radius_m=70.0,
        min_unique_traces=1,
        cluster_radius_m=35.0,
        min_arc_heading_deg=45.0,  # Reduced from 60
        min_arc_segments=2,         # Reduced from 3
        max_radius_std_ratio=0.40,  # Relaxed from 0.35
        min_arc_bbox_m=3.0,  # Reduced from 5
        arc_zero_threshold_rad=0.010,  # Relaxed further
        arc_min_segment_length_m=0.15,  # Relaxed further
        arc_max_zero_run=8,  # Relaxed further
        min_avg_heading_deg=45.0,  # Reduced from 60
        min_entry_directions=1,  # Reduced from 2
        direction_separation_deg=30.0,  # Reduced from 45
        min_arcs_for_radius_only=2,  # Reduced from 4
    )
    
    relaxed_detector = RoundaboutDetector(relaxed_config)
    relaxed_roundabouts = relaxed_detector.detect(traces, use_curl=True, use_arc=True)  # Both methods
    print(f"   Detected roundabouts (relaxed): {len(relaxed_roundabouts)}")
    
    # Check arcs with lower threshold
    print("\n7. Arcs with lower heading thresholds...")
    arcs_45_plus = sum(1 for h in arc_heading_changes if h >= 45)
    arcs_60_plus = sum(1 for h in arc_heading_changes if h >= 60)
    arcs_90_plus = sum(1 for h in arc_heading_changes if h >= 90)
    arcs_120_plus = sum(1 for h in arc_heading_changes if h >= 120)
    
    print(f"   Arcs with heading ≥ 45°: {arcs_45_plus}")
    print(f"   Arcs with heading ≥ 60°: {arcs_60_plus}")
    print(f"   Arcs with heading ≥ 90°: {arcs_90_plus}")
    print(f"   Arcs with heading ≥ 120°: {arcs_120_plus}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 60)
    if len(all_arcs) == 0:
        print("  ISSUE: No arcs extracted from traces!")
        print("  - Check if traces have enough curvature")
        print("  - Lower min_arc_segments or adjust heading threshold")
    elif len(candidates) == 0:
        print("  ISSUE: Arcs found but no circle fit candidates!")
        print("  - Arcs might not have enough heading change (need 90°)")
        print("  - Circle radius might be out of range")
        print("  - Consider lowering min_arc_heading_deg to 45°")
    elif len(roundabouts) == 0:
        print("  ISSUE: Candidates found but none passed cluster validation!")
        print("  - Need 4+ unique traces or 6+ arcs per cluster")
        print("  - Need multiple entry directions")
        print("  - Consider relaxing min_unique_traces to 2")


if __name__ == "__main__":
    diagnose_roundabout_detection()
