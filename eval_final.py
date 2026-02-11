"""Simple metrics evaluation."""
import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point
import os

def segment_lines(geoms, seg_len=20.0):
    segments = []
    lengths = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        line_len = geom.length
        if line_len <= seg_len:
            segments.append(geom)
            lengths.append(line_len)
            continue
        coords = list(geom.coords)
        current = [coords[0]]
        current_len = 0.0
        for k in range(1, len(coords)):
            seg_d = Point(coords[k]).distance(Point(coords[k - 1]))
            current.append(coords[k])
            current_len += seg_d
            if current_len >= seg_len:
                if len(current) >= 2:
                    seg = LineString(current)
                    segments.append(seg)
                    lengths.append(seg.length)
                current = [coords[k]]
                current_len = 0.0
        if len(current) >= 2:
            seg = LineString(current)
            if seg.length > 0:
                segments.append(seg)
                lengths.append(seg.length)
    return segments, np.array(lengths)


def evaluate(gen, nav, buffer_m=15.0):
    gen_segs, gen_lens = segment_lines(gen.geometry.values)
    nav_segs, nav_lens = segment_lines(nav.geometry.values)
    gen_total = gen_lens.sum()
    nav_total = nav_lens.sum()

    nav_tree = STRtree(nav_segs)
    hit_gen, _ = nav_tree.query(gen_segs, predicate="dwithin", distance=buffer_m)
    covered_gen = np.unique(hit_gen)
    prec_len = gen_lens[covered_gen].sum() if len(covered_gen) > 0 else 0

    gen_tree = STRtree(gen_segs)
    hit_nav, _ = gen_tree.query(nav_segs, predicate="dwithin", distance=buffer_m)
    covered_nav = np.unique(hit_nav)
    rec_len = nav_lens[covered_nav].sum() if len(covered_nav) > 0 else 0

    recovery = rec_len / nav_total * 100 if nav_total > 0 else 0
    precision = prec_len / gen_total * 100 if gen_total > 0 else 0
    return recovery, precision, gen_total / 1000


if __name__ == "__main__":
    os.chdir("E:/Coding projects/HERE-Geospatial-Hackathon")
    
    nav = gpd.read_file("data/Kosovo_nav_streets/nav_kosovo.gpkg").to_crs("EPSG:32634")
    nav = nav[nav.geometry.notna() & ~nav.geometry.is_empty]
    print(f"Nav: {len(nav)} links, {nav.geometry.length.sum()/1000:.1f} km\n")
    
    # Evaluate final output
    final_path = "data/final_centerline_output_4326.gpkg"
    if os.path.exists(final_path):
        final = gpd.read_file(final_path).to_crs("EPSG:32634")
        final = final[final.geometry.notna() & ~final.geometry.is_empty]
        rec, prec, gen_km = evaluate(final, nav)
        print("=" * 60)
        print("FINAL OUTPUT METRICS (Phase 5)")
        print("=" * 60)
        print(f"  Recovery (recall):  {rec:.1f}%")
        print(f"  Precision:          {prec:.1f}%")
        print(f"  Generated:          {gen_km:.1f} km")
        print(f"  Ground truth:       {nav.geometry.length.sum()/1000:.1f} km")
        print()
        
        # Per func_class
        if "func_class" in nav.columns:
            print("Recovery by road class:")
            for fc in sorted(nav["func_class"].unique()):
                fc_nav = nav[nav["func_class"] == fc]
                rec_fc, _, _ = evaluate(final, fc_nav)
                print(f"  func_class {fc}: {rec_fc:.1f}% ({fc_nav.geometry.length.sum()/1000:.1f} km)")
    else:
        print(f"Final output not found: {final_path}")
