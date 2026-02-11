"""Quick metrics evaluation of current pipeline outputs."""
import geopandas as gpd
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point


def segment_lines(geoms, seg_len=20.0):
    """Split lines into ~seg_len metre segments for accurate coverage."""
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
    """Compute segmented precision & recall."""
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
    return recovery, precision, gen_total / 1000, nav_total / 1000


if __name__ == "__main__":
    # Load
    nav = gpd.read_file("data/Kosovo_nav_streets/nav_kosovo.gpkg").to_crs("EPSG:32634")
    print(f"Nav: {len(nav)} links, {nav.geometry.length.sum()/1000:.1f} km")

    # Evaluate each phase
    phases = [
        ("Phase 2 Skeleton", "data/interim_skeleton_phase2.gpkg"),
        ("Phase 3 Probe", "data/interim_probe_skeleton_phase3.gpkg"),
        ("Phase 4 Merged", "data/merged_network_phase4.gpkg"),
        ("Phase 5 Final", "data/final_centerline_output_4326.gpkg"),
    ]

    print()
    header = f"{'Phase':<25} {'Recovery%':>10} {'Precision%':>10} {'Gen(km)':>10}"
    print(header)
    print("-" * 60)

    for name, path in phases:
        try:
            gdf = gpd.read_file(path).to_crs("EPSG:32634")
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
            rec, prec, gen_km, _ = evaluate(gdf, nav)
            print(f"{name:<25} {rec:>10.1f} {prec:>10.1f} {gen_km:>10.1f}")
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")

    # Also check func_class breakdown if available
    if "func_class" in nav.columns:
        print()
        print("Recovery by func_class (road hierarchy):")
        for fc in sorted(nav["func_class"].unique()):
            fc_nav = nav[nav["func_class"] == fc]
            try:
                final = gpd.read_file("data/final_centerline_output_4326.gpkg").to_crs("EPSG:32634")
                final = final[final.geometry.notna() & ~final.geometry.is_empty]
                rec, _, _, _ = evaluate(final, fc_nav)
                print(f"  func_class {fc}: {rec:.1f}% ({fc_nav.geometry.length.sum()/1000:.1f} km)")
            except Exception as e:
                print(f"  func_class {fc}: ERROR - {e}")
