"""
Pipeline Diagnostics — Comprehensive analysis to identify precision/recovery bottlenecks.

This script analyzes each phase output against Nav Streets ground truth
to pinpoint exactly WHERE accuracy is lost and WHY.

Diagnostic Categories:
  D1. VPD Coverage Gap Analysis — Why is recovery only ~49%?
  D2. Probe Noise Analysis — How much noise do probes add?
  D3. Phase-over-Phase Degradation — Where do we lose precision?
  D4. Phase 5 Damage Analysis — Why does Phase 5 reduce precision?
  D5. Line Length & Source Distribution — What kinds of lines are problematic?
  D6. Spatial Coverage Analysis — WHERE are the gaps?
"""

import os
import sys
import time
import warnings
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import unary_union
from shapely import STRtree

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

METRIC_CRS = "EPSG:32634"  # UTM 34N for Kosovo

# ─── File paths ───────────────────────────────────────────────────────
NAV_PATH = os.path.join(PROJECT_ROOT, "data", "Kosovo_nav_streets", "nav_kosovo.gpkg")
VPD_PHASE1 = os.path.join(PROJECT_ROOT, "data", "interim_sample_phase1.gpkg")
HPD_PHASE1 = os.path.join(PROJECT_ROOT, "data", "interim_hpd_phase1.gpkg")
SKELETON_PHASE2 = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")
PROBE_PHASE3 = os.path.join(PROJECT_ROOT, "data", "interim_probe_skeleton_phase3.gpkg")
MERGED_PHASE4 = os.path.join(PROJECT_ROOT, "data", "merged_network_phase4.gpkg")
FINAL_PHASE5 = os.path.join(PROJECT_ROOT, "data", "final_centerline_output_4326.gpkg")

BUFFER_M = 15.0  # standard evaluation buffer


def load_gdf(path, label=""):
    """Load a GeoPackage and project to metric CRS."""
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(METRIC_CRS)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    if label:
        print(f"  {label}: {len(gdf)} features, {gdf.geometry.length.sum()/1000:.1f} km")
    return gdf


def segment_lines(geoms, seg_len=20.0):
    """Split lines into ~seg_len metre segments for accurate coverage analysis."""
    from shapely.geometry import Point
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


def segmented_precision_recall(gen_gdf, nav_gdf, buffer_m=15.0):
    """Compute segmented precision & recall."""
    gen_segs, gen_lens = segment_lines(gen_gdf.geometry.values)
    nav_segs, nav_lens = segment_lines(nav_gdf.geometry.values)
    gen_total = gen_lens.sum()
    nav_total = nav_lens.sum()

    if gen_total == 0 or nav_total == 0:
        return {"recovery": 0, "precision": 0, "gen_km": 0, "nav_km": 0}

    nav_tree = STRtree(nav_segs)
    hit_gen, _ = nav_tree.query(gen_segs, predicate="dwithin", distance=buffer_m)
    covered_gen = np.unique(hit_gen)
    prec_len = gen_lens[covered_gen].sum() if len(covered_gen) > 0 else 0

    gen_tree = STRtree(gen_segs)
    hit_nav, _ = gen_tree.query(nav_segs, predicate="dwithin", distance=buffer_m)
    covered_nav = np.unique(hit_nav)
    rec_len = nav_lens[covered_nav].sum() if len(covered_nav) > 0 else 0

    return {
        "recovery": round(rec_len / nav_total * 100, 2),
        "precision": round(prec_len / gen_total * 100, 2),
        "gen_km": round(gen_total / 1000, 2),
        "nav_km": round(nav_total / 1000, 2),
        "noise_km": round((gen_total - prec_len) / 1000, 2),
        "missed_km": round((nav_total - rec_len) / 1000, 2),
    }


def classify_lines_by_proximity(gen_gdf, nav_gdf, buffer_m=15.0):
    """Classify each generated line as TP (near nav) or FP (noise)."""
    nav_segs, _ = segment_lines(nav_gdf.geometry.values)
    nav_tree = STRtree(nav_segs)

    tp_indices = []
    fp_indices = []
    for i, geom in enumerate(gen_gdf.geometry.values):
        hits = nav_tree.query(geom, predicate="dwithin", distance=buffer_m)
        if len(hits) > 0:
            tp_indices.append(i)
        else:
            fp_indices.append(i)
    return tp_indices, fp_indices


# ═══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC D1: VPD Coverage Gap Analysis
# ═══════════════════════════════════════════════════════════════════════
def diagnostic_d1_vpd_coverage(nav, vpd_raw, skeleton):
    print("\n" + "=" * 70)
    print("  D1: VPD COVERAGE GAP ANALYSIS")
    print("=" * 70)

    if nav is None or vpd_raw is None:
        print("  SKIP: Missing data")
        return {}

    results = {}

    # How many VPD traces were loaded?
    n_vpd = len(vpd_raw)
    vpd_km = vpd_raw.geometry.length.sum() / 1000
    results["vpd_traces"] = n_vpd
    results["vpd_total_km"] = round(vpd_km, 2)
    print(f"  VPD raw traces: {n_vpd} ({vpd_km:.1f} km)")

    # Nav streets total
    nav_km = nav.geometry.length.sum() / 1000
    results["nav_total_km"] = round(nav_km, 2)
    print(f"  Nav streets: {len(nav)} links ({nav_km:.1f} km)")

    # How much of nav is within VPD raw trace buffer? (use STRtree for speed)
    nav_segs, nav_seg_lens = segment_lines(nav.geometry.values)
    vpd_tree = STRtree(vpd_raw.geometry.values)
    hit_nav, _ = vpd_tree.query(nav_segs, predicate="dwithin", distance=BUFFER_M)
    covered_nav_idx = np.unique(hit_nav)
    covered_len = nav_seg_lens[covered_nav_idx].sum() if len(covered_nav_idx) > 0 else 0
    vpd_raw_coverage = covered_len / nav_seg_lens.sum() * 100 if nav_seg_lens.sum() > 0 else 0

    results["vpd_raw_nav_coverage_pct"] = round(vpd_raw_coverage, 2)
    print(f"  Nav coverage by RAW VPD traces (15m buffer): {vpd_raw_coverage:.1f}%")
    print(f"  → This is the theoretical MAXIMUM recovery from VPD alone")

    # Skeleton coverage
    if skeleton is not None:
        skel_result = segmented_precision_recall(skeleton, nav)
        results["skeleton_recovery"] = skel_result["recovery"]
        results["skeleton_precision"] = skel_result["precision"]
        results["skeleton_noise_km"] = skel_result["noise_km"]
        print(f"\n  VPD Skeleton (Phase 2):")
        print(f"    Recovery:  {skel_result['recovery']:.1f}%")
        print(f"    Precision: {skel_result['precision']:.1f}%")
        print(f"    Noise:     {skel_result['noise_km']:.1f} km of false lines")
        print(f"    Total:     {skel_result['gen_km']:.1f} km generated")

        # Gap between raw VPD coverage and skeleton recovery
        gap = vpd_raw_coverage - skel_result["recovery"]
        results["kde_gap_pct"] = round(gap, 2)
        print(f"\n  KDE/Skeleton processing gap: {gap:.1f}%")
        print(f"  → This much coverage is LOST during KDE rasterization")
        if gap > 10:
            print(f"  ⚠ SIGNIFICANT: KDE threshold may be too aggressive")

    # Nav streets by func_class if available
    if "func_class" in nav.columns:
        print(f"\n  Nav Streets by func_class:")
        for fc in sorted(nav["func_class"].unique()):
            fc_gdf = nav[nav["func_class"] == fc]
            fc_km = fc_gdf.geometry.length.sum() / 1000
            print(f"    FC {fc}: {len(fc_gdf)} links, {fc_km:.1f} km")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC D2: Probe Noise Analysis
# ═══════════════════════════════════════════════════════════════════════
def diagnostic_d2_probe_noise(nav, probe_skel, hpd_raw):
    print("\n" + "=" * 70)
    print("  D2: PROBE NOISE ANALYSIS")
    print("=" * 70)

    results = {}

    if probe_skel is None or nav is None:
        print("  SKIP: Missing probe skeleton or nav")
        return results

    # Probe skeleton stats
    n_probe = len(probe_skel)
    probe_km = probe_skel.geometry.length.sum() / 1000
    results["probe_lines"] = n_probe
    results["probe_km"] = round(probe_km, 2)
    print(f"  Probe skeleton: {n_probe} lines ({probe_km:.1f} km)")

    # Classify each probe line as TP or FP
    tp_idx, fp_idx = classify_lines_by_proximity(probe_skel, nav, buffer_m=BUFFER_M)
    tp_km = probe_skel.geometry.iloc[tp_idx].length.sum() / 1000 if tp_idx else 0
    fp_km = probe_skel.geometry.iloc[fp_idx].length.sum() / 1000 if fp_idx else 0

    results["probe_tp_lines"] = len(tp_idx)
    results["probe_fp_lines"] = len(fp_idx)
    results["probe_tp_km"] = round(tp_km, 2)
    results["probe_fp_km"] = round(fp_km, 2)
    results["probe_noise_pct"] = round(fp_km / probe_km * 100, 2) if probe_km > 0 else 0

    print(f"\n  True Positives (near nav streets):")
    print(f"    {len(tp_idx)} lines, {tp_km:.1f} km")
    print(f"  False Positives (NOISE — not near any nav street):")
    print(f"    {len(fp_idx)} lines, {fp_km:.1f} km")
    print(f"  Noise fraction: {results['probe_noise_pct']:.1f}%")

    # Length distribution of FP lines
    if fp_idx:
        fp_lengths = probe_skel.geometry.iloc[fp_idx].length.values
        print(f"\n  FP (noise) line length distribution:")
        print(f"    Min:    {fp_lengths.min():.1f} m")
        print(f"    Median: {np.median(fp_lengths):.1f} m")
        print(f"    Mean:   {fp_lengths.mean():.1f} m")
        print(f"    Max:    {fp_lengths.max():.1f} m")
        print(f"    < 20m:  {(fp_lengths < 20).sum()} lines")
        print(f"    < 50m:  {(fp_lengths < 50).sum()} lines")
        print(f"    < 100m: {(fp_lengths < 100).sum()} lines")
        results["fp_median_length_m"] = round(float(np.median(fp_lengths)), 1)

    # HPD raw trace analysis
    if hpd_raw is not None:
        n_hpd = len(hpd_raw)
        hpd_km = hpd_raw.geometry.length.sum() / 1000
        results["hpd_raw_traces"] = n_hpd
        results["hpd_raw_km"] = round(hpd_km, 2)
        print(f"\n  HPD raw probes: {n_hpd} traces ({hpd_km:.1f} km)")

        # Speed distribution if available
        if "avg_speed" in hpd_raw.columns:
            speeds = hpd_raw["avg_speed"].dropna()
            print(f"  Speed distribution:")
            print(f"    < 5 km/h:   {(speeds < 5).sum()} traces (pedestrian)")
            print(f"    5-20 km/h:  {((speeds >= 5) & (speeds < 20)).sum()} traces (slow)")
            print(f"    20-50 km/h: {((speeds >= 20) & (speeds < 50)).sum()} traces (residential)")
            print(f"    > 50 km/h:  {(speeds >= 50).sum()} traces (fast road)")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC D3: Phase-over-Phase Degradation
# ═══════════════════════════════════════════════════════════════════════
def diagnostic_d3_phase_degradation(nav, skeleton, probe, merged, final):
    print("\n" + "=" * 70)
    print("  D3: PHASE-OVER-PHASE DEGRADATION ANALYSIS")
    print("=" * 70)

    results = {}
    phases = [
        ("Phase 2: VPD Skeleton", skeleton),
        ("Phase 3: Probe Skeleton", probe),
        ("Phase 4: Merged", merged),
        ("Phase 5: Final", final),
    ]

    print(f"\n  {'Phase':<30} {'Recovery%':>10} {'Precision%':>10} {'Lines':>8} {'Gen km':>8} {'Noise km':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    prev_prec = None
    for label, gdf in phases:
        if gdf is None:
            print(f"  {label:<30} {'MISSING':>10}")
            continue
        m = segmented_precision_recall(gdf, nav)
        marker = ""
        if prev_prec is not None and m["precision"] < prev_prec:
            drop = prev_prec - m["precision"]
            marker = f" ⚠ -{drop:.1f}%"
        prev_prec = m["precision"]
        print(f"  {label:<30} {m['recovery']:>10.1f} {m['precision']:>10.1f} "
              f"{len(gdf):>8,} {m['gen_km']:>8.1f} {m['noise_km']:>8.1f}{marker}")
        results[label] = m

    # Identify the biggest precision drop
    if "Phase 4: Merged" in results and "Phase 2: VPD Skeleton" in results:
        drop_p4 = results["Phase 2: VPD Skeleton"]["precision"] - results["Phase 4: Merged"]["precision"]
        rec_gain = results["Phase 4: Merged"]["recovery"] - results["Phase 2: VPD Skeleton"]["recovery"]
        noise_added = results["Phase 4: Merged"]["noise_km"] - results["Phase 2: VPD Skeleton"]["noise_km"]
        print(f"\n  Phase 4 impact (merging probes):")
        print(f"    Precision drop: {drop_p4:.1f}%")
        print(f"    Recovery gain:  {rec_gain:.1f}%")
        print(f"    Noise added:    {noise_added:.1f} km")
        if rec_gain > 0:
            print(f"    Noise-to-recovery ratio: {noise_added/rec_gain:.1f} km noise per 1% recovery")

    if "Phase 5: Final" in results and "Phase 4: Merged" in results:
        drop_p5 = results["Phase 4: Merged"]["precision"] - results["Phase 5: Final"]["precision"]
        rec_change = results["Phase 5: Final"]["recovery"] - results["Phase 4: Merged"]["recovery"]
        print(f"\n  Phase 5 impact (optimization):")
        print(f"    Precision change: {-drop_p5:+.1f}%")
        print(f"    Recovery change:  {rec_change:+.1f}%")
        if drop_p5 > 0:
            print(f"    ⚠ Phase 5 is DEGRADING precision — it removes good lines or distorts geometry")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC D4: Phase 5 Damage Analysis
# ═══════════════════════════════════════════════════════════════════════
def diagnostic_d4_phase5_damage(nav, merged, final):
    print("\n" + "=" * 70)
    print("  D4: PHASE 5 DAMAGE ANALYSIS — What gets removed/distorted?")
    print("=" * 70)

    results = {}

    if merged is None or final is None or nav is None:
        print("  SKIP: Missing data")
        return results

    # Lines in Phase 4 but NOT in Phase 5 (removed lines)
    # Approximate: find merged lines that are NOT within 3m of any final line
    final_tree = STRtree(final.geometry.values)
    removed_indices = []
    kept_indices = []
    for i, geom in enumerate(merged.geometry.values):
        hits = final_tree.query(geom, predicate="dwithin", distance=5.0)
        if len(hits) == 0:
            removed_indices.append(i)
        else:
            kept_indices.append(i)

    removed = merged.iloc[removed_indices] if removed_indices else merged.iloc[:0]
    kept = merged.iloc[kept_indices] if kept_indices else merged.iloc[:0]

    removed_km = removed.geometry.length.sum() / 1000 if len(removed) > 0 else 0
    kept_km = kept.geometry.length.sum() / 1000 if len(kept) > 0 else 0

    print(f"  Merged (Phase 4): {len(merged)} lines ({merged.geometry.length.sum()/1000:.1f} km)")
    print(f"  Final (Phase 5):  {len(final)} lines ({final.geometry.length.sum()/1000:.1f} km)")
    print(f"  Removed by P5:    {len(removed)} lines ({removed_km:.1f} km)")

    # Were the removed lines good or bad?
    if len(removed) > 0:
        tp_rem, fp_rem = classify_lines_by_proximity(removed, nav, buffer_m=BUFFER_M)
        tp_rem_km = removed.geometry.iloc[tp_rem].length.sum() / 1000 if tp_rem else 0
        fp_rem_km = removed.geometry.iloc[fp_rem].length.sum() / 1000 if fp_rem else 0

        results["removed_tp_lines"] = len(tp_rem)
        results["removed_fp_lines"] = len(fp_rem)
        results["removed_tp_km"] = round(tp_rem_km, 2)
        results["removed_fp_km"] = round(fp_rem_km, 2)

        print(f"\n  Of removed lines:")
        print(f"    TRUE POSITIVES removed (DAMAGE):  {len(tp_rem)} lines, {tp_rem_km:.1f} km")
        print(f"    FALSE POSITIVES removed (GOOD):   {len(fp_rem)} lines, {fp_rem_km:.1f} km")
        if tp_rem_km > fp_rem_km:
            print(f"    ⚠ Phase 5 removes MORE good lines than bad lines!")

        # Source breakdown of removed
        if "source" in removed.columns:
            print(f"\n  Removed by source:")
            for src in removed["source"].unique():
                src_gdf = removed[removed["source"] == src]
                src_km = src_gdf.geometry.length.sum() / 1000
                print(f"    {src}: {len(src_gdf)} lines ({src_km:.1f} km)")

    # Check if final lines that ARE noise are original or distorted
    tp_final, fp_final = classify_lines_by_proximity(final, nav, buffer_m=BUFFER_M)
    fp_final_km = final.geometry.iloc[fp_final].length.sum() / 1000 if fp_final else 0
    results["final_fp_lines"] = len(fp_final)
    results["final_fp_km"] = round(fp_final_km, 2)
    print(f"\n  Final output noise:")
    print(f"    {len(fp_final)} false positive lines ({fp_final_km:.1f} km)")

    # Length distribution of remaining FP lines
    if fp_final:
        fp_lens = final.geometry.iloc[fp_final].length.values
        print(f"    Length distribution of noise lines:")
        print(f"      < 20m:  {(fp_lens < 20).sum()}")
        print(f"      20-50m: {((fp_lens >= 20) & (fp_lens < 50)).sum()}")
        print(f"      50-100m: {((fp_lens >= 50) & (fp_lens < 100)).sum()}")
        print(f"      > 100m: {(fp_lens >= 100).sum()}")

        # Source of noise lines
        if "source" in final.columns:
            print(f"    Noise by source:")
            fp_sources = final.iloc[fp_final]["source"].value_counts()
            for src, cnt in fp_sources.items():
                src_km = final.iloc[fp_final][final.iloc[fp_final]["source"] == src].geometry.length.sum() / 1000
                print(f"      {src}: {cnt} lines ({src_km:.1f} km)")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC D5: Line Length & Source Distribution
# ═══════════════════════════════════════════════════════════════════════
def diagnostic_d5_distributions(merged, final):
    print("\n" + "=" * 70)
    print("  D5: LINE LENGTH & SOURCE DISTRIBUTION")
    print("=" * 70)

    results = {}

    for label, gdf in [("Phase 4 Merged", merged), ("Phase 5 Final", final)]:
        if gdf is None:
            continue
        lengths = gdf.geometry.length.values
        print(f"\n  {label}:")
        print(f"    Total lines: {len(gdf)}")
        print(f"    Length stats:")
        print(f"      Min:    {lengths.min():.1f} m")
        print(f"      Median: {np.median(lengths):.1f} m")
        print(f"      Mean:   {lengths.mean():.1f} m")
        print(f"      Max:    {lengths.max():.1f} m")
        print(f"      Total:  {lengths.sum()/1000:.1f} km")

        bins = [0, 5, 10, 20, 50, 100, 200, 500, float("inf")]
        bin_labels = ["0-5m", "5-10m", "10-20m", "20-50m", "50-100m", "100-200m", "200-500m", ">500m"]
        print(f"    Length distribution:")
        for lo, hi, bl in zip(bins[:-1], bins[1:], bin_labels):
            cnt = ((lengths >= lo) & (lengths < hi)).sum()
            km = lengths[(lengths >= lo) & (lengths < hi)].sum() / 1000
            print(f"      {bl:>10}: {cnt:>6} lines ({km:.1f} km)")

        if "source" in gdf.columns:
            print(f"    Source distribution:")
            for src in gdf["source"].unique():
                src_gdf = gdf[gdf["source"] == src]
                src_km = src_gdf.geometry.length.sum() / 1000
                print(f"      {src:>12}: {len(src_gdf)} lines ({src_km:.1f} km)")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC D6: Probe-only lines in merged network
# ═══════════════════════════════════════════════════════════════════════
def diagnostic_d6_probe_in_merged(nav, merged):
    print("\n" + "=" * 70)
    print("  D6: PROBE LINES IN MERGED NETWORK — Are they helping?")
    print("=" * 70)

    results = {}
    if merged is None or nav is None:
        print("  SKIP: Missing data")
        return results

    if "source" not in merged.columns:
        print("  SKIP: No source column in merged network")
        return results

    vpd_lines = merged[merged["source"] == "VPD"]
    probe_lines = merged[merged["source"] == "Probe"]

    print(f"  VPD lines:   {len(vpd_lines)} ({vpd_lines.geometry.length.sum()/1000:.1f} km)")
    print(f"  Probe lines: {len(probe_lines)} ({probe_lines.geometry.length.sum()/1000:.1f} km)")

    # Evaluate VPD-only vs Full merged
    vpd_metrics = segmented_precision_recall(vpd_lines, nav)
    full_metrics = segmented_precision_recall(merged, nav)

    print(f"\n  VPD-only metrics:")
    print(f"    Recovery:  {vpd_metrics['recovery']:.1f}%")
    print(f"    Precision: {vpd_metrics['precision']:.1f}%")

    print(f"  Full merged metrics:")
    print(f"    Recovery:  {full_metrics['recovery']:.1f}%")
    print(f"    Precision: {full_metrics['precision']:.1f}%")

    rec_gain = full_metrics["recovery"] - vpd_metrics["recovery"]
    prec_loss = vpd_metrics["precision"] - full_metrics["precision"]
    print(f"\n  Probe contribution:")
    print(f"    Recovery gain:  +{rec_gain:.1f}%")
    print(f"    Precision loss: -{prec_loss:.1f}%")
    if prec_loss > rec_gain:
        print(f"    ⚠ Probes HURT more than they help (precision loss > recovery gain)")

    # Classify probe lines individually
    if len(probe_lines) > 0:
        tp_p, fp_p = classify_lines_by_proximity(probe_lines, nav, buffer_m=BUFFER_M)
        tp_km = probe_lines.geometry.iloc[tp_p].length.sum() / 1000 if tp_p else 0
        fp_km = probe_lines.geometry.iloc[fp_p].length.sum() / 1000 if fp_p else 0
        results["probe_tp_in_merged"] = len(tp_p)
        results["probe_fp_in_merged"] = len(fp_p)
        print(f"\n  Probe line classification in merged:")
        print(f"    True Positive:  {len(tp_p)} lines ({tp_km:.1f} km)")
        print(f"    False Positive: {len(fp_p)} lines ({fp_km:.1f} km) ← THIS IS THE NOISE")
        print(f"    Noise ratio:    {fp_km/(tp_km+fp_km)*100:.1f}%" if (tp_km+fp_km) > 0 else "")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  PIPELINE DIAGNOSTICS — Comprehensive Accuracy Analysis")
    print("=" * 70)
    t0 = time.time()

    # Load all data
    print("\nLoading data...")
    nav = load_gdf(NAV_PATH, "Nav Streets")
    vpd_raw = load_gdf(VPD_PHASE1, "VPD Phase 1")
    hpd_raw = load_gdf(HPD_PHASE1, "HPD Phase 1")
    skeleton = load_gdf(SKELETON_PHASE2, "Skeleton Phase 2")
    probe = load_gdf(PROBE_PHASE3, "Probe Phase 3")
    merged = load_gdf(MERGED_PHASE4, "Merged Phase 4")
    final = load_gdf(FINAL_PHASE5, "Final Phase 5")

    # Run all diagnostics
    d1 = diagnostic_d1_vpd_coverage(nav, vpd_raw, skeleton)
    d2 = diagnostic_d2_probe_noise(nav, probe, hpd_raw)
    d3 = diagnostic_d3_phase_degradation(nav, skeleton, probe, merged, final)
    d4 = diagnostic_d4_phase5_damage(nav, merged, final)
    d5 = diagnostic_d5_distributions(merged, final)
    d6 = diagnostic_d6_probe_in_merged(nav, merged)

    # Summary of findings
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY — Key Findings")
    print("=" * 70)

    findings = []

    if d1.get("kde_gap_pct", 0) > 10:
        findings.append(f"  1. KDE processing loses {d1['kde_gap_pct']:.1f}% of VPD coverage")
    if d1.get("vpd_raw_nav_coverage_pct", 0) < 65:
        findings.append(f"  2. VPD raw traces cover only {d1.get('vpd_raw_nav_coverage_pct',0):.1f}% of nav (need more data or lower sample)")

    if d2.get("probe_noise_pct", 0) > 20:
        findings.append(f"  3. Probe noise is {d2['probe_noise_pct']:.1f}% — too high")
    if d4.get("removed_tp_km", 0) > d4.get("removed_fp_km", 0):
        findings.append(f"  4. Phase 5 removes {d4['removed_tp_km']:.1f} km of GOOD lines vs {d4['removed_fp_km']:.1f} km noise")
    if d4.get("final_fp_km", 0) > 5:
        findings.append(f"  5. Final output has {d4['final_fp_km']:.1f} km of noise lines")

    for f in findings:
        print(f)

    if not findings:
        print("  No critical findings.")

    elapsed = time.time() - t0
    print(f"\nDiagnostics completed in {elapsed:.1f}s")

    # Write report
    report_path = os.path.join(PROJECT_ROOT, "output", "diagnostics", "diagnostic_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    # Re-run with stdout capture would be complex, so just note completion
    print(f"\nDiagnostic analysis complete. Review the numbers above to guide fixes.")


if __name__ == "__main__":
    main()
