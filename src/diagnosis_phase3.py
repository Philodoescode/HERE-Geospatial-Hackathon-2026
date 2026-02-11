"""
Phase 3 Diagnostic Suite – Why is Recall Low & Visual Output Mostly Wrong?

Runs a comprehensive battery of diagnostics on the final_network_phase3.gpkg
vs the nav_streets ground truth to pinpoint the root causes of:
  - Low recall  (50.1%)  →  GT roads not covered by the network
  - High total length (6371 km vs GT 1195 km) → massive over-generation
  - Visually incorrect output

Uses the `here_env` conda environment.
Run:  conda activate here_env && python src/diagnosis_phase3.py
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import LineString, box, MultiLineString
from shapely.ops import unary_union
import os
import sys
import warnings
import traceback

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────
DIAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "output", "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")


# ── Helpers ──────────────────────────────────────────────────────────
def _project_to_local(gdf):
    """Project geographic GeoDataFrame to local azimuthal equidistant (metres)."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.is_geographic:
        centroid = gdf.geometry.union_all().centroid
        local = (f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} "
                 f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        return gdf.to_crs(local)
    return gdf


def _safe_union(geom_series):
    """Union a GeoSeries, handling different shapely versions."""
    try:
        return geom_series.union_all()
    except AttributeError:
        return geom_series.unary_union


def _buffer_union(gdf, dist_m):
    """Buffer all geometries and dissolve into a single geometry."""
    return _safe_union(gdf.geometry.buffer(dist_m))


# ══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC TESTS
# ══════════════════════════════════════════════════════════════════════

def diag_01_length_budget(gen, gt):
    """Compare total lengths — are we generating way too much geometry?"""
    print("\n" + "="*70)
    print("  DIAG 01: Length Budget Analysis")
    print("="*70)

    vpd = gen[gen["source"] == "VPD"] if "source" in gen.columns else gen.iloc[:0]
    probe = gen[gen["source"] == "Probe"] if "source" in gen.columns else gen

    l_vpd = vpd.geometry.length.sum() / 1000.0
    l_probe = probe.geometry.length.sum() / 1000.0
    l_gen = gen.geometry.length.sum() / 1000.0
    l_gt = gt.geometry.length.sum() / 1000.0
    ratio = l_gen / l_gt if l_gt > 0 else float("inf")

    print(f"  VPD length:        {l_vpd:>10.2f} km  ({len(vpd)} segments)")
    print(f"  Probe length:      {l_probe:>10.2f} km  ({len(probe)} segments)")
    print(f"  TOTAL generated:   {l_gen:>10.2f} km")
    print(f"  Ground Truth:      {l_gt:>10.2f} km")
    print(f"  Overgeneration:    {ratio:>10.1f}x GT length")

    if ratio > 3.0:
        print("  >> CRITICAL: Generated network is >3x GT. Probe traces are being "
              "kept as raw trajectories instead of collapsed to road segments.")
    elif ratio > 1.5:
        print("  >> WARNING: Generated network ~{:.0f}% larger than GT.".format((ratio-1)*100))
    else:
        print("  OK: Length budget looks reasonable.")

    # Per-segment length stats
    if len(probe) > 0:
        lens = probe.geometry.length
        print(f"\n  Probe segment length stats:")
        print(f"    Mean:    {lens.mean():>8.1f} m")
        print(f"    Median:  {lens.median():>8.1f} m")
        print(f"    P90:     {lens.quantile(0.9):>8.1f} m")
        print(f"    Max:     {lens.max():>8.1f} m")
        if lens.mean() > 1000:
            print("  >> Probe segments average >1km — these are full trajectories, "
                  "not road segments. The clustering/collapsing step is ineffective.")

    return {
        "l_vpd_km": l_vpd, "l_probe_km": l_probe,
        "l_gen_km": l_gen, "l_gt_km": l_gt, "ratio": ratio,
    }


def diag_02_recall_by_road_class(gen, gt, buffer_m=15.0):
    """Break down recall per func_class — where are we missing roads?"""
    print("\n" + "="*70)
    print("  DIAG 02: Recall Breakdown by Road Functional Class")
    print("="*70)

    if "func_class" not in gt.columns:
        print("  Ground truth has no 'func_class' column — skipping class breakdown.")
        return {}

    # Build network buffer once (expensive — do it only once)
    print("  Building network buffer (15 m)... ", end="", flush=True)
    net_buf = _buffer_union(gen, buffer_m)
    print("done.")

    results = {}
    total_covered_km = 0.0
    total_gt_km = 0.0

    for fc in sorted(gt["func_class"].unique()):
        subset = gt[gt["func_class"] == fc]
        total_km = subset.geometry.length.sum() / 1000.0
        total_gt_km += total_km

        # Use spatial index for speed
        possible_idx = list(subset.sindex.intersection(net_buf.bounds))
        if len(possible_idx) == 0:
            covered_km = 0.0
        else:
            candidates = subset.iloc[possible_idx]
            covered = candidates.geometry.intersection(net_buf)
            covered_km = covered.length.sum() / 1000.0

        total_covered_km += covered_km
        recall = covered_km / total_km * 100 if total_km > 0 else 0
        results[fc] = {"recall": recall, "gt_km": total_km, "n_segs": len(subset)}
        print(f"  func_class {fc}: recall = {recall:5.1f}%  |  "
              f"{total_km:7.2f} km  ({len(subset):>5} segments)")

    overall = total_covered_km / total_gt_km * 100 if total_gt_km > 0 else 0
    print(f"  {'OVERALL':>12}: recall = {overall:5.1f}%  |  {total_gt_km:7.2f} km")

    # Identify the problem classes
    low_recall = {fc: v for fc, v in results.items() if v["recall"] < 60}
    if low_recall:
        worst = min(low_recall, key=lambda k: low_recall[k]["recall"])
        print(f"\n  >> Low recall on func_class(es): {list(low_recall.keys())}")
        print(f"    Worst: func_class {worst} at {low_recall[worst]['recall']:.1f}% "
              f"({low_recall[worst]['gt_km']:.1f} km)")

        # How much of total GT is in these low-recall classes?
        missing_km = sum(v["gt_km"] * (1 - v["recall"]/100) for v in low_recall.values())
        print(f"    Missing road length from low-recall classes: ~{missing_km:.1f} km "
              f"({missing_km/total_gt_km*100:.1f}% of total GT)")

    return results


def diag_03_precision_by_source(gen, gt, buffer_m=15.0):
    """Precision breakdown by source — how much probe data is off-road noise?"""
    print("\n" + "="*70)
    print("  DIAG 03: Precision Breakdown by Source")
    print("="*70)

    if "source" not in gen.columns:
        print("  No 'source' column — skipping.")
        return {}

    print("  Building GT buffer (15 m)... ", end="", flush=True)
    gt_buf = _buffer_union(gt, buffer_m)
    print("done.")

    results = {}
    for src in gen["source"].unique():
        subset = gen[gen["source"] == src]
        total_km = subset.geometry.length.sum() / 1000.0

        possible_idx = list(subset.sindex.intersection(gt_buf.bounds))
        if len(possible_idx) == 0:
            matched_km = 0.0
        else:
            candidates = subset.iloc[possible_idx]
            inside = candidates.geometry.intersection(gt_buf)
            matched_km = inside.length.sum() / 1000.0

        precision = matched_km / total_km * 100 if total_km > 0 else 0
        off_road_km = total_km - matched_km
        results[src] = {"precision": precision, "total_km": total_km,
                        "off_road_km": off_road_km, "n_segs": len(subset)}
        print(f"  {src:>8}: precision = {precision:5.1f}%  |  "
              f"{total_km:8.2f} km total, {off_road_km:8.2f} km off-road  "
              f"({len(subset)} segments)")

    if "Probe" in results and results["Probe"]["precision"] < 30:
        print("  >> CRITICAL: Probe precision <30%. Most probe data is off-road noise.")
    elif "Probe" in results and results["Probe"]["off_road_km"] > 500:
        print(f"  >> WARNING: {results['Probe']['off_road_km']:.0f} km of probe data "
              "is off-road — inflating network & tanking recall metrics.")

    return results


def diag_04_parallel_duplication(gen, gt):
    """Detect parallel duplication — multiple probe lines tracing the same road."""
    print("\n" + "="*70)
    print("  DIAG 04: Parallel Duplication Analysis")
    print("="*70)

    # Sample several GT segments and measure corridor density
    long_gt = gt[gt.geometry.length > 200].copy()
    if len(long_gt) == 0:
        print("  No GT segments > 200 m — skipping.")
        return {}

    sample_size = min(50, len(long_gt))
    sample = long_gt.sample(sample_size, random_state=42)

    factors = []
    for idx, row in sample.iterrows():
        corridor = row.geometry.buffer(25)
        # Spatial index query
        cand_idx = list(gen.sindex.intersection(corridor.bounds))
        if len(cand_idx) == 0:
            factors.append(0.0)
            continue
        cands = gen.iloc[cand_idx]
        intersecting = cands[cands.intersects(corridor)]
        if len(intersecting) == 0:
            factors.append(0.0)
            continue
        gen_len_km = intersecting.geometry.intersection(corridor).length.sum() / 1000.0
        gt_len_km = row.geometry.length / 1000.0
        factor = gen_len_km / gt_len_km if gt_len_km > 0 else 0
        factors.append(factor)

    factors = np.array(factors)
    print(f"  Sampled {sample_size} GT corridors (25 m buffer):")
    print(f"    Mean duplication factor:   {factors.mean():.2f}x")
    print(f"    Median duplication factor: {np.median(factors):.2f}x")
    print(f"    Max duplication factor:    {factors.max():.2f}x")
    print(f"    Corridors with 0 overlap:  {(factors == 0).sum()} "
          f"({(factors == 0).sum()/len(factors)*100:.0f}%)")
    print(f"    Corridors with >3x dupl:   {(factors > 3).sum()} "
          f"({(factors > 3).sum()/len(factors)*100:.0f}%)")

    if factors.mean() > 3:
        print("  >> CRITICAL: Average >3x duplication. Multiple probe trajectories "
              "are stacked on the same road corridor. DBSCAN clustering is not "
              "collapsing them.")
    elif (factors == 0).sum() / len(factors) > 0.3:
        print("  >> WARNING: >30% of GT corridors have ZERO generated overlap — "
              "this directly explains low recall.")

    # Visualize distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(factors, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(1.0, color="green", linestyle="--", linewidth=2, label="Ideal (1x)")
        ax.axvline(factors.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"Mean ({factors.mean():.1f}x)")
        ax.set_xlabel("Duplication Factor (generated / GT in corridor)")
        ax.set_ylabel("Count")
        ax.set_title("DIAG 04: Parallel Duplication per GT Corridor")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG_DIR, "diag04_duplication_histogram.png"), dpi=150)
        plt.close(fig)
        print(f"  -> Saved diag04_duplication_histogram.png")
    except Exception as e:
        print(f"  Viz failed: {e}")

    return {"mean": float(factors.mean()), "median": float(np.median(factors)),
            "zero_pct": float((factors == 0).sum() / len(factors) * 100)}


def diag_05_fragmentation(gen):
    """Segment length distribution — are segments too short or too long?"""
    print("\n" + "="*70)
    print("  DIAG 05: Fragmentation / Segment Length Analysis")
    print("="*70)

    lens = gen.geometry.length  # metres
    print(f"  Total segments: {len(gen)}")
    print(f"  Mean length:    {lens.mean():.1f} m")
    print(f"  Median length:  {lens.median():.1f} m")
    print(f"  Std dev:        {lens.std():.1f} m")
    print(f"  < 50 m:         {(lens < 50).sum()} ({(lens < 50).mean()*100:.1f}%)")
    print(f"  50-500 m:       {((lens >= 50) & (lens < 500)).sum()}")
    print(f"  500-2000 m:     {((lens >= 500) & (lens < 2000)).sum()}")
    print(f"  > 2000 m:       {(lens > 2000).sum()} ({(lens > 2000).mean()*100:.1f}%)")

    if lens.mean() > 2000:
        print("  >> CRITICAL: Average segment >2 km. These are raw trajectories, "
              "not road segments. The pipeline is NOT splitting probes into "
              "individual road links.")
    elif (lens < 50).mean() > 0.5:
        print("  >> WARNING: >50% of segments < 50 m — severe fragmentation.")
    else:
        print("  OK: Segment lengths look reasonable.")

    # Histogram
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(lens.clip(upper=5000), bins=50, color="skyblue",
                edgecolor="black", log=True)
        ax.axvline(lens.mean(), color="red", ls="--", label=f"Mean: {lens.mean():.0f} m")
        ax.axvline(lens.median(), color="green", ls="--", label=f"Median: {lens.median():.0f} m")
        ax.set_xlabel("Segment Length (m)")
        ax.set_ylabel("Count (log)")
        ax.set_title("DIAG 05: Segment Length Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG_DIR, "diag05_length_histogram.png"), dpi=150)
        plt.close(fig)
        print(f"  -> Saved diag05_length_histogram.png")
    except Exception as e:
        print(f"  Viz failed: {e}")


def diag_06_spatial_coverage_heatmap(gen, gt):
    """Grid-based spatial coverage — which cells have recall gaps?"""
    print("\n" + "="*70)
    print("  DIAG 06: Spatial Coverage Grid Analysis")
    print("="*70)

    # Build a grid over the GT extent
    bounds = gt.total_bounds  # [minx, miny, maxx, maxy]
    cell_size = 500  # metres
    n_x = int(np.ceil((bounds[2] - bounds[0]) / cell_size))
    n_y = int(np.ceil((bounds[3] - bounds[1]) / cell_size))
    print(f"  Grid: {n_x} x {n_y} cells ({cell_size} m)")

    grid_data = []
    for i in range(n_x):
        for j in range(n_y):
            x0 = bounds[0] + i * cell_size
            y0 = bounds[1] + j * cell_size
            cell = box(x0, y0, x0 + cell_size, y0 + cell_size)

            # GT in cell
            gt_idx = list(gt.sindex.intersection(cell.bounds))
            if len(gt_idx) == 0:
                continue
            gt_in = gt.iloc[gt_idx]
            gt_clipped = gt_in.geometry.intersection(cell)
            gt_len = gt_clipped.length.sum()
            if gt_len < 10:  # skip cells with negligible GT
                continue

            # Gen in cell
            gen_idx = list(gen.sindex.intersection(cell.bounds))
            if len(gen_idx) == 0:
                gen_len = 0.0
            else:
                gen_in = gen.iloc[gen_idx]
                # How much GT is covered by generated network in this cell?
                gen_buf = _safe_union(gen_in.geometry.buffer(15.0))
                gt_covered = gt_clipped.intersection(gen_buf)
                gen_len = gt_covered.length.sum()

            recall = gen_len / gt_len * 100 if gt_len > 0 else 0
            grid_data.append({
                "x": x0 + cell_size / 2,
                "y": y0 + cell_size / 2,
                "gt_len_m": gt_len,
                "recall": min(recall, 100),
            })

    df_grid = pd.DataFrame(grid_data)
    if len(df_grid) == 0:
        print("  No cells with GT data.")
        return

    print(f"  Cells analysed: {len(df_grid)}")
    print(f"  Cells with recall < 25%: {(df_grid['recall'] < 25).sum()}")
    print(f"  Cells with recall < 50%: {(df_grid['recall'] < 50).sum()}")
    print(f"  Cells with recall > 90%: {(df_grid['recall'] > 90).sum()}")

    # How much GT length is in low-recall cells?
    low = df_grid[df_grid["recall"] < 50]
    low_gt_km = low["gt_len_m"].sum() / 1000.0
    total_gt_km = df_grid["gt_len_m"].sum() / 1000.0
    print(f"  GT length in <50% recall cells: {low_gt_km:.2f} km "
          f"({low_gt_km/total_gt_km*100:.1f}% of gridded GT)")

    # Scatter heatmap
    try:
        fig, ax = plt.subplots(figsize=(12, 12))
        sc = ax.scatter(df_grid["x"], df_grid["y"],
                        c=df_grid["recall"], cmap="RdYlGn",
                        vmin=0, vmax=100, s=20, marker="s", alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Recall %")
        ax.set_title("DIAG 06: Spatial Recall Heatmap (500 m cells)")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG_DIR, "diag06_recall_heatmap.png"), dpi=150)
        plt.close(fig)
        print(f"  -> Saved diag06_recall_heatmap.png")
    except Exception as e:
        print(f"  Viz failed: {e}")

    return df_grid


def diag_07_missing_roads_visualization(gen, gt, buffer_m=15.0):
    """Visualize which GT roads are NOT covered by the generated network."""
    print("\n" + "="*70)
    print("  DIAG 07: Missing Roads Visualization")
    print("="*70)

    print("  Computing missing GT segments... ", end="", flush=True)
    net_buf = _buffer_union(gen, buffer_m)

    # For each GT segment compute coverage ratio
    gt_copy = gt.copy()
    gt_copy["gt_len"] = gt_copy.geometry.length

    # Spatial index pre-filter
    possible_idx = list(gt_copy.sindex.intersection(net_buf.bounds))
    gt_copy["covered_len"] = 0.0
    if len(possible_idx) > 0:
        candidates = gt_copy.iloc[possible_idx]
        covered = candidates.geometry.intersection(net_buf)
        gt_copy.loc[candidates.index, "covered_len"] = covered.length.values

    gt_copy["recall_pct"] = (gt_copy["covered_len"] / gt_copy["gt_len"] * 100).clip(0, 100)
    print("done.")

    missing = gt_copy[gt_copy["recall_pct"] < 30]
    partial = gt_copy[(gt_copy["recall_pct"] >= 30) & (gt_copy["recall_pct"] < 80)]
    covered = gt_copy[gt_copy["recall_pct"] >= 80]

    print(f"  Missing (<30% covered):  {len(missing):>5} segments, "
          f"{missing['gt_len'].sum()/1000:.2f} km")
    print(f"  Partial (30-80%):        {len(partial):>5} segments, "
          f"{partial['gt_len'].sum()/1000:.2f} km")
    print(f"  Covered (>80%):          {len(covered):>5} segments, "
          f"{covered['gt_len'].sum()/1000:.2f} km")

    if "func_class" in gt_copy.columns:
        print("\n  Missing roads by func_class:")
        for fc in sorted(gt_copy["func_class"].unique()):
            fc_missing = missing[missing["func_class"] == fc]
            fc_total = gt_copy[gt_copy["func_class"] == fc]
            print(f"    fc={fc}: {len(fc_missing):>5} missing / {len(fc_total):>5} total "
                  f"({fc_missing['gt_len'].sum()/1000:.1f} km missing)")

    # Visualization
    try:
        fig, ax = plt.subplots(figsize=(14, 14))
        if len(covered) > 0:
            covered.plot(ax=ax, color="green", linewidth=0.3, alpha=0.5, label="Covered (>80%)")
        if len(partial) > 0:
            partial.plot(ax=ax, color="orange", linewidth=0.5, alpha=0.7, label="Partial (30-80%)")
        if len(missing) > 0:
            missing.plot(ax=ax, color="red", linewidth=0.8, alpha=0.9, label="Missing (<30%)")

        ax.set_title(f"DIAG 07: Missing Roads\n"
                     f"Missing: {len(missing)} segs ({missing['gt_len'].sum()/1000:.1f} km)  |  "
                     f"Covered: {len(covered)} segs ({covered['gt_len'].sum()/1000:.1f} km)")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG_DIR, "diag07_missing_roads.png"), dpi=150)
        plt.close(fig)
        print(f"  -> Saved diag07_missing_roads.png")
    except Exception as e:
        print(f"  Viz failed: {e}")

    return gt_copy


def diag_08_off_road_visualization(gen, gt, buffer_m=30.0):
    """Visualize generated segments that are far from any GT road (false positives)."""
    print("\n" + "="*70)
    print("  DIAG 08: Off-Road / False Positive Analysis")
    print("="*70)

    print("  Building GT buffer (30 m)... ", end="", flush=True)
    gt_buf = _buffer_union(gt, buffer_m)
    print("done.")

    # Compute how much of each generated segment is outside the GT buffer
    gen_copy = gen.copy()
    gen_copy["gen_len"] = gen_copy.geometry.length

    possible_idx = list(gen_copy.sindex.intersection(gt_buf.bounds))
    gen_copy["inside_len"] = 0.0
    if len(possible_idx) > 0:
        candidates = gen_copy.iloc[possible_idx]
        inside = candidates.geometry.intersection(gt_buf)
        gen_copy.loc[candidates.index, "inside_len"] = inside.length.values

    gen_copy["precision_pct"] = (gen_copy["inside_len"] / gen_copy["gen_len"] * 100).clip(0, 100)

    off_road = gen_copy[gen_copy["precision_pct"] < 20]
    on_road = gen_copy[gen_copy["precision_pct"] >= 80]
    mixed = gen_copy[(gen_copy["precision_pct"] >= 20) & (gen_copy["precision_pct"] < 80)]

    print(f"  On-road (>80% inside GT buffer):   {len(on_road):>5} segs, "
          f"{on_road['gen_len'].sum()/1000:.2f} km")
    print(f"  Mixed (20-80%):                     {len(mixed):>5} segs, "
          f"{mixed['gen_len'].sum()/1000:.2f} km")
    print(f"  Off-road (<20% inside GT buffer):  {len(off_road):>5} segs, "
          f"{off_road['gen_len'].sum()/1000:.2f} km")

    total_gen_km = gen_copy["gen_len"].sum() / 1000.0
    off_road_km = off_road["gen_len"].sum() / 1000.0
    print(f"  Off-road fraction: {off_road_km/total_gen_km*100:.1f}% of generated network")

    if "source" in gen_copy.columns:
        print("\n  Off-road breakdown by source:")
        for src in gen_copy["source"].unique():
            src_off = off_road[off_road["source"] == src]
            src_all = gen_copy[gen_copy["source"] == src]
            print(f"    {src:>8}: {len(src_off):>5} off-road / {len(src_all):>5} total  "
                  f"({src_off['gen_len'].sum()/1000:.1f} km off-road)")

    # Visualization
    try:
        fig, ax = plt.subplots(figsize=(14, 14))
        gt.plot(ax=ax, color="blue", linewidth=0.3, alpha=0.3, label="Ground Truth")
        if len(on_road) > 0:
            on_road.plot(ax=ax, color="green", linewidth=0.3, alpha=0.5, label="On-road")
        if len(off_road) > 0:
            off_road.plot(ax=ax, color="red", linewidth=0.5, alpha=0.7, label="Off-road")

        ax.set_title(f"DIAG 08: Off-Road Analysis\n"
                     f"Off-road: {off_road_km:.0f} km ({off_road_km/total_gen_km*100:.1f}%)  |  "
                     f"On-road: {on_road['gen_len'].sum()/1000:.0f} km")
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG_DIR, "diag08_off_road.png"), dpi=150)
        plt.close(fig)
        print(f"  -> Saved diag08_off_road.png")
    except Exception as e:
        print(f"  Viz failed: {e}")


def diag_09_geometric_quality(gen):
    """Geometric checks — jitter, simplification potential, self-intersections."""
    print("\n" + "="*70)
    print("  DIAG 09: Geometric Quality Checks")
    print("="*70)

    # Simplification reduction
    original_len = gen.geometry.length.sum()
    simplified = gen.geometry.simplify(tolerance=3.0)
    simplified_len = simplified.length.sum()
    reduction = (original_len - simplified_len) / original_len * 100

    print(f"  Simplification (3 m tolerance):")
    print(f"    Original total length:   {original_len/1000:.2f} km")
    print(f"    Simplified total length: {simplified_len/1000:.2f} km")
    print(f"    Reduction:               {reduction:.2f}%")

    if reduction > 5:
        print("  >> Significant jitter detected. Geometry needs smoothing/simplification.")
    else:
        print("  OK: Geometry is reasonably smooth.")

    # Vertex density
    total_vertices = sum(len(g.coords) if hasattr(g, 'coords') else 0
                         for g in gen.geometry)
    total_len_km = original_len / 1000.0
    verts_per_km = total_vertices / total_len_km if total_len_km > 0 else 0
    print(f"  Total vertices: {total_vertices:,}")
    print(f"  Vertices per km: {verts_per_km:.0f}")

    # Self-intersections
    invalid = gen[~gen.geometry.is_valid]
    print(f"  Invalid geometries: {len(invalid)} / {len(gen)}")


def diag_10_network_overlay_comparison(gen, gt):
    """Side-by-side visual comparison of GT vs Generated in a sample area."""
    print("\n" + "="*70)
    print("  DIAG 10: Network Overlay Visualization (Sample Areas)")
    print("="*70)

    # Pick 3 sample areas from different parts of the GT extent
    bounds = gt.total_bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    half_w = 1500  # 3km x 3km boxes

    sample_areas = [
        ("Center", box(cx - half_w, cy - half_w, cx + half_w, cy + half_w)),
        ("NE", box(cx, cy, cx + 2*half_w, cy + 2*half_w)),
        ("SW", box(cx - 2*half_w, cy - 2*half_w, cx, cy)),
    ]

    try:
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        for ax, (name, area) in zip(axes, sample_areas):
            gt_clip = gt[gt.geometry.intersects(area)]
            gen_clip = gen[gen.geometry.intersects(area)]

            gt_clip.plot(ax=ax, color="blue", linewidth=1.5, alpha=0.7, label="GT")
            if len(gen_clip) > 0:
                gen_clip.plot(ax=ax, color="red", linewidth=0.5, alpha=0.5, label="Generated")

            ax.set_title(f"{name}\nGT: {len(gt_clip)} segs | Gen: {len(gen_clip)} segs")
            ax.set_aspect("equal")
            ax.legend(fontsize=8)

        fig.suptitle("DIAG 10: GT (blue) vs Generated (red) — Sample Areas", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(DIAG_DIR, "diag10_overlay_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  -> Saved diag10_overlay_comparison.png")
    except Exception as e:
        print(f"  Viz failed: {e}")


def diag_11_probe_trajectory_analysis(gen):
    """Analyze probe trajectory characteristics to explain why clustering fails."""
    print("\n" + "="*70)
    print("  DIAG 11: Probe Trajectory Characteristics")
    print("="*70)

    if "source" not in gen.columns:
        print("  No 'source' column — skipping.")
        return

    probes = gen[gen["source"] == "Probe"].copy()
    if len(probes) == 0:
        print("  No probe segments.")
        return

    # Length analysis
    lens = probes.geometry.length
    print(f"  Probe trajectory count: {len(probes)}")
    print(f"  Length percentiles (m):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:>2}: {lens.quantile(p/100):>10.1f} m")

    # Bounding box diagonal — proxy for trajectory spread
    def bbox_diagonal(geom):
        b = geom.bounds
        return np.sqrt((b[2] - b[0])**2 + (b[3] - b[1])**2)

    diags = probes.geometry.apply(bbox_diagonal)
    print(f"\n  BBox diagonal (spread) stats:")
    print(f"    Mean:   {diags.mean():.1f} m")
    print(f"    Median: {diags.median():.1f} m")
    print(f"    Max:    {diags.max():.1f} m")

    # Sinuosity (actual length / straight-line distance)
    def sinuosity(geom):
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            return 1.0
        start = geom.coords[0]
        end = geom.coords[-1]
        straight_dist = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        if straight_dist < 1:
            return 1.0
        return geom.length / straight_dist

    sins = probes.geometry.apply(sinuosity)
    print(f"\n  Sinuosity (length / straight-line):")
    print(f"    Mean:   {sins.mean():.2f}")
    print(f"    > 3.0 (very winding paths): {(sins > 3).sum()} "
          f"({(sins > 3).mean()*100:.1f}%)")
    print(f"    > 10.0 (extreme):           {(sins > 10).sum()}")

    if lens.mean() > 2000:
        print("\n  >> DIAGNOSIS: Probe data consists of long multi-road trajectories "
              "(avg {:.0f} m), NOT individual road segments.".format(lens.mean()))
        print("    Root cause: The gap-filling step keeps full probe trajectories. "
              "It should split them at intersections or along GT road links first.")
    if sins.mean() > 2.0:
        print("  >> Many probes are winding/circuitous — likely full trip traces, "
              "not straight road segments.")


# ══════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════

def write_summary_report(results, filepath):
    """Write a text summary report to disk."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  PHASE 3 DIAGNOSTIC SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("ROOT CAUSE ANALYSIS\n")
        f.write("-" * 40 + "\n")

        issues = []
        if results.get("length_budget", {}).get("ratio", 0) > 3:
            issues.append(
                "1. MASSIVE OVERGENERATION: The network is {:.1f}x the GT length. "
                "Probe trajectories ({:.0f} km) are being kept as full multi-road "
                "traces instead of being collapsed to individual road segments. "
                "This alone explains both low recall (probe data dilutes the signal) "
                "and visually incorrect output.".format(
                    results["length_budget"]["ratio"],
                    results["length_budget"]["l_probe_km"]))

        if results.get("recall_by_class"):
            low = {k: v for k, v in results["recall_by_class"].items() if v["recall"] < 60}
            if low:
                issues.append(
                    "2. LOW RECALL ON MINOR ROADS: func_class(es) {} have < 60% recall. "
                    "These are mostly local/residential roads (func_class 4-5) where "
                    "probe coverage exists but the gap-filling overlap filter (>80% "
                    "exclusion) is too aggressive.".format(list(low.keys())))

        if results.get("duplication", {}).get("mean", 0) > 3:
            issues.append(
                "3. PARALLEL DUPLICATION: Average {:.1f}x duplication factor. Multiple "
                "probe trajectories follow the same road but are not merged. DBSCAN "
                "eps=30m with hemispherical heading bins is too simplistic for long, "
                "overlapping trajectories.".format(results["duplication"]["mean"]))

        if results.get("duplication", {}).get("zero_pct", 0) > 30:
            issues.append(
                "4. COVERAGE GAPS: {:.0f}% of sampled GT corridors have ZERO generated "
                "overlap. The probe data either doesn't cover these roads or the "
                "filtering step is incorrectly discarding relevant probes.".format(
                    results["duplication"]["zero_pct"]))

        if not issues:
            issues.append("No critical issues detected (unexpected given metrics).")

        for issue in issues:
            f.write(f"\n{issue}\n")

        f.write("\n\nRECOMMENDED FIXES\n")
        f.write("-" * 40 + "\n")
        f.write("""
1. SPLIT PROBE TRAJECTORIES: Before gap-filling, split each probe trace at
   intersections with existing skeleton roads. This converts km-long trajectories
   into individual road-link-sized segments.

2. REDUCE EXCLUSION THRESHOLD: Change the overlap filter from <20% to <50%
   to keep probe segments that partially overlap the skeleton (these often
   represent parallel or connecting roads).

3. MAP-MATCH PROBES TO GT: Use proximity-based map-matching to associate probe
   segments with GT road links, enabling per-link coverage analysis.

4. IMPROVE CLUSTERING: Replace hemispherical heading bins with +/-30 degree
   angular tolerance. Use projection onto a reference line instead of midpoint
   DBSCAN.

5. POST-FILTER BY LENGTH: Remove generated segments shorter than 20m (noise)
   and longer than 2km (uncollapsed trajectories).
""")

    print(f"\n  Summary report saved to: {filepath}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def run_diagnostics():
    gen_path = os.path.join(DATA_DIR, "final_network_phase3.gpkg")
    gt_path = os.path.join(DATA_DIR, "Kosovo_nav_streets", "nav_kosovo.gpkg")

    for p in [gen_path, gt_path]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            return

    # -- Load --
    print("Loading data...")
    gt_raw = gpd.read_file(gt_path)
    gen_raw = gpd.read_file(gen_path)

    # -- Project to local metric CRS --
    print("Projecting to local metric CRS...")
    gt = _project_to_local(gt_raw)
    gen = gen_raw.to_crs(gt.crs)
    print(f"  GT: {len(gt)} segments  |  Gen: {len(gen)} segments")

    # -- Run all diagnostics --
    results = {}

    results["length_budget"] = diag_01_length_budget(gen, gt)
    results["recall_by_class"] = diag_02_recall_by_road_class(gen, gt)
    results["precision_by_source"] = diag_03_precision_by_source(gen, gt)
    results["duplication"] = diag_04_parallel_duplication(gen, gt)
    diag_05_fragmentation(gen)
    diag_06_spatial_coverage_heatmap(gen, gt)
    diag_07_missing_roads_visualization(gen, gt)
    diag_08_off_road_visualization(gen, gt)
    diag_09_geometric_quality(gen)
    diag_10_network_overlay_comparison(gen, gt)
    diag_11_probe_trajectory_analysis(gen)

    # -- Summary --
    report_path = os.path.join(DIAG_DIR, "diagnostic_report.txt")
    write_summary_report(results, report_path)

    print("\n" + "="*70)
    print("  ALL DIAGNOSTICS COMPLETE")
    print("="*70)
    print(f"  Figures saved in: {DIAG_DIR}")
    print(f"  Report saved at:  {report_path}")


if __name__ == "__main__":
    try:
        run_diagnostics()
    except Exception as e:
        traceback.print_exc()
        print(f"\nFAILED: {e}")
