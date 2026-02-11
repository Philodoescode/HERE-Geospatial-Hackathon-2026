"""
HSGA Pipeline Runner — Runs all 4 phases with metrics after each.

Usage:
    python run_pipeline.py [--sample N] [--skip-phase1]

This script:
  1. Phase 1: Data Ingestion (VPD + HPD + Nav Streets)
  2. Phase 2: VPD Skeleton Construction (Fréchet bundling)
  3. Phase 3: Probe-based Gap Filling
  4. Phase 4: Topology Refinement & Smoothing
  5. After each phase: evaluate against Nav Streets ground truth
"""

import os
import sys
import time
import gc
import argparse
import geopandas as gpd

# Project setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import quick_precision_recall, print_quick_metrics


def load_nav_streets():
    """Load Nav Streets ground truth."""
    nav_path = os.path.join(PROJECT_ROOT, "data", "Kosovo_nav_streets", "nav_kosovo.gpkg")
    if not os.path.exists(nav_path):
        print(f"  WARNING: Nav Streets not found at {nav_path}")
        return None
    nav = gpd.read_file(nav_path)
    if nav.crs is None:
        nav = nav.set_crs("EPSG:4326")
    elif str(nav.crs) != "EPSG:4326":
        nav = nav.to_crs("EPSG:4326")
    return nav


def evaluate_phase(phase_name, output_path, nav_gdf, buffer_m=15.0):
    """Fast evaluation: precision & recall via spatial index (no heavy intersection)."""
    print(f"\n{'─' * 60}")
    print(f"  METRICS: {phase_name}")
    print(f"{'─' * 60}")

    if not os.path.exists(output_path):
        print(f"  Output file not found: {output_path}")
        return None

    if nav_gdf is None:
        print("  No ground truth available for evaluation.")
        return None

    generated = gpd.read_file(output_path)
    if generated.crs is None:
        generated = generated.set_crs("EPSG:4326")
    elif str(generated.crs) != "EPSG:4326":
        generated = generated.to_crs("EPSG:4326")

    t0 = time.time()
    result = quick_precision_recall(generated, nav_gdf, buffer_m=buffer_m)
    elapsed = time.time() - t0
    print_quick_metrics(result, label=phase_name)
    print(f"  (evaluated in {elapsed:.1f}s)")
    return result


def main():
    parser = argparse.ArgumentParser(description="HSGA Pipeline Runner")
    parser.add_argument("--sample", type=int, default=5000,
                        help="VPD sample size (default: 5000)")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 if outputs exist")
    args = parser.parse_args()

    print("=" * 60)
    print("  HSGA PIPELINE — Full Execution with Metrics")
    print("=" * 60)
    print(f"  VPD sample size: {args.sample}")
    print(f"  Project root: {PROJECT_ROOT}")
    print()

    overall_start = time.time()

    # Load ground truth once
    print("Loading Nav Streets ground truth...")
    nav_gdf = load_nav_streets()
    if nav_gdf is not None:
        print(f"  Nav Streets: {len(nav_gdf)} road links")

    # ─────────────────────────────────────────────────────────────
    #  PHASE 1: Data Ingestion
    # ─────────────────────────────────────────────────────────────
    vpd_output = os.path.join(PROJECT_ROOT, "data", "interim_sample_phase1.gpkg")
    hpd_output = os.path.join(PROJECT_ROOT, "data", "interim_hpd_phase1.gpkg")

    if args.skip_phase1 and os.path.exists(vpd_output) and os.path.exists(hpd_output):
        print("\n  Phase 1: SKIPPED (outputs exist, --skip-phase1 used)")
    else:
        print("\n" + "=" * 60)
        from src.pipeline_phase1 import DataIngestionPipeline

        phase1 = DataIngestionPipeline(vpd_sample_size=args.sample)
        phase1.run()

    # Skip Phase 1 evaluation — raw VPD traces are not centerlines,
    # so precision/recall is not meaningful and it's very slow (5000 traces).
    m1 = None
    print("\n  Phase 1 metrics: SKIPPED (raw traces, not centerlines yet)")
    gc.collect()  # Free Phase 1 memory before Phase 2

    # ─────────────────────────────────────────────────────────────
    #  PHASE 2: VPD Skeleton Construction
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    skeleton_output = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")

    from src.pipeline_phase2 import Skeletonizer

    skeletonizer = Skeletonizer(vpd_output, skeleton_output)
    skeletonizer.run()
    del skeletonizer; gc.collect()  # Free Phase 2 memory

    m2 = evaluate_phase("Phase 2: VPD Skeleton", skeleton_output, nav_gdf)

    # ─────────────────────────────────────────────────────────────
    #  PHASE 3: Probe-based Gap Filling
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    network_output = os.path.join(PROJECT_ROOT, "data", "final_network_phase3.gpkg")

    from src.pipeline_phase3 import GapFiller

    filler = GapFiller(
        skeleton_path=skeleton_output,
        hpd_path=hpd_output,
        output_dir=os.path.join(PROJECT_ROOT, "data"),
    )
    filler.run()

    m3 = evaluate_phase("Phase 3: Skeleton + Probes", network_output, nav_gdf)
    del filler; gc.collect()  # Free Phase 3 memory

    # ─────────────────────────────────────────────────────────────
    #  PHASE 4: Topology Refinement
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    final_output = os.path.join(PROJECT_ROOT, "data", "final_centerline_output_4326.gpkg")

    from src.pipeline_phase4 import TopologyRefiner

    refiner = TopologyRefiner(
        network_output,
        output_dir=os.path.join(PROJECT_ROOT, "data"),
    )
    refiner.run()

    m4 = evaluate_phase("Phase 4: Final Refined", final_output, nav_gdf)
    del refiner; gc.collect()  # Free Phase 4 memory

    # ─────────────────────────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────────────────────────
    total_time = time.time() - overall_start

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()
    print(f"  {'Phase':<35} {'Recovery%':>10} {'Precision%':>10} {'Lines':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")

    for label, m in [
        ("Phase 1: Raw VPD", m1),
        ("Phase 2: Skeleton", m2),
        ("Phase 3: + Probes", m3),
        ("Phase 4: Refined", m4),
    ]:
        if m is not None:
            print(f"  {label:<35} {m['nav_recovery_pct']:>10.1f} {m['nav_precision_pct']:>10.1f} {m['num_lines']:>8}")
        else:
            print(f"  {label:<35} {'N/A':>10} {'N/A':>10} {'N/A':>8}")

    print("=" * 60)
    print("  Done!")


if __name__ == "__main__":
    main()
