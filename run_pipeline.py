"""
HSGA Pipeline Runner — Deep-Flow Hybrid Pipeline (4 phases)

Usage:
    python run_pipeline.py [--sample N] [--skip-phase1]

Phases:
  1. Data Ingestion (VPD + HPD + Nav Streets)
  2. Kharita Centerline Generation (heading-aware clustering + roundabout detection)
  3. Geometry Cleanup & Topology (Hausdorff parallel merge, Z-level resolution, smoothing, stub pruning)
  4. Final Optimization (interchange cleanup, quality selection, export)
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

from src.evaluation.metrics import print_quick_metrics


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
    """Accurate evaluation using segmented precision/recall."""
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
    # Use segmented P/R: splits lines into ~20m chunks for accurate coverage
    from src.evaluation.metrics import segmented_precision_recall
    result = segmented_precision_recall(generated, nav_gdf, buffer_m=buffer_m)
    elapsed = time.time() - t0
    print_quick_metrics(result, label=phase_name)
    print(f"  (evaluated in {elapsed:.1f}s)")
    return result


def main():
    parser = argparse.ArgumentParser(description="HSGA Deep-Flow Pipeline Runner")
    parser.add_argument("--sample", type=int, default=20000,
                        help="VPD sample size (default: 20000)")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 if outputs exist")
                        
    # Phase 2 Parameters (AGGRESSIVE tuning for max recovery)
    parser.add_argument("--cluster-radius-m", type=float, default=15.0)  # Was 10→12→15 for better clustering
    parser.add_argument("--heading-tolerance-deg", type=float, default=65.0)  # Was 55→60→65
    parser.add_argument("--heading-distance-weight-m", type=float, default=0.18)
    parser.add_argument("--min-edge-support", type=float, default=0.3)  # Was 1.0→0.5→0.3 KEY for recovery
    parser.add_argument("--reverse-edge-ratio", type=float, default=0.2)
    parser.add_argument("--sample-spacing-m", type=float, default=5.0)  # Was 8→6→5 for finer detail
    parser.add_argument("--max-points-per-trace", type=int, default=150)  # Was 120→150
    parser.add_argument("--candidate-selection-threshold", type=float, default=0.10)  # Was 0.25→0.18→0.10
    parser.add_argument("--candidate-density-scale", type=float, default=0.08)  # Was 0.15→0.12→0.08
    parser.add_argument("--enable-dynamic-weighting", action="store_true", default=True)
    parser.add_argument("--dyn-lambda-vpd", type=float, default=1.6)
    parser.add_argument("--dyn-road-likeness-beta", type=float, default=6.0)
    parser.add_argument("--dyn-road-likeness-tau", type=float, default=0.45)
    
    args = parser.parse_args()

    print("=" * 60)
    print("  HSGA DEEP-FLOW PIPELINE — Full Execution with Metrics")
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

    m1 = None
    print("\n  Phase 1 metrics: SKIPPED (raw traces, not centerlines yet)")
    gc.collect()

    # ─────────────────────────────────────────────────────────────
    #  PHASE 2: Kharita Centerline Generation
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    skeleton_output = os.path.join(PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg")

    from src.pipeline_phase2 import KDESkeletonizer, KharitaConfig  # Now uses Kharita algorithm
    
    # Configure Phase 2
    p2_config = KharitaConfig(
        cluster_radius_m=args.cluster_radius_m,
        heading_tolerance_deg=args.heading_tolerance_deg,
        heading_distance_weight_m=args.heading_distance_weight_m,
        min_edge_support=args.min_edge_support,
        reverse_edge_ratio=args.reverse_edge_ratio,
        sample_spacing_m=args.sample_spacing_m,
        max_points_per_trace=args.max_points_per_trace,
        candidate_selection_threshold=args.candidate_selection_threshold,
        candidate_density_scale=args.candidate_density_scale,
        enable_dynamic_weighting=args.enable_dynamic_weighting,
        dyn_lambda_vpd=args.dyn_lambda_vpd,
        dyn_road_likeness_beta=args.dyn_road_likeness_beta,
        dyn_road_likeness_tau=args.dyn_road_likeness_tau,
    )

    skeletonizer = KDESkeletonizer(vpd_output, skeleton_output, config=p2_config)
    skeletonizer.run()
    del skeletonizer; gc.collect()

    m2 = evaluate_phase("Phase 2: Kharita Centerlines", skeleton_output, nav_gdf)

    # ─────────────────────────────────────────────────────────────
    #  PHASE 3: VPD Geometry Refinement
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    refined_output = os.path.join(PROJECT_ROOT, "data", "interim_refined_skeleton_phase3.gpkg")

    from src.pipeline_phase3 import VPDGeometryRefiner  # Uses probe signal for refinement

    refiner = VPDGeometryRefiner(
        skeleton_path=skeleton_output,
        hpd_path=hpd_output,
        output_dir=os.path.join(PROJECT_ROOT, "data"),
    )
    refiner.run()
    del refiner; gc.collect()

    m3 = evaluate_phase("Phase 3: VPD Geometry Refinement", refined_output, nav_gdf)

    # ─────────────────────────────────────────────────────────────
    #  PHASE 4: Final Optimization (formerly Phase 5)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    final_output = os.path.join(PROJECT_ROOT, "data", "final_centerline_output_4326.gpkg")

    from src.pipeline_phase4 import GeometryOptimizer

    # Phase 4 reads directly from Phase 3 (consolidated cleanup)
    optimizer = GeometryOptimizer(
        refined_output,
        output_dir=os.path.join(PROJECT_ROOT, "data"),
    )
    optimizer.run()
    del optimizer; gc.collect()

    m4 = evaluate_phase("Phase 4: Final Output", final_output, nav_gdf)

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
        ("Phase 2: Kharita Centerlines", m2),
        ("Phase 3: Geometry Cleanup", m3),
        ("Phase 4: Final Output", m4),
    ]:
        if m is not None:
            print(f"  {label:<35} {m['nav_recovery_pct']:>10.1f} {m['nav_precision_pct']:>10.1f} {m['num_lines']:>8}")
        else:
            print(f"  {label:<35} {'N/A':>10} {'N/A':>10} {'N/A':>8}")

    print("=" * 60)
    print("  Done!")


if __name__ == "__main__":
    main()
