# Pipeline Diagnostics Report

## Executive Summary

**Root cause of disconnected lines: `merge_parallel_segments()` in Phase 3 was removing 52% of segments with overly aggressive settings.**

After fixing the parallel merge parameters:
- Components: 497 → 96 (80% fewer disconnected groups)
- Largest connected component: 122 → 5,416 segments
- Recovery: 47.1% → 49.4%
- Precision: 96.8% → 97.2%

## Detailed Findings

### Phase 3 Step-by-Step Loss Analysis

| Step | Segments | Lost | Issue |
|------|----------|------|-------|
| Load | 8,540 | - | - |
| Detect intersections | 8,540 | 0 | ✅ Good |
| Remove spurs | 8,515 | 25 | ✅ Good |
| **Merge parallels** | 7,031 | 1,484 | ⚠️ Fixed (was 4,426) |
| Z-level | 7,031 | 0 | ✅ Good |
| Sharp angles | 6,958 | 73 | ✅ Good |
| Smoothing | 6,958 | 0 | ✅ Good |
| Snapping | 6,950 | 8 | ✅ Good |
| Stitching | 6,717 | -233 | ✅ Good (merge) |
| Bridging | 6,925 | +208 | ✅ Good (create) |
| Jitter spikes | 6,925 | 0 | ✅ Good |
| Loop removal | 6,741 | 184 | ✅ Good |
| Stub pruning | 6,703 | 38 | ✅ Good |
| Pass 2 stitching | 6,693 | +26 | ✅ Good |
| Clean geometry | 6,656 | 37 | ✅ Good |

### Key Fix Applied

**File:** `src/pipeline_phase3.py` - `RefinementConfig`

```python
# BEFORE (overly aggressive - removing 52% of segments):
parallel_hausdorff_m: float = 12.0
parallel_heading_tol_deg: float = 30.0
parallel_min_length_ratio: float = 0.20
parallel_min_overlap: float = 0.75

# AFTER (conservative - removing only true duplicates):
parallel_hausdorff_m: float = 5.0       # Only merge truly parallel lines within 5m
parallel_heading_tol_deg: float = 20.0  # Stricter heading match
parallel_min_length_ratio: float = 0.50 # Don't remove short segments so easily
parallel_min_overlap: float = 0.90      # Require 90% overlap to merge
```

---

## Current Metrics

| Phase | Recovery% | Precision% | Segments | Components |
|-------|-----------|------------|----------|------------|
| Phase 2 | 50.2 | 96.8 | 8,540 | 63 |
| Phase 3 | 49.4 | 97.2 | 6,656 | 82 |
| Phase 4 | 49.4 | 97.2 | 6,558 | 96 |

**Recovery is now capped by Phase 2 (50.2%), not Phase 3.**

---

## Ground Truth Coverage Analysis

- Ground truth: **1,195 km** of roads
- Phase 2 generates: 447 km (37%)
- Phase 4 outputs: 366 km (31%)
- Sample coverage: 41% (27,010 samples are >100m from any generated segment)

**Root cause of 50% cap:** The VPD probe data simply doesn't cover all roads. 58.9% of ground truth road samples have no probe data within 15m.

---

## Tunable Parameters for Further Optimization

### Phase 2 (pipeline_phase2.py - KharitaConfig)

| Parameter | Current | Try | Effect |
|-----------|---------|-----|--------|
| `min_edge_support` | 0.5 | 0.3 | More roads, may lower precision |
| `candidate_selection_threshold` | 0.18 | 0.10 | More centerlines extracted |
| `candidate_dangling_max_length_m` | 45.0 | 80.0 | Keep longer dead-end roads |
| `candidate_dangling_min_weighted_support` | 3.0 | 1.5 | Keep lower-traffic roads |
| `min_centerline_length_m` | 6.0 | 3.0 | Keep shorter road segments |
| `cluster_radius_m` | 12.0 | 15.0 | Better endpoint clustering |

### Phase 3 (pipeline_phase3.py - RefinementConfig)

| Parameter | Current | Try | Effect |
|-----------|---------|-----|--------|
| `spur_max_length_m` | 15.0 | 8.0 | Remove fewer valid spurs |
| `stub_threshold_m` | 4.0 | 3.0 | More conservative stub removal |
| `stub_max_iterations` | 2 | 1 | Fewer pruning passes |
| `endpoint_snap_radius_m` | 8.0 | 12.0 | Connect more near-miss endpoints |
| `stitch_snap_radius_m` | 15.0 | 25.0 | Bridge larger gaps |
| `stitch_max_angle_deg` | 70.0 | 90.0 | Allow wider angle connections |

### Dead-End Gap Distribution (Current)

| Gap Distance | Count | Action Required |
|--------------|-------|-----------------|
| < 5m | 6 | Should be stitched (need snap radius) |
| 5-10m | 61 | Current snap covers |
| 10-15m | 124 | Need larger stitch radius |
| 15-25m | 137 | Need 25m stitch radius |
| 25-50m | 214 | Consider bridge or leave |
| > 50m | 209 | Leave as separate roads |

**Recommendation:** Increase `stitch_snap_radius_m` to 25m to cover most bridgeable gaps.

---

## Next Steps to Increase Recovery

1. **Increase VPD sample size** (e.g., 20000 instead of 10000) - more probe data = more roads
2. **Tune Phase 2 parameters** as listed above
3. **Increase stitch radius** to 25m to bridge more gaps

**Warning:** Each change may impact precision. Test incrementally and monitor both metrics.
