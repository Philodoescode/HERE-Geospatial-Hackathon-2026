# Roadster Adaptation Notes (HERE Hackathon Problem 1)

## 1) Algorithm Stages Implemented

1. **Subtrajectory clustering**
- Each cleaned trace segment is split into overlapping fixed-length windows (`subtraj_window_m`, `subtraj_step_m`).
- Windows are compared using a **discrete Fréchet distance** on uniformly resampled points.
- Cluster compatibility additionally enforces:
  - heading similarity (direction-aware),
  - altitude similarity (bridge/tunnel separation),
  - source-aware weighting (VPD prioritized over HPD/Probe),
  - construction and temporal weighting.

2. **Representative construction + refinement**
- For each cluster, member windows are resampled to a common complexity.
- Representative vertices are placed via weighted medians ("move to the middle").
- Turn vertices are detected from angular deflection and preserved through smoothing.

3. **Vertex inference (intersections/endpoints)**
- Candidate vertices come from representative endpoints, turn points, and representative crossings.
- Nearby candidates are snapped/clustered with altitude-aware constraints.
- Final node locations are refined with heading-line intersections (regression-style geometric refinement).

4. **Edge construction**
- Representatives are split between inferred nodes to form graph edges.
- Overlapping node pairs are merged with support aggregation.
- Tiny dangling artifacts are pruned unless strongly supported.

## 2) Essential vs Optional Roadster Components

**Essential (implemented):**
- subtrajectory-level clustering (not whole-trace clustering),
- geometry-based similarity using Fréchet-style distance approximation,
- representative refinement to improve centerline geometry,
- explicit node/edge graph construction,
- topology cleanup and support-based filtering.

**Optional / approximated for hackathon practicality:**
- exact `(k, l, epsilon)` maximal stable cluster search over epsilon lifespan,
- full free-space-diagram sweeping implementation from the paper,
- full multi-stage bundle scoring from the Java research prototype.

## 3) HERE-Specific Adaptations (Quality-Upgrading, not downgrading)

1. **VPD Fused True priority**
- `vpd_base_weight` is substantially higher than `hpd_base_weight`.
- VPD path/sensor quality boosts cluster support.

2. **Direction-aware clustering**
- Opposite-direction windows are not merged by default.
- This helps preserve divided roads and one-way structure.

3. **Altitude-aware separation**
- Cluster and node snapping enforce altitude consistency.
- `altitude_band_m` metadata is emitted for debugging stacked roads.

4. **Construction + temporal weighting**
- Construction-heavy traces are downweighted.
- Optional day/hour slicing is supported for time-consistent maps.

5. **Intersection cues**
- Crosswalk/signal attributes increase turn-node confidence.

## 4) Preprocessing Settings Used

**VPD (high quality):**
- resample: `4 m`
- simplify: `1.5 m`
- gap split: `60 m`
- turn split enabled (sharper preservation)

**HPD/Probe (supplemental):**
- resample: `8 m`
- simplify: `3 m`
- gap split: `90 m`
- time-gap split enabled when point times exist

## 5) Core Thresholds (initial defaults)

- `subtraj_window_m=65`, `subtraj_step_m=18`
- `cluster_center_radius_m=35`
- `cluster_frechet_eps_m=16`
- `cluster_heading_tolerance_deg=30`
- `cluster_altitude_eps_m=5`
- `cluster_min_members=3`, `cluster_min_weighted_support=5.5`
- `vertex_snap_m=18`, `edge_node_snap_m=20`
- `min_edge_length_m=18`, `min_edge_support=4.5`

These are tuned for Kosovo-scale urban data and are expected to be tuned per city/tile.
