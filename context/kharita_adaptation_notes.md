# Kharita Adaptation Notes (HERE Hackathon Problem 1)

This implementation keeps the core Kharita idea and adapts it to the HERE VPD/HPD data quality:

1. **Kept from Kharita**
- Heading-aware clustering of trajectory observations into road nodes.
- Directed co-occurrence graph from consecutive trajectory samples.
- Graph pruning to remove weak and redundant edges.

2. **Adapted for HERE VPD/HPD**
- Input is **WKT LINESTRING in EPSG:4326** (not point-only CSV).
- Uses a local **metric CRS (UTM)** before clustering and smoothing to make meter-based thresholds valid.
- Uses VPD rich attributes during edge aggregation: `constructionpercent`, `altitudes`, `crosswalktypes`, `trafficsignalcount`, plus `day/hour`.
- Preserves travel direction from trace order and outputs `dir_travel` (`B`, `T`, `F`).
- Uses quality-aware weights (`pathqualityscore`, `sensorqualityscore`) so better VPD traces contribute more.

3. **Quality upgrades over legacy Kharita code**
- Incremental clustering is vector-safe and scalable for larger HERE traces.
- Centerlines are stitched into continuous chains and smoothed with turn-preserving refinement.
- Problem 1 heuristic candidate selection scores each centerline by support, density, length, connectivity, and geometric plausibility.
- Selection metadata is emitted per centerline: `is_selected`, `selection_score`, and `selection_reason`.
- Added ground-truth validation against Kosovo navstreets with quantitative metrics.
