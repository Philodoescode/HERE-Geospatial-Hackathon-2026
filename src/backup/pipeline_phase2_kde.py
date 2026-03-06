"""
Phase 2: VPD Skeleton via Direction-Aware KDE + Roundabout Detection

Direction-aware rasterization into forward/backward grids to produce
separate centerlines for each direction on divided roads (dual
carriageways) and single centerlines on undivided roads.

Algorithm:
  1. Densify VPD traces at 1m spacing
  2. Direction-aware rasterization into 2 grids:
       - Forward grid: segments heading 0-180 deg
       - Backward grid: segments heading 180-360 deg
  3. Per-grid Gaussian blur, dual-threshold, morphological cleanup
  4. Per-direction skeletonization (dual lines for divided roads)
  5. Vectorize skeleton pixels to LineStrings via graph tracing
  6. Roundabout detection from VPD trace curvature patterns
  7. Filter short segments
"""

import gc
import os
import sys
import time
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.ndimage import (
    gaussian_filter, binary_closing, binary_opening, binary_dilation,
)
from scipy.spatial import cKDTree
from shapely import get_coordinates, segmentize as shp_segmentize
from shapely.geometry import LineString
from skimage.morphology import skeletonize, disk

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class KDESkeletonizer:
    """
    Phase 2: Build VPD skeleton using direction-aware KDE rasterization.

    Produces TRUE centerlines per direction of travel:
    - Divided roads get 2 separate centerlines (one per direction)
    - Undivided roads get 1 centerline (merged in Phase 5)
    - Roundabouts detected via trace curvature analysis
    """

    def __init__(
        self,
        input_path,
        output_path,
        pixel_size=2.0,           # grid resolution in metres
        blur_sigma=2.5,           # reduced blur to preserve single-trace roads
        threshold_frac=0.002,     # lower density threshold to capture func_class 5
        recovery_factor=0.5,      # recovery threshold = threshold_frac * this (was 0.3)
        min_segment_length=3.0,   # lower min to keep short residential segments
        grid_buffer=30.0,         # buffer around data bounds
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.pixel_size = pixel_size
        self.blur_sigma = blur_sigma
        self.threshold_frac = threshold_frac
        self.recovery_factor = recovery_factor
        self.min_segment_length = min_segment_length
        self.grid_buffer = grid_buffer

        self.gdf = None
        self.skeleton_gdf = None
        self.crs_projected = None

    # ------------------------------------------------------------------
    #  DATA LOADING
    # ------------------------------------------------------------------

    def load_and_project(self):
        print(f"  Loading VPD from {self.input_path}...")
        self.gdf = gpd.read_file(self.input_path)

        # Dynamic local projection (azimuthal equidistant)
        bounds = self.gdf.total_bounds
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        self.crs_projected = (
            f"+proj=aeqd +lat_0={cy} +lon_0={cx} "
            f"+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )
        self.gdf = self.gdf.to_crs(self.crs_projected)

        # Filter valid LineStrings
        self.gdf = self.gdf[
            self.gdf.geometry.notna()
            & ~self.gdf.geometry.is_empty
            & (self.gdf.geom_type == "LineString")
        ].reset_index(drop=True)

        print(f"  Loaded {len(self.gdf)} VPD traces.")
        
        # Compute per-trace quality weights (Dynamic Weighting algorithm)
        self._compute_trace_weights()

    def _compute_trace_weights(self):
        """
        Compute quality weight for each VPD trace.
        
        From Dynamic Weighting algorithm:
        - Longer traces are more reliable (length factor)
        - Heading consistency indicates road-likeness
        - Point density suggests GPS quality
        
        This weight is applied during rasterization so high-quality
        traces contribute more to the road detection.
        """
        print("  Computing per-trace quality weights (dynamic weighting)...")
        
        weights = []
        for idx, row in self.gdf.iterrows():
            geom = row.geometry
            coords = np.array(geom.coords)
            n_pts = len(coords)
            length_m = geom.length
            
            # Length factor: longer traces are more reliable (sigmoid curve)
            # Score 0.3 for 10m, 0.7 for 100m, 0.95 for 500m
            length_factor = 1.0 / (1.0 + np.exp(-(length_m - 50) / 80))
            length_factor = 0.3 + 0.7 * length_factor
            
            # Point density factor: more points per meter = better GPS
            if length_m > 0:
                pts_per_m = n_pts / length_m
                # Typical GPS: 1 pt per 5-10m is good, >0.5/m is excellent
                density_factor = min(1.0, pts_per_m * 5.0)
            else:
                density_factor = 0.5
            
            # Heading consistency factor (from Kharita algorithm)
            if n_pts >= 3:
                dx = coords[1:, 0] - coords[:-1, 0]
                dy = coords[1:, 1] - coords[:-1, 1]
                seg_headings = np.arctan2(dy, dx)
                
                # Compute circular variance of headings
                sin_sum = np.sum(np.sin(seg_headings))
                cos_sum = np.sum(np.cos(seg_headings))
                r = np.sqrt(sin_sum**2 + cos_sum**2) / len(seg_headings)
                heading_consistency = float(np.clip(r, 0.0, 1.0))
            else:
                heading_consistency = 0.5
            
            # Combined weight (multiplicative)
            weight = length_factor * (0.4 + 0.3 * density_factor + 0.3 * heading_consistency)
            weight = float(np.clip(weight, 0.1, 2.0))  # clamp to [0.1, 2.0]
            weights.append(weight)
        
        self.gdf["trace_weight"] = weights
        
        # Log statistics
        w_arr = np.array(weights)
        print(f"    Trace weights: min={w_arr.min():.2f}, max={w_arr.max():.2f}, "
              f"mean={w_arr.mean():.2f}, std={w_arr.std():.2f}")

    # ------------------------------------------------------------------
    #  DIRECTION-AWARE RASTERIZATION
    # ------------------------------------------------------------------

    def rasterize_traces(self):
        """
        Rasterize VPD traces into TWO direction grids.

        Grid 'fwd': segments heading 0-180 deg (northward/eastward half)
        Grid 'bwd': segments heading 180-360 deg (southward/westward half)

        This separates opposing lanes on divided roads while keeping all
        data for undivided roads (both grids contribute at same position).
        """
        print("  Rasterizing VPD traces (direction-aware, dual grids)...")
        t0 = time.time()

        # Grid extent
        minx, miny, maxx, maxy = self.gdf.total_bounds
        buf = self.grid_buffer
        minx -= buf; miny -= buf; maxx += buf; maxy += buf

        width = int(np.ceil((maxx - minx) / self.pixel_size))
        height = int(np.ceil((maxy - miny) / self.pixel_size))
        print(f"    Grid: {width} x {height} pixels ({self.pixel_size}m resolution)")

        # Memory check (2 grids, float32)
        mem_mb = width * height * 4 * 2 / 1e6
        print(f"    Grid memory: {mem_mb:.1f} MB (2 direction grids)")
        if mem_mb > 2000:
            print("    WARNING: Grid too large, increasing pixel size to 3m")
            self.pixel_size = 3.0
            width = int(np.ceil((maxx - minx) / self.pixel_size))
            height = int(np.ceil((maxy - miny) / self.pixel_size))

        grid_fwd = np.zeros((height, width), dtype=np.float32)
        grid_bwd = np.zeros((height, width), dtype=np.float32)

        # Densify all traces at 1m spacing
        geoms = self.gdf.geometry.values
        densified = shp_segmentize(geoms, max_segment_length=1.0)
        
        # Get trace weights (computed in _compute_trace_weights)
        trace_weights = self.gdf["trace_weight"].values if "trace_weight" in self.gdf.columns else np.ones(len(geoms))

        n_fwd = 0
        n_bwd = 0
        weighted_fwd = 0.0
        weighted_bwd = 0.0

        for i, dgeom in enumerate(densified):
            if dgeom is None or dgeom.is_empty:
                continue
            coords = get_coordinates(dgeom)[:, :2]
            if len(coords) < 2:
                continue
            
            # Get this trace's weight
            trace_weight = float(trace_weights[i])

            dx = coords[1:, 0] - coords[:-1, 0]
            dy = coords[1:, 1] - coords[:-1, 1]

            # Full 360 deg heading for direction separation
            seg_headings_360 = np.degrees(np.arctan2(dy, dx)) % 360.0

            mids = (coords[:-1] + coords[1:]) * 0.5
            cols = ((mids[:, 0] - minx) / self.pixel_size).astype(np.intp)
            rows = ((maxy - mids[:, 1]) / self.pixel_size).astype(np.intp)

            # Bounds check
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            rows_v = rows[valid]
            cols_v = cols[valid]
            headings_v = seg_headings_360[valid]

            # Split by direction - USE TRACE WEIGHT instead of 1.0
            fwd_mask = headings_v < 180.0
            bwd_mask = ~fwd_mask

            if fwd_mask.any():
                np.add.at(grid_fwd, (rows_v[fwd_mask], cols_v[fwd_mask]), trace_weight)
                n_fwd += int(fwd_mask.sum())
                weighted_fwd += trace_weight * fwd_mask.sum()
            if bwd_mask.any():
                np.add.at(grid_bwd, (rows_v[bwd_mask], cols_v[bwd_mask]), trace_weight)
                n_bwd += int(bwd_mask.sum())
                weighted_bwd += trace_weight * bwd_mask.sum()

        elapsed = time.time() - t0
        print(f"    Forward segments: {n_fwd:,} (weighted: {weighted_fwd:,.0f})")
        print(f"    Backward segments: {n_bwd:,} (weighted: {weighted_bwd:,.0f})")
        print(f"    Rasterized in {elapsed:.1f}s")

        del densified
        gc.collect()

        return grid_fwd, grid_bwd, width, height, minx, maxy

    # ------------------------------------------------------------------
    #  KDE + DUAL SKELETONIZATION
    # ------------------------------------------------------------------

    def _process_direction_grid(self, grid, label, width, height):
        """Blur, dual-threshold, cleanup, and skeletonize ONE direction."""
        density = gaussian_filter(grid, sigma=self.blur_sigma)

        d_max = density.max()
        if d_max == 0:
            return np.zeros_like(grid, dtype=bool)
        density /= d_max

        # Primary threshold (main + moderate-traffic roads)
        binary_main = density > self.threshold_frac

        # Recovery threshold for sparse single-trace roads (func_class 5)
        recovery_frac = self.threshold_frac * self.recovery_factor
        binary_recovery = density > recovery_frac
        # Only recovery pixels far from primary road blobs
        # Reduced disk(5) from disk(8) to allow small roads closer to main roads
        main_expanded = binary_dilation(binary_main, structure=disk(5))
        binary_recovery = binary_recovery & ~main_expanded

        binary = binary_main | binary_recovery
        del binary_main, binary_recovery, main_expanded, density

        print(f"      {label}: {binary.sum():,} road pixels")

        # Morphological cleanup
        binary = binary_closing(binary, structure=disk(2))
        binary = binary_opening(binary, structure=disk(1))

        # Skeletonize
        skeleton = skeletonize(binary)
        print(f"      {label}: {skeleton.sum():,} skeleton pixels")

        del binary
        return skeleton

    def kde_and_skeletonize(self, grid_fwd, grid_bwd, width, height):
        """
        Process each direction independently, then combine skeletons.
        Divided roads -> 2 separate skeletons (dual centerlines).
        Undivided roads -> 2 overlapping skeletons (merged in Phase 5).
        """
        print("  Applying KDE + dual-direction skeletonization...")

        skel_fwd = self._process_direction_grid(
            grid_fwd, "Forward", width, height
        )
        del grid_fwd
        gc.collect()

        skel_bwd = self._process_direction_grid(
            grid_bwd, "Backward", width, height
        )
        del grid_bwd
        gc.collect()

        # Combine both skeletons
        skeleton = skel_fwd | skel_bwd
        total = skeleton.sum()
        print(f"    Combined skeleton: {total:,} pixels")

        del skel_fwd, skel_bwd
        gc.collect()

        return skeleton

    # ------------------------------------------------------------------
    #  ROUNDABOUT DETECTION (Arc-Based + Circle Fitting)
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_circle_3pt(p1, p2, p3):
        """
        Fit a circle through 3 points using the circumscribed circle formula.
        Returns (center_x, center_y, radius) or None if collinear.
        """
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx, cy = p3[0], p3[1]

        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return None

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
              (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
              (cx**2 + cy**2) * (bx - ax)) / d
        r = np.sqrt((ax - ux)**2 + (ay - uy)**2)
        return ux, uy, r

    def _extract_curved_arcs(self, coords):
        """
        Extract contiguous curved sub-arcs from a trace's coordinates.

        A curved arc is a sequence of consecutive segments where the heading
        changes consistently in one direction (all CW or all CCW). This
        isolates the roundabout portion of a trace from the straight
        entry/exit segments.

        Returns list of (arc_coords, cumulative_heading_change) tuples.
        """
        if len(coords) < 4:
            return []

        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])
        seg_lengths = np.sqrt(dx**2 + dy**2)

        # Compute headings and heading changes
        headings = np.arctan2(dy, dx)
        dh = np.diff(headings)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        arcs = []
        i = 0
        n = len(dh)

        while i < n:
            # Skip near-zero heading changes (straight segments)
            if abs(dh[i]) < 0.03 or seg_lengths[i] < 0.3:
                i += 1
                continue

            # Start a new arc
            arc_sign = 1 if dh[i] > 0 else -1
            arc_sum = dh[i]
            arc_start = i
            j = i + 1
            zero_run = 0

            while j < n:
                if abs(dh[j]) < 0.03 or seg_lengths[j] < 0.3:
                    zero_run += 1
                    if zero_run > 2:  # allow up to 2 near-straight segments in arc
                        break
                    j += 1
                    continue
                zero_run = 0
                if np.sign(dh[j]) == arc_sign:
                    arc_sum += dh[j]
                    j += 1
                else:
                    break

            arc_end = j
            n_arc_segs = arc_end - arc_start

            i = arc_end  # advance main pointer

            # Need at least 3 turning segments and >= 45 degrees total
            if n_arc_segs < 3 or abs(arc_sum) < np.radians(45):
                continue

            # Extract the arc coordinates (vertex arc_start to arc_end+1)
            arc_v_start = arc_start
            arc_v_end = min(arc_end + 1, len(coords))
            arc_coords = coords[arc_v_start:arc_v_end]

            if len(arc_coords) < 4:
                continue

            arcs.append((arc_coords, arc_sum))

        return arcs

    def detect_roundabouts(self):
        """
        Detect roundabouts using arc-based analysis + 3-point circle fitting.

        Instead of analyzing whole traces (which include straight entry/exit
        segments), this method:
        1. Extracts curved sub-arcs from each trace
        2. Fits circles to the curved portions using 3-point circumscription
        3. Validates the arc sits on a tight circle (radius 5-60m)
        4. Clusters arc centers across multiple traces
        5. Generates roundabout circles where multiple traces confirm
        """
        print("  Detecting roundabouts from trace curvature arcs...")
        candidates = []

        for trace_idx, geom in enumerate(self.gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            coords = np.array(geom.coords)
            if len(coords) < 5:
                continue

            total_length = geom.length
            if total_length < 15:
                continue

            # Extract curved sub-arcs from this trace
            arcs = self._extract_curved_arcs(coords)

            for arc_coords, arc_heading_change in arcs:
                n = len(arc_coords)

                # Fit circle using 3 well-spaced points on the arc
                p1 = arc_coords[0]
                p2 = arc_coords[n // 2]
                p3 = arc_coords[-1]

                result = self._fit_circle_3pt(p1, p2, p3)
                if result is None:
                    continue

                cx, cy, radius = result

                # Roundabout radius check (8-45m - tighter range)
                if radius < 8 or radius > 45:
                    continue

                # Verify arc points lie roughly on this circle
                dists = np.sqrt(
                    (arc_coords[:, 0] - cx)**2 + (arc_coords[:, 1] - cy)**2
                )
                dist_std = np.std(dists)
                if dist_std > 0.45 * radius:
                    continue

                # Arc compactness: bbox shouldn't be too tiny
                bbox_w = np.ptp(arc_coords[:, 0])
                bbox_h = np.ptp(arc_coords[:, 1])
                if max(bbox_w, bbox_h) < 6:
                    continue

                candidates.append((
                    cx, cy, radius,
                    abs(arc_heading_change),
                    trace_idx
                ))

        if not candidates:
            print("    No roundabout arc candidates found")
            return []

        print(f"    Found {len(candidates)} curved arc candidates")

        # Cluster nearby arc centers from DIFFERENT traces
        centers = np.array([(c[0], c[1]) for c in candidates])
        radii = np.array([c[2] for c in candidates])
        heading_changes = np.array([c[3] for c in candidates])
        trace_ids = np.array([c[4] for c in candidates])

        tree = cKDTree(centers)
        visited = set()
        clusters = []

        # Tighter cluster radius (25m instead of 40m)
        for i in range(len(centers)):
            if i in visited:
                continue
            neighbors = tree.query_ball_point(centers[i], r=25)
            cluster = [j for j in neighbors if j not in visited]
            for j in cluster:
                visited.add(j)
            if cluster:
                clusters.append(cluster)

        # Generate circular geometries from confirmed clusters
        roundabout_lines = []
        for cluster in clusters:
            c_centers = centers[cluster]
            c_radii = radii[cluster]
            c_headings = heading_changes[cluster]
            c_traces = trace_ids[cluster]

            # Count unique traces contributing to this cluster
            n_unique_traces = len(set(c_traces.tolist()))
            n_arcs = len(cluster)

            # TIGHTENED Acceptance criteria to reduce false positives:
            #   - >= 4 unique traces (strong evidence), OR
            #   - >= 3 unique traces with avg heading > 90 deg, OR
            #   - >= 5 arcs from any number of traces with consistent radii
            avg_heading = np.mean(c_headings)
            max_heading = np.max(c_headings)
            radius_std = np.std(c_radii)
            avg_radius = np.mean(c_radii)
            
            # Radius consistency check: std should be < 30% of mean
            radius_consistent = radius_std < 0.3 * avg_radius if avg_radius > 0 else False

            accepted = False
            if n_unique_traces >= 4:
                accepted = True
            elif n_unique_traces >= 3 and avg_heading > np.radians(90):
                accepted = True
            elif n_arcs >= 5 and radius_consistent and avg_heading > np.radians(70):
                accepted = True
            
            if not accepted:
                continue

            # Weight by heading change for center/radius estimation
            weights = c_headings / c_headings.sum()
            avg_cx = np.average(c_centers[:, 0], weights=weights)
            avg_cy = np.average(c_centers[:, 1], weights=weights)
            avg_r = max(float(np.average(c_radii, weights=weights)), 8.0)

            # Generate circle
            n_pts = 48
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
            circle = [
                (avg_cx + avg_r * np.cos(a), avg_cy + avg_r * np.sin(a))
                for a in angles
            ]
            roundabout_lines.append(LineString(circle))

        print(
            f"    Detected {len(roundabout_lines)} roundabouts "
            f"from {len(candidates)} arc candidates "
            f"({len(clusters)} clusters, "
            f"{len(clusters) - len(roundabout_lines)} rejected)"
        )
        return roundabout_lines

    # ------------------------------------------------------------------
    #  VECTORIZATION (skeleton pixels -> LineStrings)
    # ------------------------------------------------------------------

    def vectorize_skeleton(self, skeleton, minx, maxy):
        """
        Convert skeleton pixel mask to LineStrings via graph tracing.
        Builds an adjacency graph on skeleton pixels, identifies
        junctions and endpoints, then traces paths between them.
        """
        print("  Vectorizing skeleton...")
        t0 = time.time()

        skel_rows, skel_cols = np.where(skeleton)
        if len(skel_rows) == 0:
            return []
        print(f"    {len(skel_rows)} skeleton pixels to trace")

        # Build adjacency graph from 8-connected pixels
        pixel_set = set(zip(skel_rows.tolist(), skel_cols.tolist()))
        G = nx.Graph()
        for r, c in pixel_set:
            G.add_node((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in pixel_set:
                        G.add_edge((r, c), nb)

        ps = self.pixel_size

        def px2world(r, c):
            return (minx + (c + 0.5) * ps, maxy - (r + 0.5) * ps)

        # Classify nodes
        junctions = [n for n in G.nodes() if G.degree(n) > 2]
        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        terminals = set(junctions + endpoints)
        if not terminals and G.number_of_nodes() > 0:
            terminals = {next(iter(G.nodes()))}

        visited_edges = set()
        lines = []

        def _trace(start, neighbor):
            path = [start, neighbor]
            visited_edges.add((start, neighbor))
            visited_edges.add((neighbor, start))
            cur, prev = neighbor, start
            while cur not in terminals:
                nbrs = [
                    n for n in G.neighbors(cur)  # noqa: F821
                    if n != prev and (cur, n) not in visited_edges
                ]
                if not nbrs:
                    break
                nxt = nbrs[0]
                visited_edges.add((cur, nxt))
                visited_edges.add((nxt, cur))
                path.append(nxt)
                prev, cur = cur, nxt
            return path

        for t in terminals:
            for nb in list(G.neighbors(t)):
                if (t, nb) not in visited_edges:
                    path = _trace(t, nb)
                    if len(path) >= 2:
                        coords = [px2world(r, c) for r, c in path]
                        lines.append(LineString(coords))

        # Leftover cycles
        for u, v in list(G.edges()):
            if (u, v) not in visited_edges:
                path = _trace(u, v)
                if len(path) >= 2:
                    coords = [px2world(r, c) for r, c in path]
                    lines.append(LineString(coords))

        elapsed = time.time() - t0
        print(f"    Traced {len(lines)} raw lines in {elapsed:.1f}s")

        del G, pixel_set
        gc.collect()

        return lines

    # ------------------------------------------------------------------
    #  ISOLATED TRACE RECOVERY
    # ------------------------------------------------------------------

    def recover_isolated_traces(self, skeleton_lines):
        """
        Recover VPD traces missed by the KDE skeleton.

        These are typically func_class 5 roads with only 1-2 traces that
        fall below the KDE density threshold. Uses two strategies:
        1. Fully isolated traces (no skeleton within 8m)
        2. Partially uncovered traces (>50% of length outside skeleton buffer)
        """
        if not skeleton_lines or self.gdf is None:
            return skeleton_lines

        print("  Recovering isolated VPD traces (missed by KDE)...")

        from shapely import STRtree as _STRtree
        from shapely.ops import unary_union

        tree = _STRtree(skeleton_lines)

        # Build a buffer union of all skeleton lines for partial coverage check
        skel_buf = unary_union([l.buffer(8.0) for l in skeleton_lines])

        recovered = []
        for geom in self.gdf.geometry:
            if geom is None or geom.is_empty or geom.length < 15:
                continue

            # Strategy 1: Check if ANY skeleton line is within 8m of this trace
            nearby = tree.query(geom, predicate="dwithin", distance=8.0)
            if len(nearby) == 0:
                # Fully isolated -> add simplified version
                simplified = geom.simplify(2.0, preserve_topology=True)
                if simplified.length >= 12.0 and not simplified.is_empty:
                    recovered.append(simplified)
                continue

            # Strategy 2: Check partial coverage — if >50% of trace
            # is outside the skeleton buffer, recover the uncovered portion
            try:
                outside = geom.difference(skel_buf)
                if outside.is_empty:
                    continue
                uncovered_frac = outside.length / geom.length
                if uncovered_frac > 0.50 and outside.length >= 15.0:
                    # Recover the uncovered portion
                    if outside.geom_type == 'MultiLineString':
                        for part in outside.geoms:
                            if part.length >= 10.0:
                                simplified = part.simplify(2.0, preserve_topology=True)
                                if not simplified.is_empty and simplified.length >= 8.0:
                                    recovered.append(simplified)
                    elif outside.geom_type == 'LineString':
                        simplified = outside.simplify(2.0, preserve_topology=True)
                        if not simplified.is_empty and simplified.length >= 8.0:
                            recovered.append(simplified)
            except Exception:
                pass

        n = len(recovered)
        if n > 0:
            total_km = sum(l.length for l in recovered) / 1000.0
            print(f"    Recovered {n} isolated/partial traces ({total_km:.1f} km)")
            return skeleton_lines + recovered
        else:
            print("    No isolated traces found")
            return skeleton_lines

    # ------------------------------------------------------------------
    #  FILTERING & EXPORT
    # ------------------------------------------------------------------

    def filter_lines(self, lines):
        """Remove short segments and simplify slightly."""
        filtered = []
        for line in lines:
            if line.length >= self.min_segment_length:
                simple = line.simplify(
                    self.pixel_size * 0.5, preserve_topology=True
                )
                if not simple.is_empty and simple.length >= self.min_segment_length:
                    filtered.append(simple)
        print(
            f"    Filtered: {len(lines)} -> {len(filtered)} lines "
            f"(min length {self.min_segment_length}m)"
        )
        return filtered

    def export(self, lines, roundabout_count=0):
        """Export skeleton lines as GeoPackage in WGS84."""
        print(f"  Exporting skeleton to {self.output_path}...")
        gdf = gpd.GeoDataFrame(geometry=lines, crs=self.crs_projected)

        # Tag roundabouts vs regular skeleton lines
        sources = ["VPD"] * len(lines)
        if roundabout_count > 0:
            for i in range(len(lines) - roundabout_count, len(lines)):
                sources[i] = "roundabout"
        gdf["source"] = sources
        gdf["length_m"] = gdf.geometry.length

        out_gdf = gdf.to_crs("EPSG:4326")
        out_gdf.to_file(self.output_path, driver="GPKG")

        self.skeleton_gdf = out_gdf
        print(f"  Saved {len(out_gdf)} skeleton segments.")
        return out_gdf

    # ------------------------------------------------------------------
    #  MAIN PIPELINE
    # ------------------------------------------------------------------

    def run(self):
        print("=" * 60)
        print("  PHASE 2: VPD Skeleton (Direction-Aware + Roundabouts)")
        print("=" * 60)
        t0 = time.time()

        self.load_and_project()

        # Direction-aware dual-grid rasterization
        grid_fwd, grid_bwd, width, height, minx, maxy = self.rasterize_traces()

        # Independent skeletonization per direction
        skeleton = self.kde_and_skeletonize(grid_fwd, grid_bwd, width, height)

        # Vectorize
        lines = self.vectorize_skeleton(skeleton, minx, maxy)
        del skeleton
        gc.collect()

        lines = self.filter_lines(lines)

        # Recover isolated VPD traces the KDE missed (func_class 5)
        lines = self.recover_isolated_traces(lines)

        # Detect and add roundabouts
        roundabout_lines = self.detect_roundabouts()
        if roundabout_lines:
            lines = lines + roundabout_lines

        self.export(lines, roundabout_count=len(roundabout_lines) if roundabout_lines else 0)

        elapsed = time.time() - t0
        total_km = sum(l.length for l in lines) / 1000.0 if lines else 0
        n_roundabouts = len(roundabout_lines) if roundabout_lines else 0
        print(f"\nPhase 2 complete in {elapsed:.1f}s")
        print(f"  Input:       {len(self.gdf)} VPD traces")
        print(f"  Skeleton:    {len(lines) - n_roundabouts} centerlines")
        print(f"  Roundabouts: {n_roundabouts}")
        print(f"  Total:       {len(lines)} segments ({total_km:.1f} km)")

        return self.skeleton_gdf


if __name__ == "__main__":
    INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "interim_sample_phase1.gpkg")
    OUTPUT_FILE = os.path.join(
        PROJECT_ROOT, "data", "interim_skeleton_phase2.gpkg"
    )

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Run Phase 1 first.")
        sys.exit(1)

    skel = KDESkeletonizer(INPUT_FILE, OUTPUT_FILE)
    skel.run()
