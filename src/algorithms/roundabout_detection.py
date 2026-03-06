"""
Roundabout Detection Module

Implements curl-based roundabout detection from vehicle heading data.
Based on the principle that vehicles traversing a roundabout exhibit
consistent rotational motion (high curl in the velocity field).

Two methods are provided:
  1. curl_based_detection: Uses velocity field curl interpolation
  2. arc_based_detection: Uses trace curvature + circle fitting

Both methods can be combined for more robust detection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely.geometry import LineString


@dataclass
class RoundaboutConfig:
    """Configuration for roundabout detection."""
    
    # Speed filtering for curl-based detection
    min_speed_kmh: float = 15.0  # Minimum speed to consider (filter stopped vehicles)
    max_speed_kmh: float = 60.0  # Maximum speed (filter highway traces)
    
    # Curl-based detection parameters
    grid_resolution: int = 80  # Grid points per axis for interpolation
    curl_threshold_ratio: float = 0.85  # Threshold as ratio of max curl
    min_curl_value: float = 0.2  # Minimum absolute curl to consider
    
    # Circle fitting parameters
    min_radius_m: float = 6.0  # Minimum roundabout radius (relaxed for small roundabouts)
    max_radius_m: float = 60.0  # Maximum roundabout radius (relaxed)
    
    # Arc-based detection parameters
    min_arc_heading_deg: float = 60.0  # Minimum heading change for arc (relaxed from 90)
    min_arc_segments: int = 3  # Minimum segments in an arc (relaxed from 4)
    max_radius_std_ratio: float = 0.35  # Max std/mean for radius consistency (relaxed)
    min_arc_bbox_m: float = 5.0  # Minimum arc bounding box dimension (reduced from 10)
    
    # Arc extraction tuning for fine-sampled data
    arc_zero_threshold_rad: float = 0.015  # Near-zero heading threshold (relaxed from 0.03)
    arc_min_segment_length_m: float = 0.2  # Min segment length (relaxed from 0.3)
    arc_max_zero_run: int = 5  # Max consecutive near-straight segments (relaxed from 2)
    
    # Curvature-density detection (new method for fine-sampled data)
    use_curvature_density: bool = True
    curvature_grid_size_m: float = 15.0  # Grid cell size for curvature density
    curvature_min_heading_per_point: float = 5.0  # Min degrees per point in cell
    curvature_min_points_per_cell: int = 8  # Min points in high-curvature cell
    curvature_min_cells_for_roundabout: int = 3  # Min adjacent high-curvature cells
    
    # Clustering parameters
    cluster_radius_m: float = 25.0  # Cluster detection radius
    
    # Confirmation thresholds (relaxed for better detection)
    min_unique_traces: int = 2  # Minimum traces confirming roundabout (relaxed from 4)
    min_arcs_for_radius_only: int = 4  # Arcs needed if fewer traces (relaxed from 8)
    min_avg_heading_deg: float = 60.0  # Minimum average heading for radius-only (relaxed from 100)
    
    # Multi-direction validation (relaxed)
    min_entry_directions: int = 1  # Minimum distinct entry directions (relaxed from 2)
    direction_separation_deg: float = 45.0  # Minimum angular separation between directions (relaxed from 60)
    
    # Output parameters
    circle_points: int = 48  # Points to generate roundabout circle


class RoundaboutDetector:
    """
    Detect roundabouts from vehicle trajectory data.
    
    Combines curl-based detection (for identifying rotation centers)
    with arc-based validation (for confirming roundabout geometry).
    """
    
    def __init__(self, config: RoundaboutConfig = None):
        self.config = config or RoundaboutConfig()
        self.detected_roundabouts: List[dict] = []
    
    # ------------------------------------------------------------------
    #  CURL-BASED DETECTION (Velocity Field Analysis)
    # ------------------------------------------------------------------
    
    def detect_from_traces_curl(
        self,
        traces: List[dict],  # List of {coords, heading, speed} dicts
    ) -> List[Tuple[float, float, float]]:
        """
        Detect roundabouts using velocity field curl analysis.
        
        The curl of a velocity field measures local rotation. High positive
        curl indicates counter-clockwise rotation (UK-style roundabouts),
        high negative curl indicates clockwise rotation (continental Europe).
        
        Args:
            traces: List of trace dictionaries with 'coords', 'heading', 'speed'
            
        Returns:
            List of (center_x, center_y, estimated_radius) tuples
        """
        config = self.config
        
        # Collect points with velocity components
        all_points = []
        
        for trace in traces:
            coords = np.asarray(trace.get("coords", []))
            headings = trace.get("headings", [])
            speeds = trace.get("speeds", [])
            
            if len(coords) < 2:
                continue
            
            for i in range(len(coords)):
                speed = speeds[i] if i < len(speeds) else 30.0  # default speed
                
                # Speed filtering
                if speed < config.min_speed_kmh or speed > config.max_speed_kmh:
                    continue
                
                heading = headings[i] if i < len(headings) else 0.0
                heading_rad = np.radians(heading)
                
                # Velocity components (unit vectors)
                vx = np.cos(heading_rad)
                vy = np.sin(heading_rad)
                
                all_points.append({
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "vx": vx,
                    "vy": vy,
                    "speed": speed,
                })
        
        if len(all_points) < 10:
            return []
        
        # Convert to arrays
        x = np.array([p["x"] for p in all_points])
        y = np.array([p["y"] for p in all_points])
        vx = np.array([p["vx"] for p in all_points])
        vy = np.array([p["vy"] for p in all_points])
        
        # Create interpolation grid
        n = config.grid_resolution
        grid_x = np.linspace(x.min(), x.max(), n)
        grid_y = np.linspace(y.min(), y.max(), n)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        
        # Interpolate velocity field
        try:
            grid_vx = griddata((x, y), vx, (grid_x, grid_y), method='linear')
            grid_vy = griddata((x, y), vy, (grid_x, grid_y), method='linear')
        except Exception:
            return []
        
        if grid_vx is None or grid_vy is None:
            return []
        
        # Compute curl: ∂vy/∂x - ∂vx/∂y
        dvx_dy = np.gradient(grid_vx, axis=0)
        dvy_dx = np.gradient(grid_vy, axis=1)
        curl = dvy_dx - dvx_dy
        
        # Find high-curl regions
        max_curl = np.nanmax(np.abs(curl))
        if max_curl < config.min_curl_value:
            return []
        
        threshold = config.curl_threshold_ratio * max_curl
        high_curl_mask = np.abs(curl) >= threshold
        
        # Extract high-curl points
        high_curl_indices = np.argwhere(high_curl_mask)
        if len(high_curl_indices) == 0:
            return []
        
        high_curl_points = []
        for idx in high_curl_indices:
            row_idx, col_idx = idx
            gx = grid_x[row_idx, col_idx]
            gy = grid_y[row_idx, col_idx]
            if not np.isnan(gx) and not np.isnan(gy):
                high_curl_points.append((gx, gy))
        
        if not high_curl_points:
            return []
        
        # Cluster high-curl points
        centers = self._cluster_curl_points(high_curl_points)
        
        # Estimate radius for each center
        roundabouts = []
        for cx, cy in centers:
            radius = self._estimate_radius_from_traces(cx, cy, traces)
            if config.min_radius_m <= radius <= config.max_radius_m:
                roundabouts.append((cx, cy, radius))
        
        return roundabouts
    
    def _cluster_curl_points(
        self,
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Cluster nearby high-curl points into roundabout centers."""
        if not points:
            return []
        
        coords = np.array(points)
        tree = cKDTree(coords)
        
        visited = set()
        centers = []
        
        for i in range(len(coords)):
            if i in visited:
                continue
            
            neighbors = tree.query_ball_point(
                coords[i],
                r=self.config.cluster_radius_m
            )
            
            cluster = [j for j in neighbors if j not in visited]
            for j in cluster:
                visited.add(j)
            
            if cluster:
                # Compute cluster centroid
                cluster_coords = coords[cluster]
                center_x = float(np.mean(cluster_coords[:, 0]))
                center_y = float(np.mean(cluster_coords[:, 1]))
                centers.append((center_x, center_y))
        
        return centers
    
    def _estimate_radius_from_traces(
        self,
        cx: float,
        cy: float,
        traces: List[dict],
        search_radius: float = 60.0,
    ) -> float:
        """Estimate roundabout radius from nearby trace points."""
        distances = []
        
        for trace in traces:
            coords = np.asarray(trace.get("coords", []))
            if len(coords) == 0:
                continue
            
            # Find points near the center
            dists = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
            nearby_mask = dists < search_radius
            
            if nearby_mask.any():
                distances.extend(dists[nearby_mask].tolist())
        
        if not distances:
            return 0.0
        
        # Use median distance as radius estimate
        return float(np.median(distances))
    
    # ------------------------------------------------------------------
    #  ARC-BASED DETECTION (Curvature + Circle Fitting)
    # ------------------------------------------------------------------
    
    def detect_from_traces_arc(
        self,
        traces: List[dict],  # List of {coords} dicts with projected coordinates
    ) -> List[Tuple[float, float, float]]:
        """
        Detect roundabouts using arc-based curvature analysis.
        
        Extracts curved sub-arcs from traces, fits circles to them,
        and clusters matching circles to confirm roundabouts.
        
        Args:
            traces: List of trace dictionaries with 'coords'
            
        Returns:
            List of (center_x, center_y, radius) tuples
        """
        config = self.config
        candidates = []
        
        for trace_idx, trace in enumerate(traces):
            coords = np.asarray(trace.get("coords", []))
            if len(coords) < 5:
                continue
            
            # Extract curved arcs from this trace
            arcs = self._extract_curved_arcs(coords)
            
            for arc_coords, arc_heading_change in arcs:
                # Skip arcs with insufficient heading change  
                # Use config value, default to 60 degrees (relaxed)
                min_heading_rad = np.radians(config.min_arc_heading_deg)
                if abs(arc_heading_change) < min_heading_rad:
                    continue
                
                # Fit circle using 3-point method
                n = len(arc_coords)
                if n < 3:
                    continue
                
                p1 = arc_coords[0]
                p2 = arc_coords[n // 2]
                p3 = arc_coords[-1]
                
                result = self._fit_circle_3pt(p1, p2, p3)
                if result is None:
                    continue
                
                cx, cy, radius = result
                
                # Radius check
                if radius < config.min_radius_m or radius > config.max_radius_m:
                    continue
                
                # Verify arc points lie on this circle
                dists = np.sqrt(
                    (arc_coords[:, 0] - cx)**2 + (arc_coords[:, 1] - cy)**2
                )
                dist_std = np.std(dists)
                
                if dist_std > config.max_radius_std_ratio * radius:
                    continue
                
                # Arc should span reasonable area (use config value)
                min_bbox = getattr(config, 'min_arc_bbox_m', 5.0)
                bbox_w = np.ptp(arc_coords[:, 0])
                bbox_h = np.ptp(arc_coords[:, 1])
                if max(bbox_w, bbox_h) < min_bbox:
                    continue
                
                # Compute entry/exit angles relative to circle center
                arc_start = arc_coords[0]
                arc_end = arc_coords[-1]
                entry_angle = np.arctan2(arc_start[1] - cy, arc_start[0] - cx)
                exit_angle = np.arctan2(arc_end[1] - cy, arc_end[0] - cx)
                
                candidates.append({
                    "center_x": cx,
                    "center_y": cy,
                    "radius": radius,
                    "heading_change": abs(arc_heading_change),
                    "trace_idx": trace_idx,
                    "arc_start_angle": entry_angle,
                    "arc_end_angle": exit_angle,
                })
        
        if not candidates:
            return []
        
        # Cluster candidates
        return self._cluster_arc_candidates(candidates)
    
    def _extract_curved_arcs(
        self,
        coords: np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Extract contiguous curved sub-arcs from coordinates.
        
        A curved arc is a sequence where heading changes consistently
        in one direction (all CW or all CCW).
        
        Returns:
            List of (arc_coords, total_heading_change) tuples
        """
        config = self.config
        
        if len(coords) < 4:
            return []
        
        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])
        seg_lengths = np.sqrt(dx**2 + dy**2)
        
        # Compute headings and heading changes
        headings = np.arctan2(dy, dx)
        dh = np.diff(headings)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        
        arcs = []
        i = 0
        n = len(dh)
        
        # Use config values for thresholds (with defaults for backward compatibility)
        zero_thresh = getattr(config, 'arc_zero_threshold_rad', 0.015)
        min_seg_len = getattr(config, 'arc_min_segment_length_m', 0.2)
        max_zero_run = getattr(config, 'arc_max_zero_run', 5)
        
        while i < n:
            # Skip near-zero heading changes (straight segments)
            if abs(dh[i]) < zero_thresh or seg_lengths[i] < min_seg_len:
                i += 1
                continue
            
            # Start a new arc
            arc_sign = 1 if dh[i] > 0 else -1
            arc_sum = dh[i]
            arc_start = i
            j = i + 1
            zero_run = 0
            
            while j < n:
                if abs(dh[j]) < zero_thresh or seg_lengths[j] < min_seg_len:
                    zero_run += 1
                    if zero_run > max_zero_run:  # Allow more near-straight segments
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
            i = arc_end
            
            # Need enough turning and heading change
            min_rad = np.radians(config.min_arc_heading_deg)
            if n_arc_segs < config.min_arc_segments or abs(arc_sum) < min_rad:
                continue
            
            # Extract arc coordinates
            arc_v_start = arc_start
            arc_v_end = min(arc_end + 1, len(coords))
            arc_coords = coords[arc_v_start:arc_v_end]
            
            if len(arc_coords) < 4:
                continue
            
            arcs.append((arc_coords, arc_sum))
        
        return arcs
    
    @staticmethod
    def _fit_circle_3pt(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Fit a circle through 3 points using circumscribed circle formula.
        
        Returns:
            (center_x, center_y, radius) or None if collinear
        """
        ax, ay = float(p1[0]), float(p1[1])
        bx, by = float(p2[0]), float(p2[1])
        cx, cy = float(p3[0]), float(p3[1])
        
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return None
        
        ux = ((ax**2 + ay**2) * (by - cy) +
              (bx**2 + by**2) * (cy - ay) +
              (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) +
              (bx**2 + by**2) * (ax - cx) +
              (cx**2 + cy**2) * (bx - ax)) / d
        r = np.sqrt((ax - ux)**2 + (ay - uy)**2)
        
        return ux, uy, r
    
    def _cluster_arc_candidates(
        self,
        candidates: List[dict],
    ) -> List[Tuple[float, float, float]]:
        """Cluster arc candidates and confirm roundabouts with multi-direction validation."""
        config = self.config
        
        if not candidates:
            return []
        
        centers = np.array([
            [c["center_x"], c["center_y"]] for c in candidates
        ])
        radii = np.array([c["radius"] for c in candidates])
        heading_changes = np.array([c["heading_change"] for c in candidates])
        trace_ids = np.array([c["trace_idx"] for c in candidates])
        
        # Store arc start/end points for direction analysis
        arc_start_angles = []
        arc_end_angles = []
        for c in candidates:
            arc_start_angles.append(c.get("arc_start_angle", 0))
            arc_end_angles.append(c.get("arc_end_angle", 0))
        arc_start_angles = np.array(arc_start_angles)
        arc_end_angles = np.array(arc_end_angles)
        
        tree = cKDTree(centers)
        visited = set()
        clusters = []
        
        for i in range(len(centers)):
            if i in visited:
                continue
            
            neighbors = tree.query_ball_point(centers[i], r=config.cluster_radius_m)
            cluster = [j for j in neighbors if j not in visited]
            
            for j in cluster:
                visited.add(j)
            
            if cluster:
                clusters.append(cluster)
        
        # Confirm roundabouts from clusters with relaxed validation
        roundabouts = []
        
        for cluster in clusters:
            c_centers = centers[cluster]
            c_radii = radii[cluster]
            c_headings = heading_changes[cluster]
            c_traces = trace_ids[cluster]
            c_start_angles = arc_start_angles[cluster]
            c_end_angles = arc_end_angles[cluster]
            
            n_unique_traces = len(set(c_traces.tolist()))
            n_arcs = len(cluster)
            
            avg_heading = np.mean(c_headings)
            radius_std = np.std(c_radii)
            avg_radius = np.mean(c_radii)
            
            # VALIDATION 1: Minimum heading change (use config value)
            min_avg_heading_rad = np.radians(config.min_avg_heading_deg)
            if avg_heading < min_avg_heading_rad:
                continue
            
            # VALIDATION 2: Check for entry directions (relaxed)
            # Skip this check if we have enough unique traces
            if n_unique_traces < config.min_unique_traces:
                entry_bearings = c_start_angles[c_start_angles != 0]
                if len(entry_bearings) >= 2:
                    n_directions = self._count_distinct_directions(
                        entry_bearings, 
                        min_separation_deg=config.direction_separation_deg
                    )
                    if n_directions < config.min_entry_directions:
                        continue
            
            # VALIDATION 3: Radius consistency (relaxed)
            # True roundabouts have consistent radii across arcs
            if avg_radius > 0:
                if radius_std > config.max_radius_std_ratio * avg_radius:
                    # Allow if we have many arcs (statistical strength)
                    if n_arcs < 8:
                        continue
            
            # VALIDATION 4: Need enough evidence (relaxed)
            # Require EITHER: min_unique_traces OR min_arcs_for_radius_only arcs
            if n_unique_traces < config.min_unique_traces and n_arcs < config.min_arcs_for_radius_only:
                continue
            
            # Weight by heading change for better center/radius estimation
            weights = c_headings / c_headings.sum()
            final_cx = float(np.average(c_centers[:, 0], weights=weights))
            final_cy = float(np.average(c_centers[:, 1], weights=weights))
            final_r = max(float(np.average(c_radii, weights=weights)), config.min_radius_m)
            
            roundabouts.append((final_cx, final_cy, final_r))
        
        return roundabouts
    
    def _count_distinct_directions(
        self,
        bearings: np.ndarray,
        min_separation_deg: float = 60,
    ) -> int:
        """
        Count distinct angular directions from a set of bearings.
        
        Directions within min_separation_deg are considered the same direction.
        """
        if len(bearings) == 0:
            return 0
        
        # Normalize to [0, 360)
        bearings = np.mod(bearings, 2 * np.pi)
        bearings = np.sort(bearings)
        
        min_sep = np.radians(min_separation_deg)
        
        # Greedy clustering
        distinct = 1
        last_bearing = bearings[0]
        
        for b in bearings[1:]:
            # Angular difference accounting for wrap-around
            diff = min(abs(b - last_bearing), 2 * np.pi - abs(b - last_bearing))
            if diff >= min_sep:
                distinct += 1
                last_bearing = b
        
        return distinct
    
    # ------------------------------------------------------------------
    #  COMBINED DETECTION
    # ------------------------------------------------------------------
    
    def detect(
        self,
        traces: List[dict],
        use_curl: bool = False,
        use_arc: bool = True,
    ) -> List[dict]:
        """
        Detect roundabouts using combined methods.
        
        Args:
            traces: List of trace dictionaries with 'coords' (required)
                    and optionally 'headings', 'speeds'
            use_curl: Use curl-based detection
            use_arc: Use arc-based detection
            
        Returns:
            List of roundabout dictionaries with 'geometry', 'center', 'radius'
        """
        all_roundabouts = []
        
        if use_curl:
            curl_results = self.detect_from_traces_curl(traces)
            all_roundabouts.extend(curl_results)
        
        if use_arc:
            arc_results = self.detect_from_traces_arc(traces)
            all_roundabouts.extend(arc_results)
        
        if not all_roundabouts:
            return []
        
        _ = self.config  # Silence unused variable warning
        
        # Merge nearby detections
        merged = self._merge_nearby_detections(all_roundabouts)
        
        # Generate circle geometries
        results = []
        for cx, cy, radius in merged:
            circle = self._generate_circle(cx, cy, radius)
            results.append({
                "center_x": cx,
                "center_y": cy,
                "radius": radius,
                "geometry": circle,
                "source": "roundabout",
            })
        
        self.detected_roundabouts = results
        return results
    
    def _merge_nearby_detections(
        self,
        detections: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """Merge nearby roundabout detections."""
        if len(detections) <= 1:
            return detections
        
        centers = np.array([[d[0], d[1]] for d in detections])
        radii = np.array([d[2] for d in detections])
        
        tree = cKDTree(centers)
        visited = set()
        merged = []
        
        for i in range(len(centers)):
            if i in visited:
                continue
            
            neighbors = tree.query_ball_point(
                centers[i],
                r=self.config.cluster_radius_m
            )
            
            cluster = [j for j in neighbors if j not in visited]
            for j in cluster:
                visited.add(j)
            
            if cluster:
                # Average the cluster
                cx = float(np.mean(centers[cluster, 0]))
                cy = float(np.mean(centers[cluster, 1]))
                r = float(np.mean(radii[cluster]))
                merged.append((cx, cy, r))
        
        return merged
    
    def _generate_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
    ) -> LineString:
        """Generate a circular LineString geometry."""
        n = self.config.circle_points
        angles = np.linspace(0, 2 * np.pi, n, endpoint=True)
        points = [
            (cx + radius * np.cos(a), cy + radius * np.sin(a))
            for a in angles
        ]
        return LineString(points)


def detect_roundabouts_from_gdf(
    gdf,
    config: RoundaboutConfig = None,
    use_curl: bool = False,
    use_arc: bool = True,
) -> List[dict]:
    """
    Convenience function to detect roundabouts from a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame with LineString geometries
        config: RoundaboutConfig (optional)
        use_curl: Use curl-based detection
        use_arc: Use arc-based detection
        
    Returns:
        List of roundabout dictionaries with projected geometry
    """
    detector = RoundaboutDetector(config)
    
    traces = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not isinstance(geom, LineString):
            continue
        
        coords = np.array(geom.coords)
        if len(coords) < 3:
            continue
        
        # Compute headings from coordinates
        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])
        headings_rad = np.arctan2(dy, dx)
        headings_deg = np.degrees(headings_rad).tolist()
        # Pad to match coords length
        headings_deg.append(headings_deg[-1] if headings_deg else 0)
        
        traces.append({
            "coords": coords,
            "headings": headings_deg,
            "speeds": [30.0] * len(coords),  # Default speed
        })
    
    return detector.detect(traces, use_curl=use_curl, use_arc=use_arc)


@dataclass
class RoundaboutPostFilterConfig:
    """Configuration for post-processing roundabout validation."""
    
    # Trace density validation - check if enough traces pass near the circle
    min_traces_near_circle: int = 3  # Minimum distinct traces near circumference
    circle_buffer_m: float = 12.0  # Buffer around circle for trace matching
    min_points_on_circle: int = 10  # Minimum total points near circumference
    
    # Entry direction validation - real roundabouts have multi-directional traffic
    min_entry_directions: int = 2  # Minimum distinct entry directions
    direction_bucket_deg: float = 60.0  # Angular bucket size for direction counting
    
    # Circularity validation - traces should follow the circle arc
    min_arc_coverage_ratio: float = 0.25  # Fraction of circle that must have traces
    max_radial_std_ratio: float = 0.45  # Max std/mean for point distance from center
    
    # Duplicate removal
    merge_distance_m: float = 30.0  # Merge roundabouts within this distance
    
    # Size validation
    enforce_radius_bounds: bool = True
    min_radius_m: float = 8.0  # Minimum realistic roundabout radius
    max_radius_m: float = 50.0  # Maximum realistic roundabout radius
    
    # Statistical filtering
    filter_radius_outliers: bool = True
    radius_outlier_std: float = 2.5  # Remove radii beyond this many std deviations


def validate_roundabouts_post_detection(
    roundabouts: List[dict],
    traces: List[dict],
    config: RoundaboutPostFilterConfig = None,
) -> List[dict]:
    """
    Post-processing validation to filter out false positive roundabouts.
    
    This function validates each detected roundabout against the actual
    trace data to remove incorrect detections.
    
    Args:
        roundabouts: List of detected roundabout dicts with center_x, center_y, radius
        traces: List of trace dicts with 'coords' array (projected coordinates)
        config: Post-filter configuration
        
    Returns:
        Filtered list of validated roundabouts
    """
    if not roundabouts:
        return []
    
    config = config or RoundaboutPostFilterConfig()
    
    # Collect all trace points with trace IDs
    all_points = []
    point_trace_ids = []
    point_headings = []
    
    for trace_idx, trace in enumerate(traces):
        coords = np.asarray(trace.get("coords", []))
        if len(coords) < 2:
            continue
        
        # Compute headings for entry direction analysis
        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])
        headings = np.arctan2(dy, dx)
        headings = np.append(headings, headings[-1] if len(headings) > 0 else 0)
        
        for i, coord in enumerate(coords):
            all_points.append(coord[:2])
            point_trace_ids.append(trace_idx)
            point_headings.append(headings[i])
    
    if not all_points:
        return []
    
    all_points = np.array(all_points)
    point_trace_ids = np.array(point_trace_ids)
    point_headings = np.array(point_headings)
    
    # Build KDTree for fast spatial queries
    tree = cKDTree(all_points)
    
    validated = []
    
    for ra in roundabouts:
        cx = ra["center_x"]
        cy = ra["center_y"]
        radius = ra["radius"]
        
        # FILTER 1: Size bounds check
        if config.enforce_radius_bounds:
            if radius < config.min_radius_m or radius > config.max_radius_m:
                continue
        
        # Query points near the roundabout circumference
        # (within buffer distance of the ideal circle)
        search_radius = radius + config.circle_buffer_m
        nearby_indices = tree.query_ball_point([cx, cy], r=search_radius)
        
        if not nearby_indices:
            continue
        
        nearby_points = all_points[nearby_indices]
        nearby_traces = point_trace_ids[nearby_indices]
        nearby_headings = point_headings[nearby_indices]
        
        # Calculate distances from center
        distances = np.sqrt(
            (nearby_points[:, 0] - cx)**2 + 
            (nearby_points[:, 1] - cy)**2
        )
        
        # Points "on" the circle (within buffer of circumference)
        inner_bound = max(0, radius - config.circle_buffer_m)
        outer_bound = radius + config.circle_buffer_m
        on_circle_mask = (distances >= inner_bound) & (distances <= outer_bound)
        
        points_on_circle = nearby_points[on_circle_mask]
        traces_on_circle = nearby_traces[on_circle_mask]
        headings_on_circle = nearby_headings[on_circle_mask]
        
        # FILTER 2: Minimum points on circle
        if len(points_on_circle) < config.min_points_on_circle:
            continue
        
        # FILTER 3: Minimum distinct traces
        unique_traces = len(set(traces_on_circle.tolist()))
        if unique_traces < config.min_traces_near_circle:
            continue
        
        # FILTER 4: Entry direction diversity
        # Convert headings to entry directions (relative to center)
        entry_angles = np.arctan2(
            points_on_circle[:, 1] - cy,
            points_on_circle[:, 0] - cx
        )
        
        # Bucket into direction sectors
        bucket_rad = np.radians(config.direction_bucket_deg)
        direction_buckets = set()
        for angle in entry_angles:
            bucket = int((angle + np.pi) / bucket_rad) % int(2 * np.pi / bucket_rad)
            direction_buckets.add(bucket)
        
        if len(direction_buckets) < config.min_entry_directions:
            continue
        
        # FILTER 5: Arc coverage check
        # Check what fraction of the circle circumference has nearby traces
        angle_coverage = np.unique(np.round(entry_angles, 1))
        coverage_ratio = len(angle_coverage) / (2 * np.pi / 0.1)  # 0.1 rad buckets
        
        if coverage_ratio < config.min_arc_coverage_ratio:
            continue
        
        # FILTER 6: Circularity check - points should be at consistent distance
        if len(points_on_circle) >= 5:
            distances_on_circle = distances[on_circle_mask]
            radial_std = np.std(distances_on_circle)
            radial_mean = np.mean(distances_on_circle)
            if radial_mean > 0 and radial_std / radial_mean > config.max_radial_std_ratio:
                continue
        
        # Passed all filters
        validated.append(ra)
    
    # FILTER 7: Remove statistical radius outliers
    if config.filter_radius_outliers and len(validated) > 5:
        radii = np.array([r["radius"] for r in validated])
        mean_r = np.mean(radii)
        std_r = np.std(radii)
        if std_r > 0:
            validated = [
                r for r in validated
                if abs(r["radius"] - mean_r) <= config.radius_outlier_std * std_r
            ]
    
    # FILTER 8: Merge nearby duplicates
    if config.merge_distance_m > 0 and len(validated) > 1:
        validated = _merge_duplicate_roundabouts(validated, config.merge_distance_m)
    
    return validated


def _merge_duplicate_roundabouts(
    roundabouts: List[dict],
    merge_distance: float,
) -> List[dict]:
    """Merge roundabouts that are too close together (likely duplicates)."""
    if len(roundabouts) <= 1:
        return roundabouts
    
    centers = np.array([[r["center_x"], r["center_y"]] for r in roundabouts])
    tree = cKDTree(centers)
    
    visited = set()
    merged = []
    
    for i in range(len(roundabouts)):
        if i in visited:
            continue
        
        neighbors = tree.query_ball_point(centers[i], r=merge_distance)
        cluster = [j for j in neighbors if j not in visited]
        
        for j in cluster:
            visited.add(j)
        
        if cluster:
            # Average the cluster properties
            cluster_ras = [roundabouts[j] for j in cluster]
            avg_cx = np.mean([r["center_x"] for r in cluster_ras])
            avg_cy = np.mean([r["center_y"] for r in cluster_ras])
            avg_r = np.mean([r["radius"] for r in cluster_ras])
            
            # Use first roundabout as template, update center/radius
            merged_ra = roundabouts[cluster[0]].copy()
            merged_ra["center_x"] = float(avg_cx)
            merged_ra["center_y"] = float(avg_cy)
            merged_ra["radius"] = float(avg_r)
            
            # Regenerate geometry if present
            if "geometry" in merged_ra:
                n = 48
                angles = np.linspace(0, 2 * np.pi, n, endpoint=True)
                points = [
                    (avg_cx + avg_r * np.cos(a), avg_cy + avg_r * np.sin(a))
                    for a in angles
                ]
                merged_ra["geometry"] = LineString(points)
            
            merged.append(merged_ra)
    
    return merged
