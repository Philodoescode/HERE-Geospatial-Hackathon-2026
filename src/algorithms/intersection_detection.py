"""
Intersection Detection Module

Detects intersection points from centerline candidates using:
1. Endpoint clustering (junction nodes)
2. Mid-segment crossing detection (T-junctions, X-crossings)

Outputs intersection nodes with connectivity information for topology building.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from shapely.geometry import LineString, Point
from shapely.ops import substring
from scipy.spatial import cKDTree


@dataclass
class IntersectionNode:
    """Represents a detected intersection point."""
    node_id: int
    x: float
    y: float
    degree: int = 0  # Number of connected segments
    connected_segment_ids: List[int] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)  # Approach headings
    is_roundabout: bool = False
    z_level: Optional[float] = None
    
    @property
    def point(self) -> Point:
        return Point(self.x, self.y)
    
    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)


class UnionFind:
    """
    Union-Find (Disjoint Set Union) for clustering nearby points.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate bearing in degrees [0, 360) from point 1 to point 2."""
    dx = x2 - x1
    dy = y2 - y1
    return np.degrees(np.arctan2(dy, dx)) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest angle difference in degrees [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def get_line_endpoints_projected(geom: LineString) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Extract start and end points from a LineString."""
    coords = list(geom.coords)
    return coords[0][:2], coords[-1][:2]


def get_endpoint_heading(geom: LineString, at_start: bool = True, sample_dist: float = 5.0) -> float:
    """
    Get the heading at an endpoint of a LineString.
    
    Args:
        geom: LineString geometry
        at_start: If True, get heading at start; otherwise at end
        sample_dist: Distance along line to sample for heading calculation
    """
    coords = np.array(geom.coords)[:, :2]
    if len(coords) < 2:
        return 0.0
    
    if at_start:
        # Heading going INTO the line from start
        p0 = coords[0]
        # Find point ~sample_dist along
        cumlen = 0.0
        for i in range(1, len(coords)):
            seg_len = np.linalg.norm(coords[i] - coords[i-1])
            if cumlen + seg_len >= sample_dist or i == len(coords) - 1:
                p1 = coords[i]
                break
            cumlen += seg_len
        else:
            p1 = coords[-1]
        return bearing_from_xy(p0[0], p0[1], p1[0], p1[1])
    else:
        # Heading going INTO the line from end
        p0 = coords[-1]
        cumlen = 0.0
        for i in range(len(coords) - 2, -1, -1):
            seg_len = np.linalg.norm(coords[i+1] - coords[i])
            if cumlen + seg_len >= sample_dist or i == 0:
                p1 = coords[i]
                break
            cumlen += seg_len
        else:
            p1 = coords[0]
        return bearing_from_xy(p0[0], p0[1], p1[0], p1[1])


class IntersectionDetector:
    """
    Detects intersection points from centerline candidates.
    
    Algorithm:
    1. Extract all line endpoints
    2. Cluster nearby endpoints using Union-Find (snap_radius)
    3. Detect mid-segment crossings (lines that intersect away from endpoints)
    4. Merge crossing points with nearby endpoint clusters
    5. Return consolidated intersection nodes with connectivity info
    """
    
    def __init__(
        self,
        snap_radius_m: float = 8.0,
        min_degree: int = 1,
        crossing_buffer_m: float = 2.0,
        endpoint_exclusion_m: float = 5.0,
    ):
        """
        Args:
            snap_radius_m: Distance within which endpoints are clustered
            min_degree: Minimum connections to be considered an intersection
            crossing_buffer_m: Buffer for detecting mid-segment crossings
            endpoint_exclusion_m: Distance from endpoints to ignore crossings
        """
        self.snap_radius_m = snap_radius_m
        self.min_degree = min_degree
        self.crossing_buffer_m = crossing_buffer_m
        self.endpoint_exclusion_m = endpoint_exclusion_m
    
    def detect(
        self, 
        geometries: List[LineString],
        segment_ids: Optional[List[int]] = None,
    ) -> Tuple[List[IntersectionNode], Dict[int, Tuple[int, int]]]:
        """
        Detect all intersection points from centerline geometries.
        
        Args:
            geometries: List of LineString geometries (in projected CRS)
            segment_ids: Optional segment IDs (defaults to 0..n-1)
        
        Returns:
            intersections: List of IntersectionNode objects
            segment_nodes: Dict mapping segment_id -> (start_node_id, end_node_id)
        """
        if segment_ids is None:
            segment_ids = list(range(len(geometries)))
        
        n_segs = len(geometries)
        if n_segs == 0:
            return [], {}
        
        # Step 1: Extract all endpoints
        endpoints = []  # List of (x, y, segment_id, is_start)
        for i, (geom, seg_id) in enumerate(zip(geometries, segment_ids)):
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                continue
            start, end = get_line_endpoints_projected(geom)
            endpoints.append((start[0], start[1], seg_id, True))
            endpoints.append((end[0], end[1], seg_id, False))
        
        if len(endpoints) == 0:
            return [], {}
        
        # Step 2: Cluster endpoints using Union-Find
        ep_coords = np.array([(e[0], e[1]) for e in endpoints])
        n_eps = len(ep_coords)
        
        uf = UnionFind(n_eps)
        tree = cKDTree(ep_coords)
        
        # Find pairs within snap_radius
        pairs = tree.query_pairs(self.snap_radius_m)
        for i, j in pairs:
            uf.union(i, j)
        
        # Step 3: Detect mid-segment crossings
        crossing_points = self._detect_mid_segment_crossings(geometries, segment_ids)
        
        # Step 4: Assign crossing points to existing clusters or create new ones
        # First, build endpoint clusters
        cluster_members = defaultdict(list)
        for idx in range(n_eps):
            root = uf.find(idx)
            cluster_members[root].append(idx)
        
        # Compute cluster centroids
        cluster_centroids = {}
        for root, members in cluster_members.items():
            member_coords = ep_coords[members]
            cx = np.mean(member_coords[:, 0])
            cy = np.mean(member_coords[:, 1])
            cluster_centroids[root] = (cx, cy)
        
        # Merge crossing points with nearby clusters or add as new
        all_node_points = list(cluster_centroids.values())
        all_node_roots = list(cluster_centroids.keys())
        
        for cp in crossing_points:
            # Check if near any existing cluster
            merged = False
            for root, (cx, cy) in cluster_centroids.items():
                dist = np.hypot(cp[0] - cx, cp[1] - cy)
                if dist < self.snap_radius_m:
                    # Merge into existing cluster
                    merged = True
                    break
            
            if not merged:
                # Create new node for this crossing
                new_root = len(all_node_roots) + n_eps  # Unique ID
                all_node_roots.append(new_root)
                all_node_points.append((cp[0], cp[1]))
                cluster_centroids[new_root] = (cp[0], cp[1])
        
        # Step 5: Build intersection nodes
        # Map each endpoint to its cluster's node_id
        root_to_node_id = {root: i for i, root in enumerate(all_node_roots)}
        
        intersections = []
        segment_nodes = {}  # segment_id -> (start_node_id, end_node_id)
        
        # Track connections per node
        node_connections = defaultdict(list)  # node_id -> [(segment_id, heading)]
        
        for idx, (x, y, seg_id, is_start) in enumerate(endpoints):
            root = uf.find(idx)
            node_id = root_to_node_id[root]
            
            # Get geometry for heading calculation
            geom = geometries[segment_ids.index(seg_id)] if seg_id in segment_ids else None
            if geom is not None:
                heading = get_endpoint_heading(geom, at_start=is_start)
            else:
                heading = 0.0
            
            node_connections[node_id].append((seg_id, heading, is_start))
            
            # Update segment_nodes
            if seg_id not in segment_nodes:
                segment_nodes[seg_id] = [None, None]
            if is_start:
                segment_nodes[seg_id][0] = node_id
            else:
                segment_nodes[seg_id][1] = node_id
        
        # Convert segment_nodes to tuples
        segment_nodes = {k: tuple(v) for k, v in segment_nodes.items()}
        
        # Create IntersectionNode objects
        for node_id, (cx, cy) in enumerate(all_node_points):
            connections = node_connections.get(node_id, [])
            seg_ids = list(set(c[0] for c in connections))
            headings = [c[1] for c in connections]
            
            node = IntersectionNode(
                node_id=node_id,
                x=cx,
                y=cy,
                degree=len(seg_ids),
                connected_segment_ids=seg_ids,
                headings=headings,
            )
            intersections.append(node)
        
        # Filter by minimum degree if requested
        if self.min_degree > 1:
            intersections = [n for n in intersections if n.degree >= self.min_degree]
        
        return intersections, segment_nodes
    
    def _detect_mid_segment_crossings(
        self,
        geometries: List[LineString],
        segment_ids: List[int],
    ) -> List[Tuple[float, float, int, int]]:
        """
        Detect points where lines cross mid-segment (not at endpoints).
        
        Returns list of (x, y, seg_id_1, seg_id_2) tuples.
        """
        crossings = []
        n = len(geometries)
        
        # Build spatial index for efficiency
        from shapely import STRtree
        valid_geoms = []
        valid_ids = []
        for geom, seg_id in zip(geometries, segment_ids):
            if geom is not None and not geom.is_empty and geom.length > 0:
                valid_geoms.append(geom)
                valid_ids.append(seg_id)
        
        if len(valid_geoms) < 2:
            return crossings
        
        tree = STRtree(valid_geoms)
        
        # Query for intersections
        checked = set()
        for i, (geom_i, seg_id_i) in enumerate(zip(valid_geoms, valid_ids)):
            # Find candidates that might intersect
            candidates_idx = tree.query(geom_i)
            
            for j in candidates_idx:
                if i >= j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in checked:
                    continue
                checked.add(pair)
                
                geom_j = valid_geoms[j]
                seg_id_j = valid_ids[j]
                
                # Check for intersection
                if not geom_i.intersects(geom_j):
                    continue
                
                intersection = geom_i.intersection(geom_j)
                
                if intersection.is_empty:
                    continue
                
                # Extract crossing points
                crossing_pts = self._extract_crossing_points(
                    intersection, geom_i, geom_j, seg_id_i, seg_id_j
                )
                crossings.extend(crossing_pts)
        
        return crossings
    
    def _extract_crossing_points(
        self,
        intersection,
        geom_i: LineString,
        geom_j: LineString,
        seg_id_i: int,
        seg_id_j: int,
    ) -> List[Tuple[float, float, int, int]]:
        """Extract valid mid-segment crossing points from an intersection geometry."""
        points = []
        
        # Handle different intersection types
        if intersection.geom_type == 'Point':
            pts = [intersection]
        elif intersection.geom_type == 'MultiPoint':
            pts = list(intersection.geoms)
        elif intersection.geom_type == 'LineString':
            # Overlapping segments - use midpoint
            if intersection.length > 0:
                pts = [intersection.interpolate(0.5, normalized=True)]
            else:
                pts = []
        elif intersection.geom_type == 'GeometryCollection':
            pts = []
            for g in intersection.geoms:
                if g.geom_type == 'Point':
                    pts.append(g)
        else:
            pts = []
        
        # Filter: exclude points too close to endpoints
        for pt in pts:
            px, py = pt.x, pt.y
            
            # Distance to endpoints of geom_i
            coords_i = list(geom_i.coords)
            dist_to_start_i = np.hypot(px - coords_i[0][0], py - coords_i[0][1])
            dist_to_end_i = np.hypot(px - coords_i[-1][0], py - coords_i[-1][1])
            
            # Distance to endpoints of geom_j
            coords_j = list(geom_j.coords)
            dist_to_start_j = np.hypot(px - coords_j[0][0], py - coords_j[0][1])
            dist_to_end_j = np.hypot(px - coords_j[-1][0], py - coords_j[-1][1])
            
            min_dist = min(dist_to_start_i, dist_to_end_i, dist_to_start_j, dist_to_end_j)
            
            if min_dist > self.endpoint_exclusion_m:
                # This is a true mid-segment crossing
                points.append((px, py, seg_id_i, seg_id_j))
        
        return points


def detect_intersections(
    geometries: List[LineString],
    segment_ids: Optional[List[int]] = None,
    snap_radius_m: float = 8.0,
    min_degree: int = 1,
) -> Tuple[List[IntersectionNode], Dict[int, Tuple[int, int]]]:
    """
    Convenience function to detect intersections.
    
    Args:
        geometries: List of LineString geometries (projected CRS)
        segment_ids: Optional segment IDs
        snap_radius_m: Clustering radius for endpoints
        min_degree: Minimum degree to keep intersection
    
    Returns:
        intersections: List of IntersectionNode
        segment_nodes: Dict mapping segment_id -> (start_node, end_node)
    """
    detector = IntersectionDetector(
        snap_radius_m=snap_radius_m,
        min_degree=min_degree,
    )
    return detector.detect(geometries, segment_ids)


def split_line_at_point(
    geom: LineString,
    point: Point,
    tolerance_m: float = 1.0,
) -> List[LineString]:
    """
    Split a LineString at a point, returning two LineStrings.
    
    Returns original line if point is not on line or at endpoint.
    """
    if geom is None or geom.is_empty:
        return [geom]
    
    # Find closest point on line
    dist = geom.distance(point)
    if dist > tolerance_m:
        return [geom]
    
    # Project point onto line
    proj_dist = geom.project(point)
    
    # Check if at endpoint
    if proj_dist < tolerance_m or proj_dist > geom.length - tolerance_m:
        return [geom]
    
    # Split using substring
    try:
        part1 = substring(geom, 0, proj_dist)
        part2 = substring(geom, proj_dist, geom.length)
        
        result = []
        if part1 is not None and not part1.is_empty and part1.length > 0:
            result.append(part1)
        if part2 is not None and not part2.is_empty and part2.length > 0:
            result.append(part2)
        
        return result if result else [geom]
    except Exception:
        return [geom]


def split_lines_at_intersections(
    geometries: List[LineString],
    segment_ids: List[int],
    intersections: List[IntersectionNode],
    tolerance_m: float = 2.0,
) -> Tuple[List[LineString], List[int], List[Tuple[int, int]]]:
    """
    Split all lines at intersection points that fall mid-segment.
    
    Returns:
        new_geometries: Split line segments
        new_segment_ids: New segment IDs
        node_assignments: (start_node_id, end_node_id) for each new segment
    """
    # Build intersection point lookup
    int_coords = [(n.x, n.y, n.node_id) for n in intersections]
    if not int_coords:
        # No intersections to split at
        node_assignments = [(None, None) for _ in geometries]
        return list(geometries), list(segment_ids), node_assignments
    
    int_tree = cKDTree([(c[0], c[1]) for c in int_coords])
    
    new_geometries = []
    new_segment_ids = []
    node_assignments = []
    
    next_seg_id = max(segment_ids) + 1 if segment_ids else 0
    
    for geom, orig_seg_id in zip(geometries, segment_ids):
        if geom is None or geom.is_empty or geom.length == 0:
            continue
        
        geom_length = geom.length
        
        # Use spatial index to find nearby intersections
        # Query with buffer around line's bounding box
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        search_radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2 + tolerance_m * 2
        
        nearby_idx = int_tree.query_ball_point(center, search_radius)
        
        if not nearby_idx:
            new_geometries.append(geom)
            new_segment_ids.append(orig_seg_id)
            node_assignments.append((None, None))
            continue
        
        # Check each nearby intersection
        split_points = []  # (distance_along, node_id)
        
        for idx in nearby_idx:
            px, py, node_id = int_coords[idx]
            pt = Point(px, py)
            
            dist_to_line = geom.distance(pt)
            if dist_to_line > tolerance_m:
                continue
            
            # Project onto line
            proj_dist = geom.project(pt)
            
            # Skip if at endpoints
            if proj_dist < tolerance_m or proj_dist > geom_length - tolerance_m:
                continue
            
            split_points.append((proj_dist, node_id))
        
        if not split_points:
            # No splits needed
            new_geometries.append(geom)
            new_segment_ids.append(orig_seg_id)
            node_assignments.append((None, None))
            continue
        
        # Sort by distance along line
        split_points.sort(key=lambda x: x[0])
        
        # Remove duplicates (within tolerance)
        filtered_splits = []
        last_dist = -tolerance_m * 2
        for d, nid in split_points:
            if d - last_dist > tolerance_m:
                filtered_splits.append((d, nid))
                last_dist = d
        
        if not filtered_splits:
            new_geometries.append(geom)
            new_segment_ids.append(orig_seg_id)
            node_assignments.append((None, None))
            continue
        
        # Add start and end
        all_points = [(0.0, -1)] + filtered_splits + [(geom_length, -2)]
        
        # Create sub-segments
        for i in range(len(all_points) - 1):
            d1, node1 = all_points[i]
            d2, node2 = all_points[i + 1]
            
            if d2 - d1 < 0.5:
                continue
            
            try:
                sub_geom = substring(geom, d1, d2)
                if sub_geom is not None and not sub_geom.is_empty and sub_geom.length > 0.5:
                    new_geometries.append(sub_geom)
                    
                    if i == 0:
                        new_segment_ids.append(orig_seg_id)
                    else:
                        new_segment_ids.append(next_seg_id)
                        next_seg_id += 1
                    
                    # Assign nodes
                    start_node = node1 if node1 >= 0 else None
                    end_node = node2 if node2 >= 0 else None
                    node_assignments.append((start_node, end_node))
            except Exception:
                pass
    
    return new_geometries, new_segment_ids, node_assignments
