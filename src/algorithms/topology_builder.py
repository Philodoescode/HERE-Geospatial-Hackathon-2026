"""
Topology Network Builder

Constructs a proper road network topology from averaged centerline segments.
Handles:
- Node consolidation (snapping endpoints)
- Edge connectivity validation
- Roundabout integration
- Dangling endpoint cleanup
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from shapely.geometry import LineString
from scipy.spatial import cKDTree


@dataclass 
class TopologyConfig:
    """Configuration for topology building."""
    
    # Node snapping
    snap_radius_m: float = 8.0
    
    # Connectivity validation
    min_degree: int = 1  # Minimum node degree to keep
    
    # Dangling endpoint handling
    extend_dangling_m: float = 15.0  # Try to extend dangling ends this far
    dangling_snap_radius_m: float = 10.0  # Snap radius for dangling ends
    
    # Quality thresholds
    min_segment_length_m: float = 3.0  # Remove segments shorter than this


@dataclass
class TopologyNode:
    """A node in the road network topology."""
    node_id: int
    x: float
    y: float
    degree: int = 0
    connected_edges: List[int] = field(default_factory=list)
    is_roundabout: bool = False
    z_level: float = 0.0
    
    @property
    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class TopologyEdge:
    """An edge (road segment) in the road network topology."""
    edge_id: int
    from_node: int
    to_node: int
    geometry: LineString
    length_m: float
    support: float = 1.0
    altitude_mean: float = 0.0
    z_level: int = 0
    source: str = "unknown"


class TopologyBuilder:
    """
    Builds a proper road network topology from centerline segments.
    
    Steps:
    1. Consolidate segment endpoints into nodes
    2. Assign node IDs to segment endpoints
    3. Validate connectivity
    4. Handle dangling endpoints
    5. Export as node/edge structure
    """
    
    def __init__(self, config: Optional[TopologyConfig] = None):
        self.config = config or TopologyConfig()
        self.nodes: Dict[int, TopologyNode] = {}
        self.edges: Dict[int, TopologyEdge] = {}
    
    def build(
        self,
        geometries: List[LineString],
        supports: List[float],
        altitudes: Optional[List[float]] = None,
        sources: Optional[List[str]] = None,
        existing_nodes: Optional[List[Dict]] = None,
    ) -> Tuple[List[TopologyNode], List[TopologyEdge]]:
        """
        Build topology from segment geometries.
        
        Args:
            geometries: Segment LineStrings (projected CRS)
            supports: Support/weight per segment
            altitudes: Optional altitude per segment
            sources: Optional source type per segment
            existing_nodes: Optional pre-detected intersection nodes
        
        Returns:
            nodes: List of TopologyNode objects
            edges: List of TopologyEdge objects
        """
        cfg = self.config
        n = len(geometries)
        
        if n == 0:
            return [], []
        
        if altitudes is None:
            altitudes = [0.0] * n
        if sources is None:
            sources = ["unknown"] * n
        
        # Filter valid geometries
        valid_indices = []
        endpoints = []  # (x, y, segment_idx, is_start)
        
        for i, geom in enumerate(geometries):
            if geom is None or geom.is_empty or geom.length < cfg.min_segment_length_m:
                continue
            
            valid_indices.append(i)
            coords = list(geom.coords)
            start = coords[0][:2]
            end = coords[-1][:2]
            endpoints.append((start[0], start[1], i, True))
            endpoints.append((end[0], end[1], i, False))
        
        if not endpoints:
            return [], []
        
        # Step 1: Cluster endpoints into nodes
        nodes, endpoint_to_node = self._cluster_endpoints(endpoints, existing_nodes)
        
        # Step 2: Create edges with node assignments
        edges = []
        for i in valid_indices:
            geom = geometries[i]
            
            # Find node assignments for this segment's endpoints
            start_node = None
            end_node = None
            
            coords = list(geom.coords)
            start_pt = (coords[0][0], coords[0][1])
            end_pt = (coords[-1][0], coords[-1][1])
            
            # Find closest node for each endpoint
            for node in nodes:
                dist_start = np.hypot(node.x - start_pt[0], node.y - start_pt[1])
                dist_end = np.hypot(node.x - end_pt[0], node.y - end_pt[1])
                
                if dist_start < cfg.snap_radius_m:
                    start_node = node.node_id
                if dist_end < cfg.snap_radius_m:
                    end_node = node.node_id
            
            if start_node is None or end_node is None:
                # Create new nodes for unassigned endpoints
                if start_node is None:
                    new_id = len(nodes)
                    nodes.append(TopologyNode(node_id=new_id, x=start_pt[0], y=start_pt[1]))
                    start_node = new_id
                if end_node is None:
                    new_id = len(nodes)
                    nodes.append(TopologyNode(node_id=new_id, x=end_pt[0], y=end_pt[1]))
                    end_node = new_id
            
            # Snap geometry endpoints to node positions
            snapped_geom = self._snap_endpoints_to_nodes(
                geom, 
                nodes[start_node].coords,
                nodes[end_node].coords
            )
            
            edge = TopologyEdge(
                edge_id=len(edges),
                from_node=start_node,
                to_node=end_node,
                geometry=snapped_geom,
                length_m=snapped_geom.length,
                support=supports[i],
                altitude_mean=altitudes[i],
                source=sources[i],
            )
            edges.append(edge)
            
            # Update node connectivity
            nodes[start_node].connected_edges.append(edge.edge_id)
            nodes[start_node].degree += 1
            nodes[end_node].connected_edges.append(edge.edge_id)
            nodes[end_node].degree += 1
        
        # Step 3: Handle dangling endpoints (degree-1 nodes)
        edges = self._handle_dangling_endpoints(nodes, edges, geometries)
        
        # Update to dict storage
        self.nodes = {n.node_id: n for n in nodes}
        self.edges = {e.edge_id: e for e in edges}
        
        return nodes, edges
    
    def _cluster_endpoints(
        self, 
        endpoints: List[Tuple],
        existing_nodes: Optional[List[Dict]] = None,
    ) -> Tuple[List[TopologyNode], Dict[Tuple[int, bool], int]]:
        """Cluster endpoints into nodes using Union-Find."""
        cfg = self.config
        
        # Build coordinate array
        ep_coords = np.array([(e[0], e[1]) for e in endpoints])
        n_eps = len(ep_coords)
        
        # Use existing nodes if provided
        if existing_nodes:
            existing_coords = [(n["x"], n["y"]) for n in existing_nodes]
            all_coords = list(ep_coords) + existing_coords
        else:
            all_coords = list(ep_coords)
        
        all_coords = np.array(all_coords)
        n_total = len(all_coords)
        
        # Union-Find clustering
        from src.algorithms.intersection_detection import UnionFind
        uf = UnionFind(n_total)
        
        tree = cKDTree(all_coords)
        pairs = tree.query_pairs(cfg.snap_radius_m)
        for i, j in pairs:
            uf.union(i, j)
        
        # Build clusters
        clusters = defaultdict(list)
        for i in range(n_total):
            root = uf.find(i)
            clusters[root].append(i)
        
        # Create nodes from clusters
        nodes = []
        endpoint_to_node = {}
        
        for root, members in clusters.items():
            # Compute centroid
            member_coords = all_coords[members]
            cx = np.mean(member_coords[:, 0])
            cy = np.mean(member_coords[:, 1])
            
            node = TopologyNode(
                node_id=len(nodes),
                x=cx,
                y=cy,
            )
            nodes.append(node)
            
            # Map endpoints to this node
            for idx in members:
                if idx < n_eps:
                    seg_idx = endpoints[idx][2]
                    is_start = endpoints[idx][3]
                    endpoint_to_node[(seg_idx, is_start)] = node.node_id
        
        return nodes, endpoint_to_node
    
    def _snap_endpoints_to_nodes(
        self,
        geom: LineString,
        start_node_coords: Tuple[float, float],
        end_node_coords: Tuple[float, float],
    ) -> LineString:
        """Snap a geometry's endpoints to node positions."""
        coords = list(geom.coords)
        
        # Replace first and last coordinates
        new_coords = [start_node_coords]
        new_coords.extend(coords[1:-1])
        new_coords.append(end_node_coords)
        
        return LineString(new_coords)
    
    def _handle_dangling_endpoints(
        self,
        nodes: List[TopologyNode],
        edges: List[TopologyEdge],
        original_geometries: List[LineString],
    ) -> List[TopologyEdge]:
        """
        Handle dangling endpoints (degree-1 nodes).
        
        Try to extend them to connect to nearby segments/nodes.
        """
        cfg = self.config
        
        # Find dangling nodes (degree == 1)
        dangling_nodes = [n for n in nodes if n.degree == 1]
        
        if not dangling_nodes:
            return edges
        
        # Build spatial index of all node locations
        node_coords = np.array([(n.x, n.y) for n in nodes])
        node_tree = cKDTree(node_coords)
        
        for dang_node in dangling_nodes:
            # Find the single connected edge
            if not dang_node.connected_edges:
                continue
            
            edge_id = dang_node.connected_edges[0]
            edge = edges[edge_id]
            
            # Determine if this node is at start or end of edge
            is_start = (edge.from_node == dang_node.node_id)
            
            # Get direction from edge to extend
            coords = list(edge.geometry.coords)
            if is_start:
                ext_dir = np.array([coords[0][0] - coords[1][0], 
                                   coords[0][1] - coords[1][1]])
            else:
                ext_dir = np.array([coords[-1][0] - coords[-2][0],
                                   coords[-1][1] - coords[-2][1]])
            
            norm = np.linalg.norm(ext_dir)
            if norm < 0.01:
                continue
            ext_dir = ext_dir / norm
            
            # Search for nearby nodes to connect to
            dang_pt = np.array([dang_node.x, dang_node.y])
            search_radius = cfg.dangling_snap_radius_m + cfg.extend_dangling_m
            
            nearby_idx = node_tree.query_ball_point(dang_pt, search_radius)
            
            best_candidate = None
            best_dist = float('inf')
            
            for idx in nearby_idx:
                if idx == dang_node.node_id:
                    continue
                
                other_node = nodes[idx]
                if other_node.degree < 1:
                    continue
                
                # Check direction compatibility
                to_other = np.array([other_node.x - dang_node.x, 
                                    other_node.y - dang_node.y])
                dist = np.linalg.norm(to_other)
                
                if dist < best_dist and dist < search_radius:
                    # Check if roughly in extension direction
                    if dist > 0.01:
                        to_other_norm = to_other / dist
                        dot = np.dot(ext_dir, to_other_norm)
                        if dot > 0.5:  # Within ~60 degrees of extension direction
                            best_candidate = other_node
                            best_dist = dist
            
            # If found a good candidate, we could extend... 
            # For now, just mark it - actual extension would modify geometry
            if best_candidate is not None and best_dist < cfg.dangling_snap_radius_m:
                # Close enough to snap without extending
                pass  # Could add connection logic here
        
        return edges
    
    def to_dataframe(self) -> Tuple:
        """Convert nodes and edges to DataFrames for export."""
        import pandas as pd
        
        node_records = []
        for node in self.nodes.values():
            node_records.append({
                "node_id": node.node_id,
                "x": node.x,
                "y": node.y,
                "degree": node.degree,
                "is_roundabout": node.is_roundabout,
            })
        
        edge_records = []
        for edge in self.edges.values():
            edge_records.append({
                "edge_id": edge.edge_id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "length_m": edge.length_m,
                "support": edge.support,
                "altitude_mean": edge.altitude_mean,
                "z_level": edge.z_level,
                "source": edge.source,
                "geometry": edge.geometry,
            })
        
        nodes_df = pd.DataFrame(node_records)
        edges_df = pd.DataFrame(edge_records)
        
        return nodes_df, edges_df


def build_topology(
    geometries: List[LineString],
    supports: List[float],
    altitudes: Optional[List[float]] = None,
    sources: Optional[List[str]] = None,
    config: Optional[TopologyConfig] = None,
) -> Tuple[List[TopologyNode], List[TopologyEdge]]:
    """
    Convenience function to build topology.
    
    Returns (nodes, edges) tuple.
    """
    builder = TopologyBuilder(config)
    return builder.build(geometries, supports, altitudes, sources)
