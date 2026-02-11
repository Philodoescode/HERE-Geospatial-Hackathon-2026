import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, polygonize, linemerge
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import warnings
from tqdm import tqdm


# Suppress warnings
warnings.filterwarnings('ignore')

class TopologyRefiner:
    def __init__(self, input_path, output_dir="data", visualization_dir="output"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.visualization_dir = visualization_dir
        
        self.gdf = None
        self.G = None
        self.planar_gdf = None
        self.final_gdf = None
        
    def load_data(self):
        """Loads and prepares the network data."""
        print("Loading data...")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
        self.gdf = gpd.read_file(self.input_path)
        
        # Ensure Metric CRS
        if self.gdf.crs.is_geographic:
            print("  Input is geographic. Projecting to local AEQD...")
            centroid = self.gdf.geometry.unary_union.centroid
            local_crs = f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            self.gdf = self.gdf.to_crs(local_crs)
            
        # Explode MultiLineStrings
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)
        print(f"  Loaded {len(self.gdf)} segments.")

    def planarize_network(self):
        """
        Splits lines at all intersections to create a planar graph (nodes at crossings).
        """
        print("Planarizing network (node generation)...")
        
        # unary_union merges all lines and splits them at intersections
        lines = self.gdf.geometry.tolist()
        merged = unary_union(lines)
        
        # If merged is a MultiLineString, we can iterate
        # linemerge might actully merge connected lines, we want to split them!
        # unary_union of lines -> MultiLineString with splits at intersections?
        # Actually unary_union on lines typically dissolves them.
        # To get a planar graph, we often use: unary_union -> distinct segments.
        
        if merged.geom_type == 'MultiLineString':
            planar_lines = list(merged.geoms)
        elif merged.geom_type == 'LineString':
            planar_lines = [merged]
        else:
            # Collection or other
            planar_lines = list(merged.geoms)
            
        self.planar_gdf = gpd.GeoDataFrame(geometry=planar_lines, crs=self.gdf.crs)
        print(f"  Planarization Complete. Segments: {len(self.gdf)} -> {len(self.planar_gdf)}")
        
    def filter_z_levels(self):
        """
        Placeholder for Z-level filtering. 
        If Z data is missing, we assume all intersections are valid (at-grade).
        """
        # Check for Z columns
        # In this dataset, we likely don't have per-vertex Z-levels in the geometry
        # unless 3D coords.
        
        has_z = self.planar_gdf.geometry.has_z.any()
        if not has_z:
            print("  No Z-level data detected. Assuming planar topology (all intersections valid).")
            return
            
        # If we had Z, we would iterate nodes and check connected edges.
        print("  Z-level data present (logic not fully implemented for this demo).")

    def smooth_geometry(self):
        """
        Applies B-Spline interpolation to smooth jagged lines.
        """
        print("Smoothing geometry (B-Splines)...")
        
        smoothed_geoms = []
        
        for geom in tqdm(self.planar_gdf.geometry, desc="Smoothing"):
            if geom.geom_type != 'LineString' or len(geom.coords) < 4:
                # Too short to smooth or not a line
                smoothed_geoms.append(geom)
                continue
                
            try:
                # Extract coords
                x, y = geom.xy
                
                # B-Spline Prep
                # s is smoothing factor. Adjust based on map scale. 
                # 4m buffer -> maybe s=5 or s=10?
                tck, u = splprep([x, y], s=5, per=0) 
                
                # Evaluate new points
                new_points = splev(np.linspace(0, 1, 20), tck) # 20 points per segment
                smoothed_geoms.append(LineString(list(zip(new_points[0], new_points[1]))))
            except Exception as e:
                # Fallback
                smoothed_geoms.append(geom)
                
        self.planar_gdf['geometry'] = smoothed_geoms
        print("  Smoothing Complete.")

    def prune_network(self):
        """
        Removes 'stubs' - short dead-end segments typical of tracing artifacts.
        """
        print("Pruning network (removing stubs)...")
        
        # Build Graph
        # Use momepy or simple logic? Simple logic:
        # 1. Create unique Node IDs based on coords
        # 2. Build adjacency
        
        # Let's use nx.read_shp logic essentially, but manual for speed/control
        
        # Round coords to avoid float precision issues
        self.planar_gdf['geometry'] = self.planar_gdf.geometry.apply(
            lambda x: LineString([
                (round(c[0], 3), round(c[1], 3)) for c in x.coords
            ])
        )
        
        # Extract edges
        edges = []
        for idx, row in self.planar_gdf.iterrows():
            geom = row.geometry
            u = geom.coords[0]
            v = geom.coords[-1]
            edges.append((u, v, {'length': geom.length, 'idx': idx}))
            
        self.G = nx.Graph(edges)
        
        print(f"  Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        
        # Find Stubs
        # Degree 1 nodes
        nodes_to_remove = []
        edges_to_remove = []
        
        stubs_found = 0
        
        for node in self.G.nodes():
            if self.G.degree(node) == 1:
                # Get the edge
                neighbor = list(self.G.neighbors(node))[0]
                edge_data = self.G.get_edge_data(node, neighbor)
                
                # Check length
                # Since graph might have multiple edges, edge_data might be a dict of dicts if MultiGraph
                # But here we used Graph (simple).
                # Wait, parallel edges? Planarization usually prevents them unless bubbles.
                
                length = edge_data.get('length', 0)
                
                if length < 15.0: # 15m threshold
                    # It's a stub
                    # Mark for removal
                    # We can't remove while iterating usually
                    edges_to_remove.append((node, neighbor))
                    stubs_found += 1
                    
        print(f"  Found {stubs_found} stubs (<15m dead ends). Removing...")
        self.G.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes?
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        
        print(f"  Pruned Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        
        # Reconstruct GeoDataFrame
        final_lines = []
        # We need to preserve geometry. 
        # The edges in G only stored 'idx' which points to planar_gdf.
        # But if we removed an edge, valid edges remain.
        # We can just filter planar_gdf by the remaining edge indices?
        # Careful: G edges (u,v) might not have 'idx' if we didn't add it correctly or if we mutated G.
        
        valid_indices = set()
        for u, v, data in self.G.edges(data=True):
            if 'idx' in data:
                valid_indices.add(data['idx'])
                
        self.final_gdf = self.planar_gdf.loc[list(valid_indices)].copy()
        
    def clean_geometries(self):
        """
        Fixes invalid geometries and drops empty ones.
        """
        print("Cleaning geometries...")
        original_count = len(self.final_gdf)
        
        # Remove empty
        self.final_gdf = self.final_gdf[~self.final_gdf.geometry.is_empty]
        
        # Make valid (buffer(0) trick)
        invalid_mask = ~self.final_gdf.geometry.is_valid
        if invalid_mask.sum() > 0:
            print(f"  Fixing {invalid_mask.sum()} invalid geometries...")
            self.final_gdf.loc[invalid_mask, 'geometry'] = \
                self.final_gdf.loc[invalid_mask, 'geometry'].buffer(0)
                
        # Remove empty again after buffer(0)
        self.final_gdf = self.final_gdf[~self.final_gdf.geometry.is_empty]
        
        # Remove Non-LineStrings (Points/Polygons from buffer(0) artifacts)
        self.final_gdf = self.final_gdf[self.final_gdf.geom_type.isin(['LineString', 'MultiLineString'])]
        
        print(f"  Cleaning Complete. {original_count} -> {len(self.final_gdf)} segments.")

    def export(self):
        self.clean_geometries()
        
        output_path = os.path.join(self.output_dir, "final_centerline_output.gpkg")
        print(f"Exporting to {output_path}...")
        self.final_gdf.to_file(output_path, driver="GPKG")
        
        # Save as EPSG:4326 as well for generic use
        output_4326 = output_path.replace(".gpkg", "_4326.gpkg")
        
        # Reprojecting potentially large dataset
        print(f"  Reprojecting to WGS84 for visualization ({output_4326})...")
        gdf_4326 = self.final_gdf.to_crs("EPSG:4326")
        gdf_4326.to_file(output_4326, driver="GPKG")
        print("  Export Complete.")

    def visualize(self):
        print("Generating comparison plot...")
        try:
            # Use Agg backend for non-interactive environments
            plt.switch_backend('Agg')
            
            if not os.path.exists(self.visualization_dir):
                os.makedirs(self.visualization_dir)
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Original (Phase 3) - Simplification for speed if needed
            print("  Plotting original network...")
            self.gdf.plot(ax=ax1, color='blue', linewidth=0.5, alpha=0.6)
            ax1.set_title("Input (Phase 3 Output)")
            ax1.axis('off')
            
            # Refined (Phase 4)
            print("  Plotting refined network...")
            self.final_gdf.plot(ax=ax2, color='black', linewidth=1.0)
            
            ax2.set_title("Refined (Smoothed & Pruned)")
            ax2.axis('off')
            
            plt.tight_layout()
            viz_path = os.path.join(self.visualization_dir, "phase4_topology_comparison.png")
            plt.savefig(viz_path, dpi=300)
            print(f"  Saved plot to {viz_path}")
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")

    def run(self):
        self.load_data()
        self.planarize_network()
        self.filter_z_levels()
        self.smooth_geometry()
        self.prune_network()
        self.export()
        self.visualize()
        print("Phase 4 Complete.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "final_network_phase3.gpkg")
    
    refiner = TopologyRefiner(INPUT_FILE)
    refiner.run()
