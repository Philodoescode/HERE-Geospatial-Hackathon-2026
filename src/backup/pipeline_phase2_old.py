import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import os
import sys

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("Warning: folium not found. Interactive map will not be generated.")

# Ensure output directory exists for plots
os.makedirs("output", exist_ok=True)

class Skeletonizer:
    def __init__(self, input_path, output_path, buffer_dist=4.0, heading_threshold=45.0):
        self.input_path = input_path
        self.output_path = output_path
        self.buffer_dist = buffer_dist
        self.heading_threshold = heading_threshold
        self.gdf = None
        self.crs_projected = None

    def load_and_project(self):
        print(f"Loading data from {self.input_path}...")
        self.gdf = gpd.read_file(self.input_path)
        
        # Determine projection based on centroid
        centroid = self.gdf.geometry.union_all().centroid
        # AEQD centered on data
        self.crs_projected = f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        
        print(f"Projecting to local CRS: {self.crs_projected}")
        self.gdf = self.gdf.to_crs(self.crs_projected)
        self.gdf['length'] = self.gdf.geometry.length
        
        # Ensure we have a unique ID to use as node names
        if 'driveid' not in self.gdf.columns:
            self.gdf['driveid'] = self.gdf.index.astype(str)
        
        # ensure index is reset and accessible
        self.gdf = self.gdf.reset_index(drop=True)
        self.gdf['node_id'] = self.gdf.index

    def _angle_diff(self, a1, a2):
        """Calculates minimal difference between two angles (0-360)."""
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)

    def cluster_traces(self):
        print("Clustering traces (Relaxed Angular Matching)...")
        
        # 1. Create buffers (Standard 12m)
        buffer_radius = 12.0
        print(f"  Creating {buffer_radius}m buffers...")
        self.gdf['buffer'] = self.gdf.geometry.buffer(buffer_radius)
        
        # 2. Spatial Join
        print("  Performing spatial join...")
        # Self-join based on intersection of buffers
        # We need a temporary GDF for the spatial join
        gdf_buffers = gpd.GeoDataFrame(
            self.gdf[['node_id', 'heading']], 
            geometry=self.gdf['buffer'], 
            crs=self.gdf.crs
        )
        
        joined = gpd.sjoin(gdf_buffers, gdf_buffers, how='inner', predicate='intersects')
        
        # Filter out self-loops and duplicate pairs (keep i < j)
        joined = joined[joined['node_id_left'] < joined['node_id_right']]
        
        print(f"  Found {len(joined)} spatial overlaps. Filtering by heading diff...")
        
        # 3. Heading Difference Filter
        # Calculate angular difference between left and right traces
        def angle_diff(h1, h2):
            d = abs(h1 - h2)
            return min(d, 360 - d)
            
        # Vectorized calculation would be faster, but let's use apply for clarity if dataset isn't huge
        # Actually, let's use numpy for speed
        h1 = joined['heading_left'].values
        h2 = joined['heading_right'].values
        diffs = np.abs(h1 - h2)
        diffs = np.minimum(diffs, 360 - diffs)
        
        # Filter: traces must be within 45 degrees of each other
        valid_mask = diffs < 45.0
        valid_pairs = joined[valid_mask]
        
        print(f"  {len(valid_pairs)} pairs remained after heading filter (< 45 deg).")
        
        # 4. Build Graph
        print("  Building graph...")
        G = nx.Graph()
        G.add_nodes_from(self.gdf['node_id'])
        G.add_edges_from(zip(valid_pairs['node_id_left'], valid_pairs['node_id_right']))
        
        # 5. Connected Components
        clusters = list(nx.connected_components(G))
        print(f"  Identified {len(clusters)} clusters from {len(self.gdf)} traces.")
        
        # Assign cluster IDs
        cluster_map = {}
        for cid, nodes in enumerate(clusters):
            for node in nodes:
                cluster_map[node] = cid
        self.gdf['cluster_id'] = self.gdf['node_id'].map(cluster_map)

    def generate_skeleton(self):
        print("Generating skeleton (Centroid Selection)...")
        
        skeleton_indices = []
        
        grouped = self.gdf.groupby('cluster_id')
        
        for cid, group in grouped:
            if len(group) == 1:
                skeleton_indices.append(group.index[0])
                continue
            
            # Calculate geometric centroid of ALL lines in the cluster combined
            # (Conceptually: MultiLineString(all_lines).centroid)
            combined_geom = group.geometry.unary_union
            cluster_centroid = combined_geom.centroid
            
            # Find the single line in that cluster that is spatially closest to this centroid
            # We calculate distance from each line to the cluster centroid
            distances = group.geometry.distance(cluster_centroid)
            best_idx = distances.idxmin()
            
            skeleton_indices.append(best_idx)
            
        self.skeleton_gdf = self.gdf.loc[skeleton_indices].copy()
        print(f"  Skeleton contains {len(self.skeleton_gdf)} segments.")

    def verify_and_plot(self):
        print("Verifying and Plotting...")
        print(f"  Skeleton has {len(self.skeleton_gdf)} segments.")
        print("  Skipping plotting due to environment issues.")


    def export(self):
        print(f"Exporting skeleton to {self.output_path}...")
        columns_to_keep = ['driveid', 'length', 'heading', 'direction_bin', 'cluster_id', 'geometry']
        # Filter columns if they exist
        out_cols = [c for c in columns_to_keep if c in self.skeleton_gdf.columns]
        
        # Ensure CRS is 4326
        out_gdf = self.skeleton_gdf[out_cols].to_crs("EPSG:4326")
        out_gdf.to_file(self.output_path, driver="GPKG")
        print("Done.")

    def run(self):
        self.load_and_project()
        self.cluster_traces()
        self.generate_skeleton()
        self.verify_and_plot()
        self.export()

if __name__ == "__main__":
    INPUT_FILE = r"data/interim_sample_phase1.gpkg"
    OUTPUT_FILE = r"data/interim_skeleton_phase2.gpkg"
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Run Phase 1 first.")
        sys.exit(1)
        
    skeletonizer = Skeletonizer(INPUT_FILE, OUTPUT_FILE)
    skeletonizer.run()
