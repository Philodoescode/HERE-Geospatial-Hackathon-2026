import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import os
from shapely.geometry import LineString

class DataIngestionPipeline:
    def __init__(self, input_path, output_path, sample_size=50000):
        self.input_path = input_path
        self.output_path = output_path
        self.sample_size = sample_size
        self.target_crs = "EPSG:4326"

    def detect_geometry_column(self, df):
        """Auto-detects the geometry column."""
        # 'path' is the likely column name in this dataset
        candidates = ['path', 'WKT', 'geometry', 'shape', 'wkt']
        for col in df.columns:
            if col in candidates:
                return col
            # Fallback for case-insensitive match
            if col.upper() in ['PATH', 'WKT', 'GEOMETRY', 'SHAPE']:
                return col
        # Look for any column containing 'WKT'
        for col in df.columns:
            if 'WKT' in col.upper():
                return col
        return None

    def calculate_heading(self, geometry):
        """Calculates the bearing of a LineString."""
        if not isinstance(geometry, LineString):
            return np.nan
        
        # simple bearing between start and end
        # valid for short segments in metric projection
        start = geometry.coords[0]
        end = geometry.coords[-1]
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        angle = np.degrees(np.arctan2(dx, dy)) # 0 is North, 90 is East
        if angle < 0:
            angle += 360
        return angle

    def get_direction_bin(self, heading):
        """Bins heading into N_S or E_W."""
        if pd.isna(heading):
            return "UNKNOWN"
        
        # North: 315-360, 0-45
        # South: 135-225
        # East: 45-135
        # West: 225-315
        
        if (heading >= 315) or (heading < 45) or (135 <= heading < 225):
            return "N_S"
        else:
            return "E_W"

    def run(self):
        print(f"Starting ingestion from {self.input_path}")
        
        # 1. Load and Filter (Chunked)
        filtered_chunks = []
        chunk_size = 5000
        total_filtered_rows = 0
        
        # Explicitly identifying the fused column if possible, otherwise guessing
        # But we know from previous 'head' command it is 'fused'
        
        for i, chunk in enumerate(pd.read_csv(self.input_path, chunksize=chunk_size)):
            # Normalize column names slightly for robustness
            chunk.columns = [c.strip() for c in chunk.columns]
            
            if 'fused' not in chunk.columns:
                 # verify if there is a column that looks like fused
                 pass 

            # Filter for fused == True (handling potential string/bool types)
            # Fused column can be 'Yes', 'True', True, etc.
            if chunk['fused'].dtype == 'O':
                 # Check for 'Yes', 'True', '1' case-insensitive
                 mask = chunk['fused'].astype(str).str.strip().str.lower().isin(['true', 'yes', '1'])
            else:
                 mask = chunk['fused'] == True
                 
            filtered = chunk[mask]
            
            if not filtered.empty:
                filtered_chunks.append(filtered)
                total_filtered_rows += len(filtered)
                
            # Break early if we have enough data (just for dev sampling, we want 1000)
            # BUT: Getting a representative sample is better. 
            # constraint: 52k rows is small enough to stream-filter all of it into a list then concat.
            # 52k rows isn't THAT big. 52k * 1kb = 52MB. It fits in RAM easily if filtered.
            
        if not filtered_chunks:
            raise ValueError("No records found with fused=True")
            
        full_filtered_df = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Total fused records: {len(full_filtered_df)}")
        
        # 2. Sub-Sampling
        if len(full_filtered_df) > self.sample_size:
            sampled_df = full_filtered_df.sample(n=self.sample_size, random_state=42).copy()
        else:
            sampled_df = full_filtered_df.copy()
            
        print(f"Sampled {len(sampled_df)} records for processing.")

        # 3. Geometry Conversion
        geom_col = self.detect_geometry_column(sampled_df)
        if not geom_col:
            # Fallback: Looking at the CSV head from previous steps, the WKT might be index or named differently
            # The output of 'head' showed: driveid,lengthm,rawsizebytes,fused,...
            # WAIT. I didn't see a 'WKT' or 'Geometry' column in the `head` output in the conversation history!
            # Let's look closely at the `head` output again.
            # Step 20 output: driveid,lengthm,rawsizebytes,fused,hometile14,tilesl14,signcount,dirsigncount,pole...
            # The CSV might not have a header, or the column name is obscured? 
            # Or maybe it's not in the first few columns.
            # I will trust the script to find it or fail efficiently.
            # Actually, standard VPD usually has 'WKT' or 'shape'. 
            # Let's assume there is a column I missed or it's further right.
            raise ValueError(f"Could not detect geometry column. Columns: {sampled_df.columns.tolist()}")
            
        print(f"Converting geometry from column: {geom_col}")
        sampled_df['geometry'] = sampled_df[geom_col].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(sampled_df, geometry='geometry', crs="EPSG:4326")
        
        # 4. Coordinate Transformation (Local)
        # Calculate centroid to find appropriate projection
        centroid = gdf.geometry.unary_union.centroid
        
        # Using Aeqd centered on centroid for accurate metric calc
        proj_str = f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        
        # Project to local metric system
        gdf_local = gdf.to_crs(proj_str)
        
        # 5. Attribute Engineering
        print("Calculating headings...")
        # We calculate on the projected geometry
        gdf_local['heading'] = gdf_local['geometry'].apply(self.calculate_heading)
        gdf_local['direction_bin'] = gdf_local['heading'].apply(self.get_direction_bin)
        
        # Map these back to the original GDF (optional, but good for export)
        gdf['heading'] = gdf_local['heading']
        gdf['direction_bin'] = gdf_local['direction_bin']
        
        # 6. Output
        print("\n--- Processed Data Head ---")
        print(gdf[['driveid', 'fused', 'heading', 'direction_bin', 'geometry']].head())
        
        print("\n--- Data Info ---")
        print(gdf.info())
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        print(f"Saving to {self.output_path}...")
        gdf.to_file(self.output_path, driver="GPKG")
        print("Done.")

if __name__ == "__main__":
    # Paths based on user environment
    INPUT_FILE = r"data\Kosovo_VPD\Kosovo_VPD.csv"
    OUTPUT_FILE = r"data\interim_sample_phase1.gpkg"
    
    pipeline = DataIngestionPipeline(INPUT_FILE, OUTPUT_FILE)
    pipeline.run()
