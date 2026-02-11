
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
import os

# Load the "Spaghetti" Skeleton
input_path = "./data/interim_skeleton_phase2.gpkg"
if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    exit()

print(f"Loading {input_path}...")
df = gpd.read_file(input_path)

# Spatial Join to find nearest neighbors
# We look for lines within 20 meters to see the spread
print("Buffering and joining...")
df_buffer = df.copy()
df_buffer.geometry = df.geometry.buffer(10)  # Expand search area
joined = gpd.sjoin(df_buffer, df, how="inner", predicate="intersects")

# Filter out self-matches
joined = joined[joined.index != joined.index_right]

# Calculate stats
print(f"Total traces: {len(df)}")
print(f"Total conflicting pairs found: {len(joined)}")
print("Diagnosis: If this number is high, your lines are just slightly too far apart.")
