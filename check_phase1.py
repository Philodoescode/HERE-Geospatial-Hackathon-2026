
import geopandas as gpd
import os

path = "data/interim_sample_phase1.gpkg"
if os.path.exists(path):
    df = gpd.read_file(path)
    print(f"File: {path}")
    print(f"Dimensions: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if not df.empty:
        print(f"Head:\n{df.head()}")
        # Check projection (assume raw is close to meters or convert)
        if df.crs and df.crs.is_geographic:
             print(f"CRS: {df.crs} (Geographic)")
        else:
             print(f"CRS: {df.crs}")
             print(f"Total Length: {df.geometry.length.sum() / 1000:.2f} km")
else:
    print(f"File not found: {path}")
