"""Check what columns exist in the pipeline outputs."""
import geopandas as gpd
import pandas as pd

paths = [
    ('Phase 3 output', 'E:/Coding projects/HERE-Geospatial-Hackathon/data/interim_probe_skeleton_phase3.gpkg'),
    ('Phase 4 output', 'E:/Coding projects/HERE-Geospatial-Hackathon/data/merged_network_phase4.gpkg'),
    ('Final output', 'E:/Coding projects/HERE-Geospatial-Hackathon/data/final_centerline_output_4326.gpkg'),
]

for name, path in paths:
    try:
        gdf = gpd.read_file(path)
        print(f'\n{name}: {len(gdf)} rows')
        print(f'  Columns: {list(gdf.columns)}')
        if 'source' in gdf.columns:
            src_counts = gdf["source"].value_counts().to_dict()
            print(f'  Sources: {src_counts}')
        if 'weighted_support' in gdf.columns:
            print(f'  Weighted support: min={gdf["weighted_support"].min():.2f}, max={gdf["weighted_support"].max():.2f}')
        if 'road_likeness_score' in gdf.columns:
            print(f'  Road likeness: min={gdf["road_likeness_score"].min():.3f}, max={gdf["road_likeness_score"].max():.3f}')
    except Exception as e:
        print(f'{name}: Error - {e}')
