"""
Check if probe data and ground truth extend beyond the Kosovo bounding box.
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import wkt
from shapely.geometry import box, LineString
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from centerline.io_utils import load_bbox_from_txt, load_navstreet_csv

# Load bbox
bbox_file = Path("data/Kosovo_bounding_box.txt")
bbox = load_bbox_from_txt(bbox_file)
min_lon, min_lat, max_lon, max_lat = bbox
print(f"Kosovo Bounding Box: {bbox}")
print(f"  Longitude: [{min_lon}, {max_lon}]")
print(f"  Latitude:  [{min_lat}, {max_lat}]")
print()

bbox_poly = box(*bbox)

# Check VPD data (sample for speed)
print("=" * 60)
print("Checking VPD data (sampling 1000 rows)...")
print("=" * 60)
vpd_file = Path("data/Kosovo_VPD/Kosovo_VPD.parquet")
if vpd_file.exists():
    # Read only first 1000 rows for quick check
    vpd_df = pd.read_parquet(vpd_file, columns=['path'])
    sample_size = min(1000, len(vpd_df))
    vpd_df = vpd_df.head(sample_size)

    all_coords = []
    for idx, row in vpd_df.iterrows():
        try:
            geom = wkt.loads(str(row['path']))
            if isinstance(geom, LineString):
                all_coords.extend(list(geom.coords))
            elif hasattr(geom, 'geoms'):
                for g in geom.geoms:
                    if isinstance(g, LineString):
                        all_coords.extend(list(g.coords))
        except:
            continue

    if all_coords:
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]

        vpd_min_lon, vpd_max_lon = min(lons), max(lons)
        vpd_min_lat, vpd_max_lat = min(lats), max(lats)

        print(f"VPD Data Extent (from {sample_size} rows):")
        print(f"  Longitude: [{vpd_min_lon:.6f}, {vpd_max_lon:.6f}]")
        print(f"  Latitude:  [{vpd_min_lat:.6f}, {vpd_max_lat:.6f}]")
        print()

        extends_beyond = (
            vpd_min_lon < min_lon or vpd_max_lon > max_lon or
            vpd_min_lat < min_lat or vpd_max_lat > max_lat
        )

        if extends_beyond:
            print("⚠️  VPD DATA EXTENDS BEYOND BBOX!")
            if vpd_min_lon < min_lon:
                print(f"   - Min longitude {vpd_min_lon:.6f} < bbox {min_lon:.6f}")
            if vpd_max_lon > max_lon:
                print(f"   - Max longitude {vpd_max_lon:.6f} > bbox {max_lon:.6f}")
            if vpd_min_lat < min_lat:
                print(f"   - Min latitude {vpd_min_lat:.6f} < bbox {min_lat:.6f}")
            if vpd_max_lat > max_lat:
                print(f"   - Max latitude {vpd_max_lat:.6f} > bbox {max_lat:.6f}")
        else:
            print("✓ VPD data is within bbox")
        print()

# Check HPD data (sample for speed)
print("=" * 60)
print("Checking HPD data (sampling 5000 rows per file)...")
print("=" * 60)
hpd_files = [
    Path("data/Kosovo_HPD/XKO_HPD_week_1.parquet"),
    Path("data/Kosovo_HPD/XKO_HPD_week_2.parquet")
]

all_hpd_lons = []
all_hpd_lats = []

for hpd_file in hpd_files:
    if hpd_file.exists():
        print(f"Reading {hpd_file.name}...")
        hpd_df = pd.read_parquet(hpd_file, columns=['longitude', 'latitude'])
        sample_size = min(5000, len(hpd_df))
        hpd_df = hpd_df.head(sample_size)
        hpd_df['longitude'] = pd.to_numeric(hpd_df['longitude'], errors='coerce')
        hpd_df['latitude'] = pd.to_numeric(hpd_df['latitude'], errors='coerce')
        hpd_df = hpd_df.dropna()

        all_hpd_lons.extend(hpd_df['longitude'].tolist())
        all_hpd_lats.extend(hpd_df['latitude'].tolist())

if all_hpd_lons:
    hpd_min_lon, hpd_max_lon = min(all_hpd_lons), max(all_hpd_lons)
    hpd_min_lat, hpd_max_lat = min(all_hpd_lats), max(all_hpd_lats)
    
    print(f"HPD Data Extent:")
    print(f"  Longitude: [{hpd_min_lon:.6f}, {hpd_max_lon:.6f}]")
    print(f"  Latitude:  [{hpd_min_lat:.6f}, {hpd_max_lat:.6f}]")
    print()
    
    extends_beyond = (
        hpd_min_lon < min_lon or hpd_max_lon > max_lon or
        hpd_min_lat < min_lat or hpd_max_lat > max_lat
    )
    
    if extends_beyond:
        print("⚠️  HPD DATA EXTENDS BEYOND BBOX!")
        if hpd_min_lon < min_lon:
            print(f"   - Min longitude {hpd_min_lon:.6f} < bbox {min_lon:.6f}")
        if hpd_max_lon > max_lon:
            print(f"   - Max longitude {hpd_max_lon:.6f} > bbox {max_lon:.6f}")
        if hpd_min_lat < min_lat:
            print(f"   - Min latitude {hpd_min_lat:.6f} < bbox {min_lat:.6f}")
        if hpd_max_lat > max_lat:
            print(f"   - Max latitude {hpd_max_lat:.6f} > bbox {max_lat:.6f}")
    else:
        print("✓ HPD data is within bbox")
    print()

# Check ground truth (sample for speed)
print("=" * 60)
print("Checking Ground Truth (Navstreet) data (sampling 1000 rows)...")
print("=" * 60)

# Find navstreet file - try parquet first for speed
nav_parquet = Path("data/Kosovo's nav streets/nav_kosovo.parquet")
if nav_parquet.exists():
    print(f"Found navstreet file: {nav_parquet}")
    nav_df = pd.read_parquet(nav_parquet)
    sample_size = min(1000, len(nav_df))
    nav_df = nav_df.head(sample_size)

    # Parse geometries
    if 'geom' in nav_df.columns:
        geom_col = 'geom'
    elif 'geometry_wkt' in nav_df.columns:
        geom_col = 'geometry_wkt'
    else:
        geom_col = None

    all_nav_coords = []
    if geom_col:
        for geom_str in nav_df[geom_col]:
            try:
                geom = wkt.loads(str(geom_str))
                if isinstance(geom, LineString) and not geom.is_empty:
                    all_nav_coords.extend(list(geom.coords))
                elif hasattr(geom, 'geoms'):
                    for g in geom.geoms:
                        if isinstance(g, LineString):
                            all_nav_coords.extend(list(g.coords))
            except:
                continue
    
    if all_nav_coords:
        nav_lons = [c[0] for c in all_nav_coords]
        nav_lats = [c[1] for c in all_nav_coords]

        nav_min_lon, nav_max_lon = min(nav_lons), max(nav_lons)
        nav_min_lat, nav_max_lat = min(nav_lats), max(nav_lats)

        print(f"Ground Truth Extent (from {sample_size} rows):")
        print(f"  Longitude: [{nav_min_lon:.6f}, {nav_max_lon:.6f}]")
        print(f"  Latitude:  [{nav_min_lat:.6f}, {nav_max_lat:.6f}]")
        print()

        extends_beyond = (
            nav_min_lon < min_lon or nav_max_lon > max_lon or
            nav_min_lat < min_lat or nav_max_lat > max_lat
        )

        if extends_beyond:
            print("⚠️  GROUND TRUTH EXTENDS BEYOND BBOX!")
            if nav_min_lon < min_lon:
                print(f"   - Min longitude {nav_min_lon:.6f} < bbox {min_lon:.6f}")
            if nav_max_lon > max_lon:
                print(f"   - Max longitude {nav_max_lon:.6f} > bbox {max_lon:.6f}")
            if nav_min_lat < min_lat:
                print(f"   - Min latitude {nav_min_lat:.6f} < bbox {min_lat:.6f}")
            if nav_max_lat > max_lat:
                print(f"   - Max latitude {nav_max_lat:.6f} > bbox {max_lat:.6f}")
        else:
            print("✓ Ground truth is within bbox")
    else:
        print("Could not parse geometries from ground truth")
else:
    print("Ground truth navstreet file not found")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("The evaluation script has --apply-bbox option (default=True)")
print("which clips BOTH generated and ground truth to the same bbox.")
print("This ensures fair comparison.")

