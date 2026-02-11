"""
Quick check of data extents vs bbox using pyarrow for efficient reading.
"""
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from centerline.io_utils import load_bbox_from_txt

# Load bbox
bbox_file = Path("data/Kosovo_bounding_box.txt")
bbox = load_bbox_from_txt(bbox_file)
min_lon, min_lat, max_lon, max_lat = bbox
print(f"Kosovo Bounding Box: {bbox}")
print(f"  Longitude: [{min_lon}, {max_lon}]")
print(f"  Latitude:  [{min_lat}, {max_lat}]")
print()

# Check HPD data - this is point data so easier to check
print("=" * 60)
print("Checking HPD data...")
print("=" * 60)
hpd_files = [
    Path("data/Kosovo_HPD/XKO_HPD_week_1.parquet"),
    Path("data/Kosovo_HPD/XKO_HPD_week_2.parquet")
]

for hpd_file in hpd_files:
    if hpd_file.exists():
        print(f"\nReading {hpd_file.name}...")
        # Use pyarrow to read in batches
        parquet_file = pq.ParquetFile(hpd_file)
        
        min_lon_data = float('inf')
        max_lon_data = float('-inf')
        min_lat_data = float('inf')
        max_lat_data = float('-inf')
        
        for batch in parquet_file.iter_batches(batch_size=10000, columns=['longitude', 'latitude']):
            df_batch = batch.to_pandas()
            df_batch['longitude'] = pd.to_numeric(df_batch['longitude'], errors='coerce')
            df_batch['latitude'] = pd.to_numeric(df_batch['latitude'], errors='coerce')
            df_batch = df_batch.dropna()
            
            if len(df_batch) > 0:
                min_lon_data = min(min_lon_data, df_batch['longitude'].min())
                max_lon_data = max(max_lon_data, df_batch['longitude'].max())
                min_lat_data = min(min_lat_data, df_batch['latitude'].min())
                max_lat_data = max(max_lat_data, df_batch['latitude'].max())
        
        print(f"  Longitude: [{min_lon_data:.6f}, {max_lon_data:.6f}]")
        print(f"  Latitude:  [{min_lat_data:.6f}, {max_lat_data:.6f}]")
        
        extends_beyond = (
            min_lon_data < min_lon or max_lon_data > max_lon or
            min_lat_data < min_lat or max_lat_data > max_lat
        )
        
        if extends_beyond:
            print(f"  ⚠️  {hpd_file.name} EXTENDS BEYOND BBOX!")
            if min_lon_data < min_lon:
                print(f"     - Min longitude {min_lon_data:.6f} < bbox {min_lon:.6f} (diff: {min_lon - min_lon_data:.6f})")
            if max_lon_data > max_lon:
                print(f"     - Max longitude {max_lon_data:.6f} > bbox {max_lon:.6f} (diff: {max_lon_data - max_lon:.6f})")
            if min_lat_data < min_lat:
                print(f"     - Min latitude {min_lat_data:.6f} < bbox {min_lat:.6f} (diff: {min_lat - min_lat_data:.6f})")
            if max_lat_data > max_lat:
                print(f"     - Max latitude {max_lat_data:.6f} > bbox {max_lat:.6f} (diff: {max_lat_data - max_lat:.6f})")
        else:
            print(f"  ✓ {hpd_file.name} is within bbox")

print()
print("=" * 60)
print("SUMMARY & SOLUTION")
print("=" * 60)
print()
print("The evaluation script (scripts/evaluate_centerlines.py) has the")
print("--apply-bbox option which is TRUE by default.")
print()
print("When --apply-bbox is enabled, the evaluation script clips BOTH:")
print("  1. Generated centerlines")
print("  2. Ground truth navstreet data")
print("to the SAME bounding box before computing metrics.")
print()
print("This ensures a fair comparison even if the probe data or ground")
print("truth extends beyond the bbox used during centerline generation.")
print()
print("RECOMMENDED EVALUATION COMMAND:")
print("-" * 60)
print()
print('python scripts/evaluate_centerlines.py \\')
print('  --generated outputs/kharita_full_tuned_no_deepmg/kharita_full_tuned_no_deepmg.gpkg \\')
print('  --ground-truth "data/Kosovo\'s nav streets/nav_kosovo.parquet" \\')
print('  --apply-bbox \\')
print('  --bbox-file data/Kosovo_bounding_box.txt \\')
print('  --buffer-m 15.0 \\')
print('  --out outputs/evaluation/kharita_full_tuned_no_deepmg_metrics.json')
print()
print("This will ensure both generated and ground truth are clipped to")
print("the same bbox [21.088588, 42.571255, 21.188588, 42.671255]")
print("before computing precision, recall, and F1 scores.")

