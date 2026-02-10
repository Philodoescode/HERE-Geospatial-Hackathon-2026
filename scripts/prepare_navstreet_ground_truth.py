from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.io_utils import load_navstreet_csv


def find_default_nav_csv(data_root: Path) -> Path:
    matches = list(data_root.glob("*nav streets*/Kosovo.csv"))
    if not matches:
        raise FileNotFoundError("Could not auto-locate Kosovo navstreet CSV under data/.")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HERE navstreet ground truth from Kosovo CSV.")
    parser.add_argument("--nav-csv", type=Path, default=None, help="Path to navstreet CSV (with geom WKT column).")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ground_truth"), help="Output directory.")
    parser.add_argument("--stem", default="kosovo_navstreet_ground_truth", help="Output file stem.")
    args = parser.parse_args()

    nav_csv = args.nav_csv or find_default_nav_csv(Path("data"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_navstreet_csv(nav_csv)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()].copy()

    # Normalize direction field labels.
    if "dir_travel" in gdf.columns:
        gdf["dir_travel"] = gdf["dir_travel"].astype(str).str.upper().str[0]

    write_gdf = gdf.drop(columns=["geom"], errors="ignore").copy()

    gpkg_path = args.out_dir / f"{args.stem}.gpkg"
    for c in write_gdf.columns:
        if c == "geometry":
            continue
        if write_gdf[c].dtype == "object":
            write_gdf[c] = write_gdf[c].map(lambda v: json.dumps(v) if isinstance(v, (list, dict, set)) else v)
    write_gdf.to_file(gpkg_path, layer="navstreet", driver="GPKG")

    csv_path = args.out_dir / f"{args.stem}.csv"
    out_csv = gdf.copy()
    out_csv["geometry_wkt"] = out_csv.geometry.astype(str)
    out_csv = out_csv.drop(columns=["geometry"])
    out_csv.to_csv(csv_path, index=False)

    print(f"Saved ground truth GPKG: {gpkg_path}")
    print(f"Saved ground truth CSV : {csv_path}")
    print(f"Rows: {len(gdf)}")


if __name__ == "__main__":
    main()
