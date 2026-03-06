from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from centerline.io_utils import (
    clip_line_geometries_to_bbox,
    infer_local_projected_crs,
    load_bbox_from_txt,
    load_navstreet_csv,
)


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
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=Path("data/Kosovo_bounding_box.txt"),
        help="Optional WGS84 bbox text file used to clip ground truth.",
    )
    parser.add_argument(
        "--apply-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip navstreet geometries to bbox before export.",
    )
    args = parser.parse_args()

    nav_csv = args.nav_csv or find_default_nav_csv(Path("data"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_navstreet_csv(nav_csv)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()].copy()

    if args.apply_bbox:
        bbox = load_bbox_from_txt(args.bbox_file)
        gdf = clip_line_geometries_to_bbox(gdf, bbox_wgs84=bbox, geometry_col="geometry")
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

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
    parquet_path = args.out_dir / f"{args.stem}.parquet"
    out_csv = gdf.copy()
    out_csv["geometry_wkt"] = out_csv.geometry.astype(str)
    out_csv = out_csv.drop(columns=["geometry"])
    out_csv.to_csv(csv_path, index=False)
    out_csv.to_parquet(parquet_path, index=False)

    # Build node-layer proxy from reference/non-reference node ids.
    node_rows = []
    for r in gdf.itertuples(index=False):
        if r.geometry is None or r.geometry.is_empty:
            continue
        coords = list(r.geometry.coords)
        if len(coords) < 2:
            continue
        s_lon, s_lat = float(coords[0][0]), float(coords[0][1])
        e_lon, e_lat = float(coords[-1][0]), float(coords[-1][1])
        node_rows.append({"node_id": str(r.ref_node_pvid), "lon": s_lon, "lat": s_lat})
        node_rows.append({"node_id": str(r.nref_node_pvid), "lon": e_lon, "lat": e_lat})

    nodes_path = None
    nodes_parquet_path = None
    if node_rows:
        nd = pd.DataFrame(node_rows).groupby("node_id", as_index=False).agg({"lon": "mean", "lat": "mean"})
        nodes_gdf = gpd.GeoDataFrame(nd, geometry=gpd.points_from_xy(nd["lon"], nd["lat"]), crs="EPSG:4326")
        nodes_path = args.out_dir / f"{args.stem}_nodes.gpkg"
        nodes_gdf.to_file(nodes_path, layer="navstreet_nodes", driver="GPKG")
        nodes_parquet_path = args.out_dir / f"{args.stem}_nodes.parquet"
        nodes_out = nodes_gdf.copy()
        nodes_out["geometry_wkt"] = nodes_out.geometry.astype(str)
        nodes_out = nodes_out.drop(columns=["geometry"])
        nodes_out.to_parquet(nodes_parquet_path, index=False)

    # Save projected copy to simplify metric-space evaluation.
    proj_crs = infer_local_projected_crs(list(gdf.geometry))
    gdf_proj = gdf.to_crs(proj_crs)
    gpkg_proj_path = args.out_dir / f"{args.stem}_projected.gpkg"
    parquet_proj_path = args.out_dir / f"{args.stem}_projected.parquet"
    write_proj = gdf_proj.drop(columns=["geom"], errors="ignore").copy()
    for c in write_proj.columns:
        if c == "geometry":
            continue
        if write_proj[c].dtype == "object":
            write_proj[c] = write_proj[c].map(lambda v: json.dumps(v) if isinstance(v, (list, dict, set)) else v)
    write_proj.to_file(gpkg_proj_path, layer="navstreet_projected", driver="GPKG")
    out_proj = gdf_proj.copy()
    out_proj["geometry_wkt"] = out_proj.geometry.astype(str)
    out_proj = out_proj.drop(columns=["geometry"])
    out_proj.to_parquet(parquet_proj_path, index=False)

    summary = {
        "rows": int(len(gdf)),
        "projected_crs": str(proj_crs),
        "apply_bbox": bool(args.apply_bbox),
        "bbox_file": str(args.bbox_file) if args.apply_bbox else None,
        "files": {
            "gpkg_wgs84": str(gpkg_path),
            "gpkg_projected": str(gpkg_proj_path),
            "csv": str(csv_path),
            "parquet_wgs84": str(parquet_path),
            "parquet_projected": str(parquet_proj_path),
            "nodes": str(nodes_path) if nodes_path is not None else None,
            "nodes_parquet": str(nodes_parquet_path) if nodes_parquet_path is not None else None,
        },
    }
    summary_path = args.out_dir / f"{args.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved ground truth GPKG: {gpkg_path}")
    print(f"Saved projected GPKG   : {gpkg_proj_path}")
    print(f"Saved ground truth CSV : {csv_path}")
    print(f"Saved ground truth PQT : {parquet_path}")
    if nodes_path is not None:
        print(f"Saved node layer       : {nodes_path}")
    if nodes_parquet_path is not None:
        print(f"Saved node parquet     : {nodes_parquet_path}")
    print(f"Saved summary          : {summary_path}")
    print(f"Rows: {len(gdf)}")


if __name__ == "__main__":
    main()
