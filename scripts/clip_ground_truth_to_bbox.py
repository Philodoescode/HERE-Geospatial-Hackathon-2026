from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from centerline.io_utils import clip_line_geometries_to_bbox, load_bbox_from_txt


# Edit these paths only if your files move.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_GPKG = PROJECT_ROOT / "data" / "Kosovo's nav streets" / "nav_kosovo.gpkg"
GT_LAYER = "Kosovo22"
BBOX_FILE = PROJECT_ROOT / "data" / "Kosovo_bounding_box.txt"
OUT_GPKG = PROJECT_ROOT / "outputs" / "ground_truth" / "nav_kosovo_clipped_bbox.gpkg"
OUT_LAYER = "navstreet"


def main() -> None:
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(GT_GPKG, layer=GT_LAYER)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[gdf.geometry.notnull()].copy()

    bbox = load_bbox_from_txt(BBOX_FILE)
    clipped = clip_line_geometries_to_bbox(gdf, bbox_wgs84=bbox, geometry_col="geometry")
    clipped = gpd.GeoDataFrame(clipped, geometry="geometry", crs="EPSG:4326")

    # Keep schema GPKG-friendly for object columns.
    write_gdf = clipped.drop(columns=["geom"], errors="ignore").copy()
    for col in write_gdf.columns:
        if col == "geometry":
            continue
        if write_gdf[col].dtype == "object":
            write_gdf[col] = write_gdf[col].astype(str)

    write_gdf.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")

    print(f"input_rows={len(gdf)}")
    print(f"clipped_rows={len(write_gdf)}")
    print(f"out={OUT_GPKG}")
    print(f"layer={OUT_LAYER}")


if __name__ == "__main__":
    main()
