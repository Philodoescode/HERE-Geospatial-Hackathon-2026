"""
Central configuration for the HERE Geospatial Hackathon pipeline.
All paths, constants, and CRS settings live here.
"""

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

VPD_CSV = DATA_DIR / "Kosovo_VPD" / "Kosovo_VPD.csv"
HPD_WEEK_1_CSV = DATA_DIR / "Kosovo_HPD" / "XKO_HPD_week_1.csv"
HPD_WEEK_2_CSV = DATA_DIR / "Kosovo_HPD" / "XKO_HPD_week_2.csv"
NAV_STREETS_GPKG = DATA_DIR / "Kosovo_s_nav_streets" / "Kosovo.gpkg"
NAV_STREETS_CSV = DATA_DIR / "Kosovo_s_nav_streets" / "Kosovo.csv"

# ── Output Paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Coordinate Reference System ──────────────────────────────────────────────
CRS = "EPSG:4326"

# ── Study Area Bounding Box (minx, miny, maxx, maxy) in EPSG:4326 ────────────
BBOX = (21.088588, 42.571255, 21.188588, 42.671255)

# ── VPD Loading ───────────────────────────────────────────────────────────────
VPD_CHUNK_SIZE = 10_000  # rows per chunk when reading the ~4GB VPD CSV

# Key VPD columns to keep (drop heavy sensor-type lists we don't need yet)
VPD_KEEP_COLUMNS = [
    "driveid",
    "lengthm",
    "fused",
    "signcount",
    "trafficsignalcount",
    "crosswalktypes",
    "constructionpercent",
    "nighttimepercent",
    "path",
    "altitudes",
    "pathqualityscore",
    "sensorqualityscore",
    "day",
]
