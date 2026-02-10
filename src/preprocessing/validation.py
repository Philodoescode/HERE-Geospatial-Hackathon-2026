"""
Data Validation & Schema Checks  (Phase 2).

Verifies assumptions about loaded GeoDataFrames before any algorithm
consumes them.  Each check returns a ``ValidationResult`` and the
top-level ``run_all_validations`` prints a formatted summary report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import geopandas as gpd
from shapely.geometry import box

from src.config import BBOX, CRS

logger = logging.getLogger(__name__)

# ── Result container ──────────────────────────────────────────────────────────

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


@dataclass
class ValidationResult:
    """Outcome of a single validation check."""

    name: str
    status: str  # PASS | WARN | FAIL
    message: str
    details: dict = field(default_factory=dict)


# ── Individual checks ─────────────────────────────────────────────────────────


def check_crs(
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
) -> ValidationResult:
    """All GeoDataFrames must have CRS == EPSG:4326."""
    issues: list[str] = []
    for label, gdf in [("VPD", vpd_gdf), ("HPD", hpd_gdf), ("Nav", nav_gdf)]:
        if gdf.crs is None:
            issues.append(f"{label}: CRS is None")
        elif str(gdf.crs).upper() != CRS.upper():
            issues.append(f"{label}: CRS is {gdf.crs} (expected {CRS})")

    if issues:
        return ValidationResult(
            name="CRS Consistency",
            status=STATUS_FAIL,
            message="; ".join(issues),
        )
    return ValidationResult(
        name="CRS Consistency",
        status=STATUS_PASS,
        message=f"All GeoDataFrames have CRS = {CRS}.",
    )


def check_bounding_box(
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
    buffer_deg: float = 0.01,
) -> ValidationResult:
    """Geometries should lie within (or near) the study-area bounding box.

    A small buffer (default 0.01 deg ~1 km) is added around the bbox so
    edge geometries that barely spill over are not flagged.
    """
    minx, miny, maxx, maxy = BBOX
    buffered_bbox = box(
        minx - buffer_deg,
        miny - buffer_deg,
        maxx + buffer_deg,
        maxy + buffer_deg,
    )

    report: dict[str, dict] = {}
    total_outside = 0

    for label, gdf in [("VPD", vpd_gdf), ("HPD", hpd_gdf), ("Nav", nav_gdf)]:
        if gdf.empty:
            report[label] = {"total": 0, "outside": 0}
            continue
        outside_mask = ~gdf.geometry.intersects(buffered_bbox)
        n_outside = int(outside_mask.sum())
        report[label] = {"total": len(gdf), "outside": n_outside}
        total_outside += n_outside

    parts = [
        f"{lbl}: {info['outside']}/{info['total']} outside"
        for lbl, info in report.items()
    ]
    msg = "; ".join(parts)

    if total_outside == 0:
        return ValidationResult(
            name="Bounding Box",
            status=STATUS_PASS,
            message=f"All geometries within bbox (+{buffer_deg} deg buffer).",
            details=report,
        )
    return ValidationResult(
        name="Bounding Box",
        status=STATUS_WARN,
        message=f"Geometries outside bbox: {msg}",
        details=report,
    )


def check_fused_filter(vpd_gdf: gpd.GeoDataFrame) -> ValidationResult:
    """After loading, VPD should contain only ``fused == 'Yes'`` rows."""
    if "fused" not in vpd_gdf.columns:
        # The column may have been dropped after filtering — that's fine
        return ValidationResult(
            name="Fused Filter",
            status=STATUS_WARN,
            message="Column 'fused' not present in VPD GeoDataFrame "
            "(may have been dropped after filtering).",
        )

    non_fused_mask = vpd_gdf["fused"].astype(str).str.strip().str.lower() != "yes"
    n_bad = int(non_fused_mask.sum())

    if n_bad == 0:
        return ValidationResult(
            name="Fused Filter",
            status=STATUS_PASS,
            message=f"All {len(vpd_gdf)} VPD rows have fused == 'Yes'.",
        )
    return ValidationResult(
        name="Fused Filter",
        status=STATUS_FAIL,
        message=f"{n_bad} VPD rows have fused != 'Yes' (out of {len(vpd_gdf)}).",
        details={"non_fused_count": n_bad},
    )


def check_geometry_validity(
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
) -> ValidationResult:
    """No None, empty, or invalid (self-intersecting) geometries."""
    report: dict[str, dict] = {}
    total_issues = 0

    for label, gdf in [("VPD", vpd_gdf), ("HPD", hpd_gdf), ("Nav", nav_gdf)]:
        n_total = len(gdf)
        if n_total == 0:
            report[label] = {"total": 0, "none": 0, "empty": 0, "invalid": 0}
            continue

        n_none = int(gdf.geometry.isna().sum())
        # Only check emptiness on non-null geometries
        non_null = gdf.geometry.dropna()
        n_empty = int(non_null.is_empty.sum()) if len(non_null) > 0 else 0
        # Only check validity on non-null, non-empty geometries
        usable = non_null[~non_null.is_empty]
        n_invalid = int((~usable.is_valid).sum()) if len(usable) > 0 else 0

        info = {
            "total": n_total,
            "none": n_none,
            "empty": n_empty,
            "invalid": n_invalid,
        }
        report[label] = info
        total_issues += n_none + n_empty + n_invalid

    parts = []
    for lbl, info in report.items():
        problems = []
        if info["none"]:
            problems.append(f"{info['none']} None")
        if info["empty"]:
            problems.append(f"{info['empty']} empty")
        if info["invalid"]:
            problems.append(f"{info['invalid']} invalid")
        if problems:
            parts.append(f"{lbl}: {', '.join(problems)}")

    if total_issues == 0:
        return ValidationResult(
            name="Geometry Validity",
            status=STATUS_PASS,
            message="All geometries are non-null, non-empty, and valid.",
            details=report,
        )
    return ValidationResult(
        name="Geometry Validity",
        status=STATUS_WARN,
        message="Issues found: " + "; ".join(parts),
        details=report,
    )


def check_hpd_trace_integrity(hpd_gdf: gpd.GeoDataFrame) -> ValidationResult:
    """Every HPD trace should have >= 2 points (i.e. point_count >= 2)."""
    if "point_count" not in hpd_gdf.columns:
        return ValidationResult(
            name="HPD Trace Integrity",
            status=STATUS_WARN,
            message="Column 'point_count' not found — cannot verify trace integrity.",
        )

    bad_mask = hpd_gdf["point_count"] < 2
    n_bad = int(bad_mask.sum())

    if n_bad == 0:
        return ValidationResult(
            name="HPD Trace Integrity",
            status=STATUS_PASS,
            message=f"All {len(hpd_gdf)} HPD traces have >= 2 points.",
        )
    return ValidationResult(
        name="HPD Trace Integrity",
        status=STATUS_FAIL,
        message=f"{n_bad} traces have < 2 points (out of {len(hpd_gdf)}).",
        details={"under_2_points": n_bad},
    )


def check_missing_values(
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
) -> ValidationResult:
    """Report null counts for critical columns across datasets."""
    report: dict[str, dict[str, int]] = {}
    total_nulls = 0

    # ── VPD critical columns ──────────────────────────────────────────
    vpd_critical = ["geometry", "altitudes", "pathqualityscore", "sensorqualityscore"]
    vpd_nulls: dict[str, int] = {}
    for col in vpd_critical:
        if col == "geometry":
            n = int(vpd_gdf.geometry.isna().sum())
        elif col in vpd_gdf.columns:
            n = int(vpd_gdf[col].isna().sum())
        else:
            continue
        if n > 0:
            vpd_nulls[col] = n
            total_nulls += n
    if vpd_nulls:
        report["VPD"] = vpd_nulls

    # ── HPD critical columns ──────────────────────────────────────────
    hpd_critical = ["geometry", "avg_speed", "point_count"]
    hpd_nulls: dict[str, int] = {}
    for col in hpd_critical:
        if col == "geometry":
            n = int(hpd_gdf.geometry.isna().sum())
        elif col in hpd_gdf.columns:
            n = int(hpd_gdf[col].isna().sum())
        else:
            continue
        if n > 0:
            hpd_nulls[col] = n
            total_nulls += n
    if hpd_nulls:
        report["HPD"] = hpd_nulls

    # ── Nav critical columns ──────────────────────────────────────────
    nav_critical = ["geometry", "func_class", "road_link_id"]
    nav_nulls: dict[str, int] = {}
    for col in nav_critical:
        if col == "geometry":
            n = int(nav_gdf.geometry.isna().sum())
        elif col in nav_gdf.columns:
            n = int(nav_gdf[col].isna().sum())
        else:
            continue
        if n > 0:
            nav_nulls[col] = n
            total_nulls += n
    if nav_nulls:
        report["Nav"] = nav_nulls

    if total_nulls == 0:
        return ValidationResult(
            name="Missing Values",
            status=STATUS_PASS,
            message="No nulls in critical columns across all datasets.",
            details=report,
        )

    parts = []
    for ds, cols in report.items():
        col_strs = [f"{col}={cnt}" for col, cnt in cols.items()]
        parts.append(f"{ds}: {', '.join(col_strs)}")
    return ValidationResult(
        name="Missing Values",
        status=STATUS_WARN,
        message=f"Nulls found: {'; '.join(parts)}",
        details=report,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

_SEPARATOR = "-" * 72


def run_all_validations(
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
) -> list[ValidationResult]:
    """Run every Phase-2 validation check and print a summary report.

    Parameters
    ----------
    vpd_gdf, hpd_gdf, nav_gdf : gpd.GeoDataFrame
        The loaded datasets from Phase 1 loaders.

    Returns
    -------
    list[ValidationResult]
        One result per check, in execution order.
    """
    results: list[ValidationResult] = [
        check_crs(vpd_gdf, hpd_gdf, nav_gdf),
        check_bounding_box(vpd_gdf, hpd_gdf, nav_gdf),
        check_fused_filter(vpd_gdf),
        check_geometry_validity(vpd_gdf, hpd_gdf, nav_gdf),
        check_hpd_trace_integrity(hpd_gdf),
        check_missing_values(vpd_gdf, hpd_gdf, nav_gdf),
    ]

    _print_report(results, vpd_gdf, hpd_gdf, nav_gdf)
    return results


def _print_report(
    results: list[ValidationResult],
    vpd_gdf: gpd.GeoDataFrame,
    hpd_gdf: gpd.GeoDataFrame,
    nav_gdf: gpd.GeoDataFrame,
) -> None:
    """Pretty-print validation results to the console / log."""
    print()
    print(_SEPARATOR)
    print("  Phase 2 -- Data Validation Report")
    print(_SEPARATOR)

    # Dataset size summary
    print()
    print("  Dataset sizes:")
    print(f"    VPD : {len(vpd_gdf):>8,} rows, {len(vpd_gdf.columns):>3} columns")
    print(f"    HPD : {len(hpd_gdf):>8,} traces, {len(hpd_gdf.columns):>3} columns")
    print(f"    Nav : {len(nav_gdf):>8,} links, {len(nav_gdf.columns):>3} columns")
    print()

    # Table header
    header = f"  {'#':<4} {'Check':<25} {'Status':<8} {'Details'}"
    print(header)
    print(
        f"  {'---':<4} {'-------------------------':<25} {'------':<8} "
        f"{'----------------------------------------'}"
    )

    for i, r in enumerate(results, 1):
        print(f"  {i:<4} {r.name:<25} {r.status:<8} {r.message}")

    # Summary line
    n_pass = sum(1 for r in results if r.status == STATUS_PASS)
    n_warn = sum(1 for r in results if r.status == STATUS_WARN)
    n_fail = sum(1 for r in results if r.status == STATUS_FAIL)
    print()
    print(
        f"  Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL  "
        f"(out of {len(results)} checks)"
    )
    print(_SEPARATOR)
    print()
