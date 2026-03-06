# Preprocessing subpackage
from .validation import (
    ValidationResult,
    check_bounding_box,
    check_crs,
    check_fused_filter,
    check_geometry_validity,
    check_hpd_trace_integrity,
    check_missing_values,
    run_all_validations,
)
from .cleaning import (
    clip_to_bbox,
    validate_geometries,
    densify_line,
    densify_gdf,
    simplify_line,
    simplify_gdf,
    compute_heading,
    snap_to_grid,
)
