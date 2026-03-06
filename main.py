"""
HERE Geospatial Hackathon -- Main entry point.

Loads all three datasets (VPD, HPD, Nav Streets) and runs
Phase 2 data validation checks.
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    from src.loaders import load_hpd, load_nav_streets, load_vpd
    from src.preprocessing.validation import run_all_validations

    t0 = time.time()

    # ── Phase 1: Load datasets ────────────────────────────────────────
    logger.info("=== Phase 1: Loading datasets ===")

    logger.info("Loading Nav Streets...")
    nav_gdf = load_nav_streets()
    logger.info("Nav Streets: %d rows loaded.", len(nav_gdf))

    logger.info("Loading HPD (probe traces)...")
    hpd_gdf = load_hpd()
    logger.info("HPD: %d traces loaded.", len(hpd_gdf))

    logger.info("Loading VPD (vehicle path data)...")
    vpd_gdf = load_vpd()
    logger.info("VPD: %d rows loaded.", len(vpd_gdf))

    t_load = time.time() - t0
    logger.info("All datasets loaded in %.1f s.", t_load)

    # ── Phase 2: Validate & Sanity-Check ──────────────────────────────
    logger.info("=== Phase 2: Running data validations ===")
    results = run_all_validations(vpd_gdf, hpd_gdf, nav_gdf)

    # Exit with error code if any check FAILed
    n_fail = sum(1 for r in results if r.status == "FAIL")
    if n_fail:
        logger.error("%d validation check(s) FAILED.", n_fail)
        sys.exit(1)

    logger.info("All validations passed (%.1f s total).", time.time() - t0)


if __name__ == "__main__":
    main()
