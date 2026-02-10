"""Abstract base class for centerline generation algorithms.

All centerline generation algorithms must subclass ``BaseCenterlineAlgorithm``
and implement the ``generate`` method.  The base class defines a clear
contract so that algorithms are interchangeable inside the pipeline.

Typical usage::

    class MyAlgorithm(BaseCenterlineAlgorithm):
        name = "my-algorithm"
        description = "My custom algorithm."

        def add_cli_args(self, parser):
            parser.add_argument("--my-param", type=float, default=1.0)

        def configure(self, args):
            self.my_param = args.my_param

        def generate(self, traces, projected_crs, to_proj, to_wgs):
            ...
            return result_dict
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString


@dataclass
class AlgorithmConfig:
    """Base config that all algorithms share.

    Algorithm-specific subclasses can add their own fields while inheriting
    these common ones.  The ``from_dict`` classmethod makes it easy to
    construct from parsed CLI args or a JSON/YAML file.
    """

    sample_spacing_m: float = 8.0
    max_points_per_trace: int = 120
    smooth_iterations: int = 2
    min_centerline_length_m: float = 12.0
    vpd_base_weight: float = 1.2
    hpd_base_weight: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmConfig":
        """Create config from a dict, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


class BaseCenterlineAlgorithm(ABC):
    """Interface that every centerline generation algorithm must implement.

    Subclasses must set the ``name`` class attribute (used for registry
    look-up and the ``--algorithm`` CLI flag) and implement ``generate()``.

    Lifecycle
    ---------
    1. ``__init__()`` -- instantiate (can accept a config dataclass or nothing)
    2. ``add_cli_args(parser)`` -- *optional* hook to register algorithm-
       specific CLI arguments with an ``argparse.ArgumentParser``
    3. ``configure(args)`` -- *optional* hook called after argument parsing so
       the algorithm can store its resolved parameters
    4. ``generate(traces, projected_crs, to_proj, to_wgs)`` -- run the
       algorithm and return the standard result dict
    """

    # Subclasses MUST override these.
    name: str = ""
    description: str = ""

    def add_cli_args(self, parser) -> None:
        """Register algorithm-specific arguments on *parser*.

        Override this to expose tuning knobs on the command line.  The
        default implementation does nothing.
        """

    def configure(self, args) -> None:
        """Store resolved CLI arguments.

        Called after ``argparse.parse_args()`` so the algorithm can read its
        own parameters from the ``Namespace``.  Override as needed.
        """

    @abstractmethod
    def generate(
            self,
            traces: pd.DataFrame,
            projected_crs: CRS,
            to_proj: Transformer,
            to_wgs: Transformer,
    ) -> dict:
        """Run the centerline generation algorithm.

        Parameters
        ----------
        traces : pd.DataFrame
            Fused VPD + HPD traces with at least these columns:
            ``trace_id, source, geometry, day, hour, construction_percent,
            altitudes, crosswalk_types, traffic_signal_count,
            path_quality_score, sensor_quality_score``.
            ``geometry`` is a Shapely ``LineString`` in WGS84.
        projected_crs : pyproj.CRS
            Metric CRS inferred from the data (UTM zone).
        to_proj : pyproj.Transformer
            WGS84 -> projected CRS transformer.
        to_wgs : pyproj.Transformer
            Projected CRS -> WGS84 transformer.

        Returns
        -------
        dict with keys:
            ``projected_crs`` : pyproj.CRS
            ``nodes``         : pd.DataFrame  (node_id, x, y, lon, lat, heading, weight, point_count)
            ``edges``         : pd.DataFrame  (u, v, support, ..., geometry)
            ``centerlines``   : pd.DataFrame  (node_path, support, ..., geometry)
            ``trace_count``   : int
            ``sample_point_count`` : int
        """
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
