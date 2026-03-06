"""Centerline generation package for HERE Geospatial Hackathon Problem 1."""

from .generation import (
    CenterlineConfig,
    generate_centerlines,
    generate_centerlines_with_algorithm,
    save_centerline_outputs,
)
from .evaluation import build_evaluation_context, evaluate_centerlines
from .algorithms import (
    BaseCenterlineAlgorithm,
    get_algorithm,
    list_algorithms,
    register_algorithm,
)
