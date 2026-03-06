"""Algorithm registry for centerline generation.

Provides ``register_algorithm()``, ``get_algorithm()``, and
``list_algorithms()`` so that new algorithms can be added without touching
the core pipeline code.

Built-in algorithms are registered at import time.  Third-party algorithms
can register themselves via ``register_algorithm()`` after importing this
module.

Usage::

    from centerline.algorithms import get_algorithm, list_algorithms

    # Get an algorithm by name
    algo = get_algorithm("kharita")

    # List all registered algorithms
    for name, desc in list_algorithms():
        print(f"{name}: {desc}")
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Type

from .base import AlgorithmConfig, BaseCenterlineAlgorithm

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaseCenterlineAlgorithm]] = {}


def register_algorithm(
        cls: Type[BaseCenterlineAlgorithm],
) -> Type[BaseCenterlineAlgorithm]:
    """Register an algorithm class.  Can be used as a decorator::

    @register_algorithm
    class MyAlgorithm(BaseCenterlineAlgorithm):
        name = "my-algo"
        ...
    """
    if not cls.name:
        raise ValueError(
            f"Algorithm class {cls.__name__} must set a non-empty 'name' attribute."
        )
    _REGISTRY[cls.name] = cls
    return cls


def get_algorithm(name: str, **kwargs) -> BaseCenterlineAlgorithm:
    """Instantiate a registered algorithm by name.

    Parameters
    ----------
    name : str
        The algorithm ``name`` attribute (e.g. ``"kharita"``).
    **kwargs
        Forwarded to the algorithm constructor.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown algorithm {name!r}.  Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_algorithms() -> List[Tuple[str, str]]:
    """Return a list of ``(name, description)`` for all registered algorithms."""
    return [(name, cls.description) for name, cls in sorted(_REGISTRY.items())]


# ---------------------------------------------------------------------------
# Register built-in algorithms
# ---------------------------------------------------------------------------

from .kharita import KharitaAlgorithm  # noqa: E402
from .roadster import RoadsterAlgorithm  # noqa: E402

register_algorithm(KharitaAlgorithm)
register_algorithm(RoadsterAlgorithm)

# Convenience re-exports
__all__ = [
    "AlgorithmConfig",
    "BaseCenterlineAlgorithm",
    "KharitaAlgorithm",
    "RoadsterAlgorithm",
    "get_algorithm",
    "list_algorithms",
    "register_algorithm",
]
