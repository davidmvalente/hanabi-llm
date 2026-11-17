"""Name -> Python function registry.

The ontology refers to behaviour by *name* (``hn:checkFunction``,
``hn:effectFunction``). Implementations register themselves here with
``@bind("name")``. ``ontology.verify_bindings`` then proves that every name
the ontology uses is implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: Function signatures differ per role (precondition, effect, predicate), so the
#: values are deliberately loose here and typed precisely at each definition.
FUNCTIONS: dict[str, Callable[..., Any]] = {}


def bind[F: Callable[..., Any]](name: str) -> Callable[[F], F]:
    """Register ``fn`` under an ontology function name."""

    def decorator(fn: F) -> F:
        if name in FUNCTIONS:
            raise RuntimeError(f"function {name!r} is already bound")
        FUNCTIONS[name] = fn
        return fn

    return decorator
