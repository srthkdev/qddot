"""STL parsing, deterministic English rendering, and robustness evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .ast import Node
from .english import to_english_node
from .parser import Parser, SpecError
from .semantics import SpecResult, evaluate_node


def parse(stl: str) -> Node:
    """Parse an STL formula into its immutable syntax tree."""
    if not isinstance(stl, str):
        raise TypeError("stl must be a string")
    return Parser(stl).parse()


def to_english(stl: str) -> str:
    """Translate an STL formula into deterministic plain English."""
    return to_english_node(parse(stl))


def evaluate(stl: str, t: np.ndarray, signals: Mapping[str, np.ndarray]) -> SpecResult:
    """Evaluate STL robustness at each sample and return the time-zero result.

    Empty temporal windows use the robustness identities: +inf for always and
    -inf for eventually/until. Windows are selected by sample time and clipped
    to the available trace; reported windows retain their requested time bounds.
    """
    return evaluate_node(parse(stl), stl, t, signals)


def evaluate_specs(specs: list[dict], t: np.ndarray, signals: Mapping[str, np.ndarray]) -> list[dict]:
    """Evaluate task specs into verdict-compatible dictionaries."""
    results = []
    for spec in specs:
        stl = str(spec["stl"])
        result = evaluate(stl, t, signals)
        results.append(
            {
                "name": spec["name"],
                "stl": stl,
                "english": to_english(stl),
                "passed": result.passed,
                "robustness": result.robustness,
                "witness_t": result.witness_t,
                "window": list(result.window) if result.window is not None else None,
            }
        )
    return results


__all__ = ["SpecError", "SpecResult", "evaluate", "evaluate_specs", "parse", "to_english"]
