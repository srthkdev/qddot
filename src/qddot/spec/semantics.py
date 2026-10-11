"""Vectorized quantitative STL evaluation and critical-sample tracking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches

import numpy as np

from .ast import Expr, Node
from .parser import SpecError


@dataclass(frozen=True)
class SpecResult:
    passed: bool
    robustness: float
    witness_t: float | None
    window: tuple[float, float] | None


@dataclass
class _Trace:
    value: np.ndarray
    witness: np.ndarray
    window: list[tuple[float, float] | None]


def _expr(expr: Expr, signals: Mapping[str, np.ndarray], source: str, size: int) -> np.ndarray:
    if expr.op == "constant":
        return np.full(size, float(expr.value), dtype=np.float64)
    if expr.op == "signal":
        name = str(expr.value)
        if name not in signals:
            suggestions = get_close_matches(name, list(signals), n=3, cutoff=0.45)
            suffix = f". Did you mean {', '.join(suggestions)}?" if suggestions else ""
            suffix += f" Available signals: {', '.join(list(signals)[:10])}."
            raise SpecError(f"Unknown signal {name!r}{suffix}", source, (expr.pos or 0) + 1)
        return np.asarray(signals[name], dtype=np.float64)
    assert expr.left is not None
    left = _expr(expr.left, signals, source, size)
    if expr.op == "neg":
        return -left
    if expr.op == "abs":
        return np.abs(left)
    assert expr.right is not None
    right = _expr(expr.right, signals, source, size)
    if expr.op == "add":
        return left + right
    if expr.op == "sub":
        return left - right
    if expr.op == "mul":
        return left * right
    if expr.op == "div":
        return left / right
    raise ValueError(f"Unknown arithmetic operator: {expr.op}")


def _ranges(times: np.ndarray, bound: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    low, high = bound if bound is not None else (0.0, np.inf)
    starts = np.searchsorted(times, times + low, side="left")
    ends = np.searchsorted(times, times + high, side="right")
    return starts, ends


def _range_extreme(values: np.ndarray, starts: np.ndarray, ends: np.ndarray, maximum: bool) -> tuple[np.ndarray, np.ndarray]:
    """Range min/max via a vectorized sparse table; empty ranges use STL identities."""
    n = len(values)
    identity = -np.inf if maximum else np.inf
    out = np.full(n, identity, dtype=np.float64)
    indexes = np.full(n, -1, dtype=np.int64)
    length = ends - starts
    valid = length > 0
    if not np.any(valid):
        return out, indexes
    # Two overlapping power-of-two blocks answer each static range query in O(1).
    table = [values]
    arg = [np.arange(n, dtype=np.int64)]
    width = 1
    while width * 2 <= n:
        prev = table[-1]
        prevarg = arg[-1]
        take_right = prev[width:] > prev[:-width] if maximum else prev[width:] < prev[:-width]
        table.append(np.where(take_right, prev[width:], prev[:-width]))
        arg.append(np.where(take_right, prevarg[width:], prevarg[:-width]))
        width *= 2
    rows = np.flatnonzero(valid)
    lens = length[rows]
    levels = np.floor(np.log2(lens)).astype(np.int64)
    offsets = 1 << levels
    left_idx = starts[rows]
    right_idx = ends[rows] - offsets
    va = np.empty(len(rows), dtype=np.float64)
    vb = np.empty(len(rows), dtype=np.float64)
    ia = np.empty(len(rows), dtype=np.int64)
    ib = np.empty(len(rows), dtype=np.int64)
    for level in np.unique(levels):
        mask = levels == level
        va[mask] = table[level][left_idx[mask]]
        vb[mask] = table[level][right_idx[mask]]
        ia[mask] = arg[level][left_idx[mask]]
        ib[mask] = arg[level][right_idx[mask]]
    choose_b = vb > va if maximum else vb < va
    out[rows] = np.where(choose_b, vb, va)
    indexes[rows] = np.where(choose_b, ib, ia)
    return out, indexes


def _better(a: np.ndarray, b: np.ndarray, maximum: bool) -> np.ndarray:
    return b > a if maximum else b < a


def _eval(node: Node, times: np.ndarray, signals: Mapping[str, np.ndarray], source: str) -> _Trace:
    n = len(times)
    all_idx = np.arange(n, dtype=np.int64)
    if node.op == "compare":
        assert node.expr is not None and node.expr.left is not None and node.expr.right is not None
        lhs = _expr(node.expr.left, signals, source, n)
        rhs = _expr(node.expr.right, signals, source, n)
        if node.comparator in {">", ">="}:
            val = lhs - rhs
        elif node.comparator in {"<", "<="}:
            val = rhs - lhs
        else:
            val = -np.abs(lhs - rhs)
        return _Trace(val, all_idx, [None] * n)
    if node.op == "not":
        child = _eval(node.left, times, signals, source)  # type: ignore[arg-type]
        return _Trace(-child.value, child.witness, child.window)
    if node.op in {"always", "eventually"}:
        child = _eval(node.left, times, signals, source)  # type: ignore[arg-type]
        maximum = node.op == "eventually"
        starts, ends = _ranges(times, node.bound)
        values, selected = _range_extreme(child.value, starts, ends, maximum)
        witnesses = np.full(n, -1, dtype=np.int64)
        windows: list[tuple[float, float] | None] = [None] * n
        for i in range(n):
            if selected[i] >= 0:
                j = selected[i]
                witnesses[i] = child.witness[j]
                # The first temporal operator from the root owns the reported window.
                windows[i] = (float(times[starts[i]]), float(times[ends[i] - 1]))
        return _Trace(values, witnesses, windows)
    if node.op in {"and", "or", "implies"}:
        left = _eval(node.left, times, signals, source)  # type: ignore[arg-type]
        right = _eval(node.right, times, signals, source)  # type: ignore[arg-type]
        lv = -left.value if node.op == "implies" else left.value
        maximum = node.op in {"or", "implies"}
        choose_right = _better(lv, right.value, maximum)
        val = np.maximum(lv, right.value) if maximum else np.minimum(lv, right.value)
        chosen = np.where(choose_right, right.witness, left.witness)
        chosen_windows = [right.window[i] if choose_right[i] else left.window[i] for i in range(n)]
        return _Trace(val, chosen, chosen_windows)
    if node.op == "until":
        left = _eval(node.left, times, signals, source)  # type: ignore[arg-type]
        right = _eval(node.right, times, signals, source)  # type: ignore[arg-type]
        starts, ends = _ranges(times, node.bound)
        values = np.full(n, -np.inf, dtype=np.float64)
        witnesses = np.full(n, -1, dtype=np.int64)
        windows: list[tuple[float, float] | None] = [None] * n
        for i in range(n):
            best = -np.inf
            best_witness = -1
            best_window: tuple[float, float] | None = (float(times[starts[i]]), float(times[ends[i] - 1])) if ends[i] > starts[i] else None
            for j in range(starts[i], ends[i]):
                # Prefix min of the left operand over [i, j], inclusive.
                prefix_value = float(np.min(left.value[i : j + 1]))
                prefix_index = i + int(np.argmin(left.value[i : j + 1]))
                candidate = min(prefix_value, float(right.value[j]))
                if candidate > best:
                    best = candidate
                    best_witness = right.witness[j] if right.value[j] <= prefix_value else left.witness[prefix_index]
            values[i] = best
            witnesses[i] = best_witness
            windows[i] = best_window
        return _Trace(values, witnesses, windows)
    raise ValueError(f"Unknown formula operator: {node.op}")


def evaluate_node(node: Node, source: str, t: np.ndarray, signals: Mapping[str, np.ndarray]) -> SpecResult:
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("t must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("t must contain finite, non-decreasing sample times")
    normalized = {name: np.asarray(values, dtype=np.float64) for name, values in signals.items()}
    for name, values in normalized.items():
        if values.ndim != 1 or len(values) != len(times):
            raise ValueError(f"Signal {name!r} must be one-dimensional and match t")
    trace = _eval(node, times, normalized, source)
    index = int(trace.witness[0])
    return SpecResult(
        passed=bool(trace.value[0] >= 0.0),
        robustness=float(trace.value[0]),
        witness_t=float(times[index]) if 0 <= index < len(times) else None,
        window=trace.window[0],
    )
