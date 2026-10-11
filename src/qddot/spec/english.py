"""Deterministic plain-English rendering for STL formulas."""

from __future__ import annotations

from .ast import Expr, Node


def _number(value: float) -> str:
    return f"{value:g}"


def _expr(expr: Expr) -> str:
    if expr.op == "constant":
        return _number(float(expr.value))
    if expr.op == "signal":
        return str(expr.value)
    assert expr.left is not None
    if expr.op == "neg":
        return f"minus {_expr(expr.left)}"
    if expr.op == "abs":
        return f"the absolute value of {_expr(expr.left)}"
    assert expr.right is not None
    symbol = {"add": " plus ", "sub": " minus ", "mul": " times "}[expr.op]
    return _expr(expr.left) + symbol + _expr(expr.right)


def _atom(node: Node) -> str:
    assert node.expr is not None and node.expr.left is not None and node.expr.right is not None
    left, right = node.expr.left, node.expr.right
    comp = node.comparator
    # Most task specs compare one signal with a constant; phrase these naturally.
    if left.op == "signal" and right.op == "constant":
        signal, value = str(left.value), _number(float(right.value))
        phrase = {"<": f"{signal} becomes less than {value}", "<=": f"{signal} becomes at most {value}", ">": f"{signal} becomes greater than {value}", ">=": f"{signal} becomes at least {value}", "==": f"{signal} equals {value}"}[comp or "<"]
        return phrase
    phrase = {"<": "is less than", "<=": "is at most", ">": "is greater than", ">=": "is at least", "==": "equals"}[comp or "<"]
    return f"{_expr(left)} {phrase} {_expr(right)}"


def _render(node: Node) -> str:
    if node.op == "compare":
        return _atom(node)
    if node.op == "not":
        return "it is not the case that " + _render(node.left)  # type: ignore[arg-type]
    if node.op in {"always", "eventually"}:
        child = node.left
        assert child is not None
        duration = node.bound[1] - node.bound[0] if node.bound else None
        prefix = f"within {_number(node.bound[1])} s, " if node.op == "eventually" and node.bound else ""
        if node.op == "always":
            if child.op == "compare" and duration is not None:
                return f"{_render(child)} and then stays {('below' if child.comparator in {'<', '<='} else 'above' if child.comparator in {'>', '>='} else 'at')} its threshold for {_number(duration)} s"
            if duration is not None:
                return f"{_render(child)} holds continuously for {_number(duration)} s"
            return f"{_render(child)} always holds"
        if child.op == "always" and child.left is not None and child.bound is not None and node.bound is not None:
            atom = child.left
            if atom.op == "compare" and atom.expr and atom.expr.left and atom.expr.right and atom.expr.left.op == "signal" and atom.expr.right.op == "constant":
                signal = str(atom.expr.left.value)
                value = _number(float(atom.expr.right.value))
                if atom.comparator in {"<", "<="}:
                    return f"within {_number(node.bound[1])} s, {signal} becomes less than {value} and then stays below it for {_number(child.bound[1] - child.bound[0])} s"
        return prefix + _render(child)
    if node.op in {"and", "or", "implies", "until"}:
        assert node.left is not None and node.right is not None
        if node.op == "until":
            when = f" within {_number(node.bound[1])} s" if node.bound else ""
            return f"{_render(node.left)} holds until{when} {_render(node.right)}"
        word = {"and": "and", "or": "or", "implies": "implies"}[node.op]
        return f"({_render(node.left)}) {word} ({_render(node.right)})"
    raise ValueError(f"Unknown formula operator: {node.op}")


def to_english_node(node: Node) -> str:
    return _render(node)
