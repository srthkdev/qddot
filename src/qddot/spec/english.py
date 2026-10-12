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
        return f"-{_expr(expr.left)}"
    if expr.op == "abs":
        return f"|{_expr(expr.left)}|"
    assert expr.right is not None
    symbol = {"add": " + ", "sub": " - ", "mul": " × ", "div": " / "}[expr.op]
    right = _expr(expr.right)
    if expr.op == "sub" and (
        (expr.right.op == "constant" and float(expr.right.value) < 0)
        or (expr.right.op == "neg" and expr.right.left is not None
            and expr.right.left.op == "constant")
    ):
        right = f"({right})"
    return _expr(expr.left) + symbol + right


def _parts(node: Node) -> tuple[Expr, Expr, str]:
    assert node.expr is not None and node.expr.op == "sub"
    assert node.expr.left is not None and node.expr.right is not None
    return node.expr.left, node.expr.right, node.comparator or "<"


def _atom(node: Node) -> str:
    left, right, comparator = _parts(node)
    words = {"<": "is below", "<=": "is at most", ">": "is above",
             ">=": "is at least", "==": "equals"}
    if right.op == "constant" and left.op == "signal":
        signal, value = str(left.value), float(right.value)
        if value >= 0.5 and signal.startswith("ev.") and comparator in {">", ">="}:
            return f"a {signal[3:].replace('_', ' ')} is hit"
        if value <= 0.5 and signal.startswith("ev.") and comparator in {"<", "<="}:
            return f"no {signal[3:].replace('_', ' ')} occurs"
        if value >= 0.5 and signal.startswith("col_"):
            label = signal[4:].replace("_", " ")
            return f"no {label} collision occurs" if comparator in {"<", "<="} else f"a {label} collision occurs"
    return f"{_expr(left)} {words[comparator]} {_expr(right)}"


def _threshold(node: Node) -> tuple[str, str, str] | None:
    if node.op != "compare":
        return None
    left, right, comparator = _parts(node)
    if right.op != "constant":
        return None
    return _expr(left), comparator, _number(float(right.value))


def _held(node: Node, duration: float) -> str | None:
    threshold = _threshold(node)
    if threshold is None:
        return None
    expression, comparator, value = threshold
    if comparator in {"<", "<="}:
        return f"{expression} drops below {value} and stays there for {_number(duration)} s"
    if comparator in {">", ">="}:
        return f"{expression} rises above {value} and stays there for {_number(duration)} s"
    return f"{expression} equals {value} and stays there for {_number(duration)} s"


def _render(node: Node) -> str:
    if node.op == "compare":
        return _atom(node)
    if node.op == "not":
        child = node.left
        if child is not None and child.op == "eventually" and child.left is not None:
            atom = child.left
            if atom.op == "compare":
                left, _, _ = _parts(atom)
                if left.op == "signal" and str(left.value).startswith("ev."):
                    return f"no {str(left.value)[3:].replace('_', ' ')} is ever hit"
                if left.op == "signal" and str(left.value).startswith("col_"):
                    return f"no {str(left.value)[4:].replace('_', ' ')} collision ever occurs"
            duration = child.bound[1] if child.bound else None
            threshold = _threshold(atom)
            if threshold:
                expression, comparator, value = threshold
                action = "rises above" if comparator in {">", ">="} else "drops below"
                suffix = f" within {_number(duration)} s" if duration is not None else ""
                return f"{expression} never {action} {value}{suffix}"
            suffix = f" within {_number(duration)} s" if duration is not None else ""
            return f"{_render(atom)} never occurs{suffix}"
        return "not " + _render(child) if child is not None else "not true"
    if node.op == "eventually":
        child = node.left
        assert child is not None
        if child.op == "always" and child.left is not None:
            hold = _held(child.left, child.bound[1] - child.bound[0] if child.bound else 0.0)
            if hold:
                if node.bound:
                    low, high = node.bound
                    prefix = (f"within {_number(high)} s, " if low == 0 else
                              f"between {_number(low)} and {_number(high)} s, ")
                else:
                    prefix = ""
                return prefix + hold
        prefix = f"within {_number(node.bound[1])} s, " if node.bound else "eventually, "
        return prefix + _render(child)
    if node.op == "always":
        child = node.left
        assert child is not None
        if child.op == "implies" and child.left is not None and child.right is not None:
            prefix = f"from {_number(node.bound[0])} to {_number(node.bound[1])} s, " if node.bound else ""
            return prefix + f"whenever {_render(child.left)}, {_render(child.right)}"
        threshold = _threshold(child)
        if threshold:
            expression, comparator, value = threshold
            if expression.startswith(("col_", "ev.")):
                phrase = _atom(child)
                if node.bound is None:
                    return phrase + " for the whole run"
                interval = (f"for {_number(node.bound[1])} s" if node.bound[0] == 0 else
                            f"from {_number(node.bound[0])} to {_number(node.bound[1])} s")
                return phrase + f" {interval}"
            stay = "below" if comparator in {"<", "<="} else "above" if comparator in {">", ">="} else "equal to"
            if expression.startswith(("no ", "a ")):
                phrase = _render(child)
                return phrase + (" for the whole run" if node.bound is None else " throughout the interval")
            if node.bound is None:
                return f"{expression} stays {stay} {value} for the whole run"
            low, high = node.bound
            if low == 0:
                return f"for {_number(high)} s, {expression} stays {stay} {value}"
            return f"from {_number(low)} to {_number(high)} s, {expression} stays {stay} {value}"
        if node.bound:
            return f"from {_number(node.bound[0])} to {_number(node.bound[1])} s, {_render(child)}"
        return f"{_render(child)} for the whole run"
    if node.op == "until":
        assert node.left is not None and node.right is not None
        bound = f" within {_number(node.bound[1])} s" if node.bound else ""
        left, right = _threshold(node.left), _threshold(node.right)
        if left and right:
            left_action = "below" if left[1] in {"<", "<="} else "above"
            right_action = "rises above" if right[1] in {">", ">="} else "drops below"
            return f"{left[0]} stays {left_action} {left[2]} until{bound} {right[0]} {right_action} {right[2]}"
        return f"{_render(node.left)} holds until{bound} {_render(node.right)}"
    if node.op in {"and", "or", "implies"}:
        assert node.left is not None and node.right is not None
        if node.op == "implies":
            return f"if {_render(node.left)}, then {_render(node.right)}"
        word = "and" if node.op == "and" else "or"
        return f"({_render(node.left)}) {word} ({_render(node.right)})"
    raise ValueError(f"Unknown formula operator: {node.op}")


def to_english_node(node: Node) -> str:
    return _render(node)
