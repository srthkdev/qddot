"""Abstract syntax tree nodes for the supported STL subset."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    op: str
    value: float | str | None = None
    left: Expr | None = None
    right: Expr | None = None
    pos: int | None = None


@dataclass(frozen=True)
class Node:
    op: str
    left: Node | None = None
    right: Node | None = None
    expr: Expr | None = None
    comparator: str | None = None
    bound: tuple[float, float] | None = None
