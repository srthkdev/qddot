"""Lexer and recursive-descent parser for qddot's STL syntax."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ast import Expr, Node

_TOKEN = re.compile(
    r"\s*(?:(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_.]*)|(?P<op><=|>=|==|[()\[\],+*/<>-]))"
)


class SpecError(ValueError):
    """Invalid STL or signal reference, with source location when available."""

    def __init__(self, message: str, source: str | None = None, column: int = 1):
        if source is not None:
            column = max(1, column)
            message = f"{message} at column {column}\n{source}\n{' ' * (column - 1)}^"
        super().__init__(message)
        self.column = column


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens: list[Token] = []
        pos = 0
        while pos < len(source):
            match = _TOKEN.match(source, pos)
            if not match:
                if source[pos:].strip() == "":
                    pos = len(source)
                    break
                raise SpecError("Unexpected character", source, pos + 1)
            kind = "number" if match.lastgroup == "number" else "name" if match.lastgroup == "name" else "op"
            self.tokens.append(Token(kind, match.group(kind), match.start(kind)))
            pos = match.end()
        self.tokens.append(Token("eof", "", len(source)))
        self.i = 0

    @property
    def token(self) -> Token:
        return self.tokens[self.i]

    def take(self, text: str | None = None) -> Token:
        token = self.token
        if text is not None and token.text.lower() != text:
            self.error(f"Expected {text!r}")
        self.i += 1
        return token

    def error(self, message: str) -> None:
        raise SpecError(message, self.source, self.token.pos + 1)

    def accept(self, text: str) -> bool:
        if self.token.text.lower() == text:
            self.i += 1
            return True
        return False

    def parse(self) -> Node:
        node = self.implies()
        if self.token.kind != "eof":
            self.error(f"Unexpected token {self.token.text!r}")
        return node

    def implies(self) -> Node:
        left = self.until()
        if self.accept("implies"):
            return Node("implies", left, self.implies())
        return left

    def until(self) -> Node:
        node = self.disjunction()
        while self.accept("until"):
            bound = self.bounds()
            node = Node("until", node, self.disjunction(), bound=bound)
        return node

    def disjunction(self) -> Node:
        node = self.conjunction()
        while self.accept("or"):
            node = Node("or", node, self.conjunction())
        return node

    def conjunction(self) -> Node:
        node = self.unary()
        while self.accept("and"):
            node = Node("and", node, self.unary())
        return node

    def unary(self) -> Node:
        if self.accept("not"):
            return Node("not", self.unary())
        if self.token.text.lower() in {"always", "eventually"}:
            op = self.take().text.lower()
            bound = self.bounds()
            self.take("(")
            inner = self.implies()
            self.take(")")
            return Node(op, inner, bound=bound)
        if self.accept("("):
            node = self.implies()
            self.take(")")
            return node
        return self.atom()

    def bounds(self) -> tuple[float, float] | None:
        if not self.accept("["):
            return None
        try:
            low = float(self.take_number())
            self.take(",")
            high = float(self.take_number())
        except ValueError:
            self.error("Bounds must be finite numbers")
        self.take("]")
        if low < 0 or high < low:
            self.error("Bounds must satisfy 0 <= lower <= upper")
        return low, high

    def take_number(self) -> str:
        if self.token.kind != "number":
            self.error("Expected a number")
        return self.take().text

    def atom(self) -> Node:
        left = self.arithmetic()
        if self.token.text not in {"<", "<=", ">", ">=", "=="}:
            self.error("Expected a comparison operator")
        comp = self.take().text
        right = self.arithmetic()
        return Node("compare", expr=Expr("sub", left=left, right=right), comparator=comp)

    def arithmetic(self) -> Expr:
        left = self.term()
        while self.token.text in {"+", "-"}:
            op = self.take().text
            left = Expr("add" if op == "+" else "sub", left=left, right=self.term())
        return left

    def term(self) -> Expr:
        left = self.factor()
        while self.token.text in {"*", "/"}:
            op = self.take().text
            left = Expr("mul" if op == "*" else "div", left=left, right=self.factor())
        return left

    def factor(self) -> Expr:
        if self.accept("-"):
            return Expr("neg", left=self.term())
        if self.accept("+"):
            return self.term()
        if self.accept("abs"):
            self.take("(")
            child = self.arithmetic()
            self.take(")")
            return Expr("abs", left=child)
        if self.accept("("):
            child = self.arithmetic()
            self.take(")")
            return child
        if self.token.kind == "number":
            return Expr("constant", float(self.take().text))
        if self.token.kind == "name" and self.token.text.lower() not in {"and", "or", "not", "implies", "until", "always", "eventually"}:
            token = self.take()
            return Expr("signal", token.text, pos=token.pos)
        self.error("Expected a signal, number, or parenthesized arithmetic expression")
        raise AssertionError
