"""Shared command output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console(no_color=True, soft_wrap=True, markup=False, highlight=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def emit_json(value: Any) -> None:
    typer.echo(json.dumps(value, default=_json_default, sort_keys=True, allow_nan=False))


def emit_error(error: BaseException, json_output: bool) -> None:
    if json_output:
        emit_json({"error": str(error), "kind": type(error).__name__})
    else:
        console.print(f"error: {error}")


def exit_error(error: BaseException, json_output: bool) -> None:
    emit_error(error, json_output)
    raise typer.Exit(2)
