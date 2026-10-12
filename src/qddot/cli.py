"""Typer application assembly for qddot."""

from __future__ import annotations

import typer

COMMANDS = [
    "qddot.commands.run",
    "qddot.commands.fuzz",
    "qddot.commands.check",
    "qddot.commands.init",
    "qddot.commands.spec",
    "qddot.commands.query",
    "qddot.commands.runs",
    "qddot.commands.skill",
    "qddot.commands.mcp",
    "qddot.commands.view",
    "qddot.commands.why",
    "qddot.commands.diff",
    "qddot.commands.replay",
    "qddot.commands.remember",
    "qddot.commands.similar",
]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

for command_module in COMMANDS:
    module = __import__(command_module, fromlist=["register"])
    module.register(app)


if __name__ == "__main__":
    app()
