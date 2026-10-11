"""Launch the qddot stdio MCP server."""

from __future__ import annotations

import typer

from ._out import exit_error


def register(app: typer.Typer) -> None:
    @app.command("mcp")
    def mcp_command() -> None:
        """Serve qddot tools over the Model Context Protocol on stdio."""
        try:
            from qddot.mcp_server import server

            server.run(transport="stdio")
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, False)
