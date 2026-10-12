"""Index recorded episodes into qddot's failure memory."""

from __future__ import annotations

from pathlib import Path

import typer

from qddot.memory.store import database_path, index_runs

from ._out import console, emit_json, exit_error


def register(app: typer.Typer) -> None:
    @app.command()
    def remember(
        episode: Path | None = typer.Argument(None),  # noqa: B008
        all_episodes: bool = typer.Option(False, "--all"),
        global_store: bool = typer.Option(False, "--global"),
        root: Path = typer.Option(Path(".qddot"), "--root"),  # noqa: B008
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Index one episode or all project runs into failure memory."""
        try:
            if all_episodes:
                count = index_runs(root / "runs", database_path(root, global_store))
            elif episode is not None:
                if not (episode / "verdict.json").is_file():
                    raise FileNotFoundError(f"episode verdict not found: {episode / 'verdict.json'}")
                count = index_runs(episode.parent, database_path(root, global_store))
            else:
                raise ValueError("provide an EPISODE or use --all")
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        result = {"indexed": count, "database": str(database_path(root, global_store))}
        if json_output:
            emit_json(result)
        else:
            console.print(f"indexed {count} episode(s) into {result['database']}")
