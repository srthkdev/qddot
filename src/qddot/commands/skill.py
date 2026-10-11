"""Install the qddot agent workflow skill."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from ._out import console, exit_error


def _skill_source() -> Path:
    return Path(__file__).resolve().parents[3] / "skills" / "qddot"


def install_skill(destination: Path, force: bool = False) -> bool:
    """Copy the packaged skill; return whether a write occurred."""
    source = _skill_source()
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"qddot skill source not found: {source}")
    if destination.exists():
        same = destination.is_dir() and all(
            (destination / item.relative_to(source)).is_file()
            and (destination / item.relative_to(source)).read_bytes() == item.read_bytes()
            for item in source.rglob("*") if item.is_file()
        )
        if same:
            return False
        if not force:
            raise FileExistsError(f"{destination} exists; pass --force to overwrite")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return True


def register(app: typer.Typer) -> None:
    @app.command("skill")
    def skill_command(
        install: str = typer.Argument(..., help="Action: install."),
        claude: bool = typer.Option(False, "--claude", help="Install for Claude Code."),
        codex: bool = typer.Option(False, "--codex", help="Install for Codex."),
        project: bool = typer.Option(False, "--project", help="Install in this project's .claude directory."),
        force: bool = typer.Option(False, "--force", help="Replace an existing different skill."),
    ) -> None:
        """Install qddot's robot simulation workflow skill."""
        try:
            if install != "install":
                raise ValueError("usage: qddot skill install [--claude|--codex] [--project]")
            if sum((claude, codex, project)) > 1:
                raise ValueError("choose only one of --claude, --codex, or --project")
            if claude:
                destination = Path.home() / ".claude/skills/qddot"
            elif project:
                destination = Path.cwd() / ".claude/skills/qddot"
            else:
                destination = Path.home() / ".codex/skills/qddot"
            changed = install_skill(destination, force=force)
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, False)
        console.print(f"{'installed' if changed else 'already installed'}: {destination}")
