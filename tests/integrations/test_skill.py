from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from qddot.cli import app
from qddot.commands.skill import install_skill


def test_skill_install_is_idempotent_and_force_replaces(tmp_path: Path) -> None:
    target = tmp_path / "skills/qddot"
    assert install_skill(target)
    assert (target / "SKILL.md").is_file()
    assert (target / "reference/events.md").is_file()
    assert not install_skill(target)
    (target / "SKILL.md").write_text("custom", encoding="utf-8")
    try:
        install_skill(target)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected overwrite protection")
    assert install_skill(target, force=True)
    assert "Never claim done" in (target / "SKILL.md").read_text(encoding="utf-8")


def test_skill_cli_uses_temporary_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    result = CliRunner().invoke(app, ["skill", "install", "--codex"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".codex/skills/qddot/SKILL.md").exists()
    second = CliRunner().invoke(app, ["skill", "install", "--codex"])
    assert second.exit_code == 0
    assert "already installed" in second.output
