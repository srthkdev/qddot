import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import qddot.diag
from qddot.cli import app

runner = CliRunner()


def _module(monkeypatch, name: str, **attributes) -> None:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def _files(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "task.yaml"
    controller = tmp_path / "controller.py"
    task.write_text("name: fake", encoding="utf-8")
    controller.write_text("class Controller: pass", encoding="utf-8")
    return task, controller


def _install_task(monkeypatch) -> None:
    _module(monkeypatch, "qddot.task", load_task=lambda path: SimpleNamespace(specs=[{"name": "reach", "stl": "x > 0"}]))


def test_run_pass_and_failure_exit_codes_and_json(monkeypatch, tmp_path: Path) -> None:
    task, controller = _files(tmp_path)
    _install_task(monkeypatch)
    verdict = {"task": "fake", "episode": "fake-ctrl-s0-aabbcc", "passed": True, "robustness": 0.12, "specs": [{"name": "reach", "passed": True, "robustness": 0.12}]}
    _module(monkeypatch, "qddot.sim.runner", run_episode=lambda *a, **k: SimpleNamespace(verdict=verdict, episode_dir=tmp_path / "run"))
    result = runner.invoke(app, ["run", str(task), "--controller", str(controller), "--perturb", "mass_scale=1.2", "--perturb", "controller_budget_ms=null"])
    assert result.exit_code == 0, result.output
    assert "PASS fake" in result.output
    assert "reach" in result.output

    verdict["passed"] = False
    result = runner.invoke(app, ["run", str(task), "--controller", str(controller), "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["passed"] is False

    monkeypatch.setattr(qddot.diag, "why", lambda _path: {"likely_causes": [{
        "cause": "bad sign", "confidence": "high"
    }]})
    result = runner.invoke(app, ["run", str(task), "--controller", str(controller)])
    assert result.exit_code == 1
    assert "why: bad sign [high]" in result.output


def test_perturb_parse_error_is_usage_error(monkeypatch, tmp_path: Path) -> None:
    task, controller = _files(tmp_path)
    _install_task(monkeypatch)
    result = runner.invoke(app, ["run", str(task), "--controller", str(controller), "--perturb", "bad", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout) == {"error": "perturbation 'bad' must use key=value", "kind": "ValueError"}


def test_plain_output_preserves_markup_brackets_and_long_paths(monkeypatch, tmp_path: Path) -> None:
    task, controller = _files(tmp_path)
    _install_task(monkeypatch)
    long_path = tmp_path / ("folder[" + "x" * 150 + "]")
    verdict = {"task": "fake", "episode": "ep", "passed": True, "robustness": 1.0, "specs": []}
    _module(monkeypatch, "qddot.sim.runner", run_episode=lambda *a, **k: SimpleNamespace(verdict=verdict, episode_dir=long_path))
    result = runner.invoke(app, ["run", str(task), "--controller", str(controller)])
    assert result.exit_code == 0, result.output
    assert str(long_path) in result.output
    assert "[bold" not in result.output


def test_fuzz_reports_summary_and_failure_exit(monkeypatch, tmp_path: Path) -> None:
    task, controller = _files(tmp_path)
    _install_task(monkeypatch)
    summary = {"n": 2, "passed": 1, "pass_rate": 0.5, "worst": [{"episode": "worst", "robustness": -0.2}], "by_param": {"mass_scale": {"pass_corr": 0.4}}}
    _module(monkeypatch, "qddot.sim.fuzz", fuzz=lambda *a, **k: summary)
    result = runner.invoke(app, ["fuzz", str(task), "--controller", str(controller), "-n", "2"])
    assert result.exit_code == 1
    assert "1/2 passed" in result.output
    assert "mass_scale" in result.output


def test_spec_uses_english_translation(monkeypatch, tmp_path: Path) -> None:
    task, _ = _files(tmp_path)
    _install_task(monkeypatch)
    _module(monkeypatch, "qddot.spec", to_english=lambda stl: "x is positive")
    result = runner.invoke(app, ["spec", str(task)])
    assert result.exit_code == 0
    assert "x is positive" in result.output


def test_check_json_and_load_error_exit_code(tmp_path: Path) -> None:
    model = tmp_path / "broken.xml"
    model.write_text("<mujoco>", encoding="utf-8")
    result = runner.invoke(app, ["check", str(model), "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)[0]["code"] == "load_error"


def test_query_json_and_sql_error_help(tmp_path: Path) -> None:
    result = runner.invoke(app, ["query", "select count(*) as n from signals", "--root", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["rows"] == [{"n": 0}]
    broken = runner.invoke(app, ["query", "select * from missing_view", "--root", str(tmp_path)])
    assert broken.exit_code == 2
    assert "Available views and columns" in broken.output
    assert "signals(" in broken.output
    broken_json = runner.invoke(app, ["query", "select * from missing_view", "--root", str(tmp_path), "--json"])
    assert broken_json.exit_code == 2
    assert set(json.loads(broken_json.stdout)) == {"error", "kind"}
