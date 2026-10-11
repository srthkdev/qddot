import runpy
import sys
import types
from pathlib import Path

from bench.workdir import build_workdir

ROOT = Path(__file__).resolve().parents[2]


def test_workdir_excludes_private_files_and_has_condition(tmp_path):
    workdir = build_workdir(ROOT / "tasks/cartpole_balance", "none", repo_root=ROOT, destination=tmp_path / "wd")
    assert (workdir / "PROMPT.md").is_file()
    assert (workdir / "task.yaml").is_file()
    assert (workdir / "controller.py").is_file()
    assert (workdir / "CONDITION.md").is_file()
    names = {path.name for path in workdir.rglob("*") if path.is_file()}
    assert not any(name in {"good.py", "KNOWN_FAILURES.md", "expected.yaml"} or name.startswith("buggy_") for name in names)
    assert "model" in (workdir / "task.yaml").read_text()


def test_stdout_runner_prints_exactly_three_lines(tmp_path, monkeypatch, capsys):
    workdir = build_workdir(ROOT / "tasks/so101_reach", "stdout", repo_root=ROOT, destination=tmp_path / "stdout")
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    task_mod = types.ModuleType("qddot.task")
    task_mod.load_task = lambda _path: types.SimpleNamespace(signals=[{"name": "first"}, {"name": "second"}])
    runner_mod = types.ModuleType("qddot.sim.runner")
    runner_mod.run_episode = lambda *_args, **_kwargs: types.SimpleNamespace(episode_dir=episode_dir)
    parquet_mod = types.ModuleType("pyarrow.parquet")
    parquet_mod.read_table = lambda _path: types.SimpleNamespace(to_pydict=lambda: {"first": [1.5], "second": [2.5]})
    package = types.ModuleType("pyarrow")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "qddot", types.ModuleType("qddot"))
    monkeypatch.setitem(sys.modules, "qddot.task", task_mod)
    monkeypatch.setitem(sys.modules, "qddot.sim", types.ModuleType("qddot.sim"))
    monkeypatch.setitem(sys.modules, "qddot.sim.runner", runner_mod)
    monkeypatch.setitem(sys.modules, "pyarrow", package)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet_mod)
    runpy.run_path(str(workdir / "sim.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == ["first: 1.5", "second: 2.5", "done"]
