from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from qddot import mcp_server


def test_server_lists_all_tools() -> None:
    tools = asyncio.run(mcp_server.server.list_tools())
    assert {item.name for item in tools} == {
        "run", "fuzz", "why", "diff", "query", "spec", "check", "replay", "similar"
    }
    assert all(item.description for item in tools)


def test_run_fuzz_and_diagnostics_with_fakes(monkeypatch, tmp_path: Path) -> None:
    from qddot import task as task_module
    from qddot.sim import fuzz as fuzz_module
    from qddot.sim import runner

    monkeypatch.setattr(task_module, "load_task", lambda _path: "loaded-task")
    monkeypatch.setattr(runner, "run_episode", lambda *a, **kw: SimpleNamespace(
        episode_dir=tmp_path, verdict={"episode": "fake", "passed": True}
    ))
    monkeypatch.setattr(fuzz_module, "fuzz", lambda *a, **kw: {"n": 3, "passed": 3, "pass_rate": 1.0})
    assert mcp_server.run("task.yaml", "controller.py")["passed"] is True
    assert mcp_server.fuzz("task.yaml", "controller.py", n=3)["pass_rate"] == 1.0

    diag = SimpleNamespace(
        why=lambda path: {"episode": Path(path).name},
        diff=lambda a, b: {"a": Path(a).name, "b": Path(b).name},
        replay=lambda path, t, frame=False: {"path": str(path), "t": t, "frame": frame},
    )
    monkeypatch.setitem(__import__("sys").modules, "qddot.diag", diag)
    assert mcp_server.why("episode-x")["episode"] == "episode-x"
    assert mcp_server.diff("a", "b")["b"] == "b"
    assert mcp_server.replay("episode-x", 1.0, True)["frame"] is True


def test_query_caps_rows_and_spec_check(monkeypatch, tmp_path: Path) -> None:
    class Cursor:
        def __init__(self) -> None:
            self.description = [("n",)]

        def execute(self, _sql):
            return self

        def fetchmany(self, count):
            return [(i,) for i in range(count)]

        def close(self):
            pass

    from qddot import store
    monkeypatch.setattr(store, "connect", lambda _root: Cursor())
    result = mcp_server.query("select n", str(tmp_path), limit=500)
    assert len(result["rows"]) == 50
    assert result["truncated"] is True

    from qddot import spec as spec_module
    from qddot import task as task_module
    from qddot.check import static as check_module

    class Task:
        def __init__(self) -> None:
            self.name = "fake"
            self.specs = [{"name": "safe", "stl": "x < 1"}]

    monkeypatch.setattr(task_module, "load_task", lambda _path: Task())
    monkeypatch.setattr(spec_module, "to_english", lambda _stl: "x stays safe")
    monkeypatch.setattr(check_module, "check_model", lambda _path: [{"severity": "info"}])
    assert mcp_server.spec("task.yaml")["specs"][0]["english"] == "x stays safe"
    assert mcp_server.check("model.xml")["findings"][0]["severity"] == "info"
