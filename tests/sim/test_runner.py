from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from qddot.sim.fuzz import fuzz
from qddot.sim.runner import run_episode
from qddot.task import TaskError, load_task

FIXTURES = Path(__file__).parent / "fixtures"


def _task():
    return load_task(FIXTURES / "task.yaml")


def _controller(tmp_path: Path, body: str, name: str = "controller.py") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _fake_spec(monkeypatch):
    module = types.ModuleType("qddot.spec")
    module.evaluate_specs = lambda specs, t, signals: [
        {"name": item["name"], "stl": item["stl"], "english": "fixture",
         "passed": True, "robustness": 1.0, "witness_t": 0.0, "window": [0.0, float(t[-1])]} for item in specs]
    monkeypatch.setitem(sys.modules, "qddot.spec", module)


def test_load_task_and_validation_errors(tmp_path):
    task = _task()
    assert task.name == "runner_fixture"
    assert task.model_path == (FIXTURES / "pendulum.xml").resolve()
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: broken\nmodel: missing.xml\nduration: -1\ncontrol_dt: 0.02\n")
    with pytest.raises(TaskError, match="duration"):
        load_task(bad)


def test_nominal_run_writes_all_outputs_and_specs(monkeypatch, tmp_path):
    _fake_spec(monkeypatch)
    result = run_episode(_task(), FIXTURES / "controller.py", root=tmp_path)
    assert result.verdict["passed"]
    assert result.verdict["specs"][0]["name"] == "stable"
    assert {p.name for p in result.episode_dir.iterdir()} == {
        "signals.parquet", "events.parquet", "verdict.json", "meta.json", "log.mcap"}
    table = pq.read_table(result.episode_dir / "signals.parquet")
    assert table.num_rows == 11
    assert "joint.hinge.pos" in table.column_names
    assert "ev.collision" in table.column_names


def test_signal_parquet_is_deterministic(tmp_path):
    task = replace(_task(), specs=[])
    first = run_episode(task, FIXTURES / "controller.py", seed=4, root=tmp_path / "a")
    second = run_episode(task, FIXTURES / "controller.py", seed=4, root=tmp_path / "b")
    a = pq.read_table(first.episode_dir / "signals.parquet").to_pydict()
    b = pq.read_table(second.episode_dir / "signals.parquet").to_pydict()
    assert a.keys() == b.keys()
    for name in a:
        assert np.asarray(a[name], dtype=np.float64).tobytes() == np.asarray(b[name], dtype=np.float64).tobytes()


def test_latency_delays_observation_samples(tmp_path):
    output = tmp_path / "seen.txt"
    controller = _controller(tmp_path, f'''\
class Controller:
    def step(self, obs, t):
        with open({str(output)!r}, "a") as stream:
            stream.write(f"{{t}},{{obs['joint.hinge.pos']}}\\n")
        return {{}}
''')
    task = replace(_task(), specs=[])
    run_episode(task, controller, root=tmp_path / "run", perturb={"latency_ms": 40})
    samples = [line.strip().split(",") for line in output.read_text().splitlines()]
    assert len(samples) == 10
    assert float(samples[2][1]) == 0.0
    assert float(samples[3][1]) != 0.0


def test_saturation_and_joint_limit_events(tmp_path):
    controller = _controller(tmp_path, """\
class Controller:
    def step(self, obs, t):
        return {"hinge_motor": 100.0, "lift": 0.0}
""")
    task = replace(_task(), specs=[], init={"qpos": {"hinge": 0.299}})
    result = run_episode(task, controller, root=tmp_path)
    kinds = [row["kind"] for row in pq.read_table(result.episode_dir / "events.parquet").to_pylist()]
    assert "saturation" in kinds
    assert "joint_limit" in kinds


def test_contact_start_and_end_pair(tmp_path):
    controller = _controller(tmp_path, """\
class Controller:
    def step(self, obs, t):
        return {"lift": 0.3 if t >= 0.30 else 0.0}
""")
    task = replace(_task(), specs=[], duration=1.0)
    result = run_episode(task, controller, root=tmp_path)
    rows = pq.read_table(result.episode_dir / "events.parquet").to_pylist()
    contacts = [row["kind"] for row in rows
                if {row["a"], row["b"]} == {"payload", "floorbody"}]
    assert "contact_start" in contacts
    assert "contact_end" in contacts


def test_controller_exception_is_recorded(tmp_path):
    controller = _controller(tmp_path, """\
class Controller:
    def step(self, obs, t):
        raise RuntimeError("fixture failure")
""")
    result = run_episode(replace(_task(), specs=[]), controller, root=tmp_path)
    assert not result.verdict["passed"]
    assert "fixture failure" in result.verdict["errors"][0]
    assert result.verdict["events_summary"]["controller_error"] == 1


def test_fuzz_uses_process_pool(tmp_path):
    result = fuzz(replace(_task(), specs=[]), FIXTURES / "controller.py", n=4,
                  workers=2, seed=10, root=tmp_path)
    assert result["n"] == 4
    assert result["passed"] == 4
    assert result["pass_rate"] == 1.0
    assert len(result["worst"]) == 4
    assert all("pass_corr" in item for item in result["by_param"].values())
