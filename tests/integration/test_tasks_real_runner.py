"""Built-in controller regression coverage through the production runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qddot.sim.fuzz import fuzz
from qddot.sim.runner import run_episode
from qddot.task import load_task

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"


@pytest.mark.parametrize("task_path", sorted(TASKS.glob("*/task.yaml")), ids=lambda path: path.parent.name)
def test_builtin_controllers_real_runner(task_path: Path, tmp_path: Path) -> None:
    task_dir = task_path.parent
    task = load_task(task_path)
    expectations = yaml.safe_load((task_dir / "expected.yaml").read_text(encoding="utf-8"))
    good = run_episode(task, task_dir / "controllers/good.py", root=tmp_path, record_mcap=False)
    assert good.verdict["passed"], good.verdict

    fuzz_result = fuzz(task, task_dir / "controllers/good.py", n=10, workers=1, seed=17, root=tmp_path)
    assert fuzz_result["pass_rate"] >= 0.9, fuzz_result

    for controller_name, expectation in expectations.items():
        result = run_episode(task, task_dir / "controllers" / f"{controller_name}.py", root=tmp_path,
                             record_mcap=False)
        assert not result.verdict["passed"], result.verdict
        first = result.verdict["first_violation"]
        assert first and first["spec"] == expectation["first_failing_spec"], result.verdict
        observed = set(result.verdict["events_summary"])
        assert set(expectation["must_events"]) <= observed, result.verdict
