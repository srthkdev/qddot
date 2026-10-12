"""Built-in controller regression coverage through the production runner."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

from qddot.check.static import check_model
from qddot.sim.fuzz import fuzz
from qddot.sim.runner import run_episode
from qddot.task import load_task

ROOT = Path(__file__).resolve().parents[2]
TASKS = Path(os.environ.get("QDDOT_TASKS_DIR", ROOT / "tasks")).resolve()


@pytest.mark.parametrize(
    "task_path", sorted(TASKS.glob("*/task.yaml")), ids=lambda path: path.parent.name
)
def test_builtin_controllers_real_runner(task_path: Path, tmp_path: Path) -> None:
    task_dir = task_path.parent
    task = load_task(task_path)
    expectations = yaml.safe_load((task_dir / "expected.yaml").read_text(encoding="utf-8"))
    fixed_model = expectations.get("fixed_model")
    good_task = copy.copy(task)
    if fixed_model:
        good_task.model_path = (task_dir / fixed_model).resolve()
    good = run_episode(
        good_task, task_dir / "controllers/good.py", root=tmp_path, record_mcap=False
    )
    assert good.verdict["passed"], good.verdict

    fuzz_result = fuzz(
        good_task, task_dir / "controllers/good.py", n=10, workers=1, seed=17, root=tmp_path
    )
    assert fuzz_result["pass_rate"] >= 0.9, fuzz_result

    for controller_name, expectation in expectations.items():
        if controller_name in {"cause_rules", "fixed_model", "model_variants"}:
            continue
        result = run_episode(
            good_task,
            task_dir / "controllers" / f"{controller_name}.py",
            root=tmp_path,
            record_mcap=False,
        )
        if "fuzz_max_pass_rate" in expectation:
            assert result.verdict["passed"], result.verdict
            fuzz_result = fuzz(
                good_task,
                task_dir / "controllers" / f"{controller_name}.py",
                n=10,
                workers=1,
                seed=17,
                root=tmp_path,
            )
            assert fuzz_result["pass_rate"] <= expectation["fuzz_max_pass_rate"], fuzz_result
            continue
        assert not result.verdict["passed"], result.verdict
        first = result.verdict["first_violation"]
        assert first and first["spec"] == expectation["first_failing_spec"], result.verdict
        observed = set(result.verdict["events_summary"])
        assert set(expectation["must_events"]) <= observed, result.verdict

    for model_name, expectation in expectations.get("model_variants", {}).items():
        variant_task = copy.copy(task)
        variant_task.model_path = (task_dir / model_name).resolve()
        if expectation.get("check_errors") is not None:
            errors = [
                item for item in check_model(variant_task.model_path) if item["severity"] == "error"
            ]
            assert len(errors) == expectation["check_errors"], errors
        result = run_episode(
            variant_task, task_dir / "controllers/good.py", root=tmp_path, record_mcap=False
        )
        assert result.verdict["passed"] is expectation["passed"], result.verdict
