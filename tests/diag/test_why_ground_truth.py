"""Evaluate diagnosis rules against the documented built-in controller failures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qddot.diag import why
from qddot.sim.runner import run_episode
from qddot.task import load_task

ROOT = Path(__file__).resolve().parents[2]
TASKS = [ROOT / "tasks" / name / "task.yaml"
         for name in ("cartpole_balance", "so101_reach", "ur5e_track")]


@pytest.mark.parametrize("task_file", TASKS, ids=lambda path: path.parent.name)
def test_buggy_controllers_match_documented_cause(task_file: Path, tmp_path: Path) -> None:
    task_dir = task_file.parent
    task = load_task(task_file)
    expected = yaml.safe_load((task_dir / "expected.yaml").read_text())
    causes = [(name, config) for name, config in expected.items()
              if name != "cause_rules" and name.startswith("buggy_")]
    correct = 0
    for name, config in causes:
        episode = run_episode(task, task_dir / "controllers" / f"{name}.py",
                              root=tmp_path / "runs", record_mcap=False)
        diagnosis = why(episode.episode_dir)
        assert diagnosis["verdict"] == "FAIL"
        assert diagnosis["likely_causes"], f"no cause for {task.name}/{name}"
        evidence = [line for cause in diagnosis["likely_causes"] for line in cause["evidence"]]
        assert len(evidence) == len(set(evidence))
        rule = expected["cause_rules"][config["cause"]]
        if diagnosis["likely_causes"][0]["rule"] == rule:
            correct += 1
    assert correct / len(causes) >= 0.8, f"{task.name}: {correct}/{len(causes)}"


@pytest.mark.parametrize("task_file", TASKS, ids=lambda path: path.parent.name)
def test_good_controller_passes_without_causes(task_file: Path, tmp_path: Path) -> None:
    task = load_task(task_file)
    episode = run_episode(task, task_file.parent / "controllers" / "good.py",
                          root=tmp_path / "runs", record_mcap=False)
    diagnosis = why(episode.episode_dir)
    assert diagnosis["verdict"] == "PASS"
    assert diagnosis["likely_causes"] == []
