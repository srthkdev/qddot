from __future__ import annotations

from pathlib import Path

from qddot.diag.replay import _resimulate, visual_diff
from qddot.sim.runner import run_episode
from qddot.task import load_task


def test_recorded_control_replay_restores_states_and_reports_rotations(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    task = load_task(root / "tasks/cartpole_balance/task.yaml")
    episode = run_episode(task, root / "tasks/cartpole_balance/controllers/buggy_velocity_sign.py",
                          root=tmp_path, record_mcap=False)

    _, captures = _resimulate(episode.episode_dir, [0.0, 0.1, 2.0])
    start = captures[0.1]
    later = captures[2.0]
    assert abs(float(start["state"][1]) - float(later["state"][1])) > 1.0
    assert max(item["max_state_error"] for item in captures.values()) < 1e-8

    diffs = visual_diff(episode.episode_dir, [0.1, 2.0])
    assert {item["baseline"] for item in diffs} == {"window_start", "t=0"}
    assert any("pole" in {body["body"] for body in item["rotated_bodies"]} for item in diffs)
