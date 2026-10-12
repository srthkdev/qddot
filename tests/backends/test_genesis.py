"""Real task smoke test for the optional Genesis backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from qddot.sim.runner import run_episode
from qddot.task import load_task

ROOT = Path(__file__).resolve().parents[2]


def test_cartpole_good_controller_runs_on_genesis(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason="Genesis needs torch, which is not installed in this environment")
    pytest.importorskip("genesis")
    task = load_task(ROOT / "tasks/cartpole_balance/task.yaml")
    task.backend = "genesis"
    perturb = {
        "friction_scale": 1.0,
        "mass_scale": 1.0,
        "obs_noise_std": 0.0,
        "latency_ms": 0.0,
        "jitter_ms": 0.0,
        "controller_budget_ms": None,
    }
    result = run_episode(
        task,
        ROOT / "tests/backends/fixtures/cartpole_lqr.py",
        seed=0,
        perturb=perturb,
        root=tmp_path,
        record_mcap=False,
    )

    assert result.episode_dir.is_dir()
    assert result.verdict["stats"]["control_steps"] > 0
    status = "PASS" if result.verdict["passed"] else "FAIL"
    print(f"Genesis cartpole verdict: {status} ({result.verdict['episode']})")
    assert result.verdict["passed"], result.verdict
