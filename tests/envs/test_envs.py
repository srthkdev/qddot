from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qddot.envs.coding import evaluate_source

CARTPOLE = Path(__file__).resolve().parents[2] / "tasks" / "cartpole_balance"


def _source(name: str) -> str:
    return (CARTPOLE / "controllers" / f"{name}.py").read_text()


def test_coding_reward_orders_good_above_buggy(tmp_path: Path) -> None:
    good = evaluate_source(CARTPOLE / "task.yaml", _source("good"), n_fuzz=4, root=tmp_path / "g")
    bad = evaluate_source(CARTPOLE / "task.yaml", _source("buggy_velocity_sign"), root=tmp_path / "b")
    crash = evaluate_source(CARTPOLE / "task.yaml", "class Controller:\n    def step(self, o, t):\n        1/0\n",
                            root=tmp_path / "c")
    assert good.passed and good.reward == 1.0 and "fuzz: 4/4" in good.text
    assert not bad.passed and 0.0 <= bad.reward <= 0.25 and "first failing spec" in bad.text
    assert crash.reward == 0.0 and crash.text.startswith("ERROR: ZeroDivisionError")


def test_gymnasium_env_runs_a_policy_to_a_verdict() -> None:
    pytest.importorskip("gymnasium")
    from qddot.envs.gym import ControlEnv

    spec = importlib.util.spec_from_file_location("good", CARTPOLE / "controllers" / "good.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = ControlEnv(CARTPOLE / "task.yaml")
    policy = module.Controller()
    obs, _ = env.reset(seed=0)
    policy.reset({"actuators": env.actuators, "control_dt": env.task.control_dt, "target": env.task.target,
                  "model_path": str(env.task.model_path)})
    done, t, reward, info = False, 0.0, 0.0, {}
    while not done:
        command = policy.step(dict(zip(env.obs_keys, obs, strict=True)), t)
        obs, reward, done, _, info = env.step(np.array([command[name] for name in env.actuators]))
        t += env.task.control_dt
    assert env.action_space.shape == (1,) and env.observation_space.shape == obs.shape
    assert info["verdict"]["passed"] and reward > 0


def test_openenv_episode_ends_when_solved() -> None:
    pytest.importorskip("openenv")
    from qddot.envs.openenv import CodingEnv, ControllerAction

    env = CodingEnv(CARTPOLE / "task.yaml", max_turns=3, n_fuzz=4)
    first = env.reset()
    assert "task.yaml" in first.task_prompt and not first.done
    miss = env.step(ControllerAction(code=_source("buggy_velocity_sign")))
    assert not miss.done and not miss.passed and miss.reward < 0.5
    hit = env.step(ControllerAction(code=_source("good")))
    assert hit.done and hit.passed and hit.reward == 1.0 and env.state.step_count == 2


def test_verifiers_environment_builds_and_grades() -> None:
    pytest.importorskip("verifiers")
    import asyncio

    from qddot.envs.verifiers import load_environment, physics_reward

    env = load_environment(str(CARTPOLE.parent), n_fuzz=4)
    assert len(env.dataset) >= 15
    info = {"task_path": str(CARTPOLE / "task.yaml")}
    reply = [{"role": "assistant", "content": f"```python\n{_source('good')}```"}]
    assert asyncio.run(physics_reward(reply, info, {})) == 1.0
    assert asyncio.run(physics_reward([{"role": "assistant", "content": "no code"}], info, {})) == 0.0
