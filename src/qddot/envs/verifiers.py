"""verifiers environment (prime-rl / Environments Hub): write a controller, get physics feedback, retry."""

from __future__ import annotations

import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset

from qddot.envs.coding import evaluate_source

CODE = re.compile(r"```python\n(.*?)```", re.DOTALL)
SYSTEM = ("You write robot controllers. Reply with one ```python block defining `class Controller` "
          "with step(obs: dict, t: float) -> dict[actuator, value] (optional reset(info)). "
          "You will get the physics verdict back and may revise.")


def _last_code(messages) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            found = CODE.findall(message.get("content") or "")
            return found[-1] if found else None
    return None


class QddotEnv(vf.MultiTurnEnv):
    def __init__(self, n_fuzz: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_fuzz = n_fuzz

    async def env_response(self, messages, state, **kwargs):
        code = _last_code(messages)
        if code is None:
            return [{"role": "user", "content": "No ```python block found. Send the full controller."}]
        result = evaluate_source(state["info"]["task_path"], code, n_fuzz=self.n_fuzz)
        state["qddot_reward"] = result.reward
        return [{"role": "user", "content": result.text}]

    @vf.stop
    async def solved(self, state, **kwargs) -> bool:
        return state.get("qddot_reward", 0.0) >= 1.0


async def physics_reward(completion, info, state, **kwargs) -> float:
    """Grade the final controller fresh, independent of what the env observed mid-rollout."""
    code = _last_code(completion)
    return evaluate_source(info["task_path"], code).reward if code else 0.0


def load_environment(tasks_dir: str = "tasks", max_turns: int = 5, n_fuzz: int = 10) -> vf.Environment:
    rows = []
    for task_yaml in sorted(Path(tasks_dir).glob("*/task.yaml")):
        prompt = task_yaml.parent / "PROMPT.md"
        question = (prompt.read_text() if prompt.is_file() else task_yaml.parent.name)
        starter = task_yaml.parent / "controllers" / "starter.py"
        context = f"\n\ntask.yaml:\n{task_yaml.read_text()}" + (
            f"\n\nstarter controller:\n```python\n{starter.read_text()}```" if starter.is_file() else "")
        rows.append({"question": question + context, "info": {"task_path": str(task_yaml.resolve())}})
    return QddotEnv(dataset=Dataset.from_list(rows), system_prompt=SYSTEM, max_turns=max_turns,
                    n_fuzz=n_fuzz, rubric=vf.Rubric(funcs=[physics_reward]))
