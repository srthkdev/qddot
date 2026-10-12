"""OpenEnv environment: the agent submits controller source and gets a physics verdict back."""

from __future__ import annotations

import uuid
from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from qddot.envs.coding import evaluate_source
from qddot.task import load_task


class ControllerAction(Action):
    code: str = Field(description="Python source defining `class Controller` with step(obs, t)")


class VerdictObservation(Observation):
    feedback: str = Field(default="", description="Verdict, failing spec and diagnosis")
    task_prompt: str = Field(default="", description="The task description (on reset)")
    passed: bool = False
    fuzz_pass_rate: float | None = None


class CodingEnv(Environment[ControllerAction, VerdictObservation, State]):
    """Multi-turn: each step grades a controller; the episode ends on success or after `max_turns`."""

    SUPPORTS_CONCURRENT_SESSIONS = True  # every evaluation runs in its own temp directory

    def __init__(self, task_path: str | Path, max_turns: int = 5, n_fuzz: int = 10) -> None:
        super().__init__()
        self.task_path, self.max_turns, self.n_fuzz = Path(task_path), max_turns, n_fuzz
        self.task = load_task(self.task_path)
        self._state = State()

    def reset(self, seed=None, episode_id=None, **kwargs) -> VerdictObservation:
        self._state = State(episode_id=episode_id or uuid.uuid4().hex, step_count=0)
        prompt = self.task_path.parent / "PROMPT.md"
        text = prompt.read_text() if prompt.is_file() else self.task.description
        return VerdictObservation(task_prompt=f"{text}\n\ntask.yaml:\n{self.task_path.read_text()}")

    def step(self, action: ControllerAction, timeout_s=None, **kwargs) -> VerdictObservation:
        self._state.step_count += 1
        result = evaluate_source(self.task, action.code, n_fuzz=self.n_fuzz)
        solved = result.passed and (result.fuzz_pass_rate or 0.0) >= 0.9
        return VerdictObservation(feedback=result.text, passed=result.passed,
                                  fuzz_pass_rate=result.fuzz_pass_rate, reward=result.reward,
                                  done=solved or self._state.step_count >= self.max_turns)

    @property
    def state(self) -> State:
        return self._state
