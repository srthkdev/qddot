"""Gymnasium environment: a policy commands the actuators every control step."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np

from qddot.sim.runner import Episode
from qddot.task import Task, load_task


class ControlEnv(gym.Env):
    """Observations are the task's signals; actions are actuator commands (clipped to ctrlrange).

    The reward is sparse: at the end of the episode it is the verdict's spec
    robustness (positive means every spec holds). `info["verdict"]` has the full result.
    Set `perturb=True` to sample the task's fuzz perturbations on every reset.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, task: Task | str | Path, perturb: bool = False) -> None:
        self.task = task if isinstance(task, Task) else load_task(task)
        self.perturb = perturb
        probe = Episode(self.task)
        self.obs_keys = sorted(k for k in probe.observe() if k != "t")
        self.actuators = [a.name for a in probe.sim.actuators]
        low = np.array([a.low if a.limited else -np.inf for a in probe.sim.actuators], dtype=np.float64)
        high = np.array([a.high if a.limited else np.inf for a in probe.sim.actuators], dtype=np.float64)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(self.obs_keys),), dtype=np.float64)
        self.episode: Episode | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        params = None
        if self.perturb:
            from qddot.sim.perturb import sample

            params = sample(self.task, self.np_random)
        self.episode = Episode(self.task, seed=0 if seed is None else seed, perturb=params)
        return self._obs(self.episode.observe()), {}

    def step(self, action):
        episode = self.episode
        episode.act(dict(zip(self.actuators, np.asarray(action, dtype=np.float64).tolist(), strict=True)))
        if episode.done:
            verdict = episode.verdict()
            reward = float(verdict["robustness"] or 0.0) if not verdict["errors"] else -1.0
            return self._obs(episode.history[-1]), reward, True, False, {"verdict": verdict}
        return self._obs(episode.observe()), 0.0, False, False, {}

    def _obs(self, obs: dict[str, float]) -> np.ndarray:
        return np.array([obs[k] for k in self.obs_keys], dtype=np.float64)
