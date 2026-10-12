"""The physics backend interface.

Everything above the simulator (events, signals, specs, storage, diagnosis) needs
only what this protocol exposes, so adding an engine means implementing ~10
methods. MuJoCo is the reference implementation (`qddot.sim.backends.mujoco`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Limit:
    """A named joint or actuator with an optional [low, high] range."""

    name: str
    limited: bool
    low: float
    high: float


@dataclass(frozen=True)
class Contact:
    """One active contact, deduplicated per body pair by `key`."""

    key: tuple[Any, ...]
    a: str
    b: str
    geom_a: str
    geom_b: str
    normal_force: float
    slip_speed: float
    penetration: float


class Backend(Protocol):
    name: str
    version: str
    timestep: float
    joints: list[Limit]       # 1-DoF joints (hinge/slide), in model order
    actuators: list[Limit]

    def reset(self, init: dict[str, Any], friction_scale: float, mass_scale: float) -> None:
        """Load the initial state (keyframe, per-joint qpos) and apply physics scaling."""

    def ctrl(self) -> np.ndarray: ...
    def set_ctrl(self, ctrl: np.ndarray) -> None: ...
    def step(self) -> None: ...
    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions and velocities aligned with `joints`."""

    def actuator_force(self) -> np.ndarray: ...
    def position(self, name: str, kind: str | None = None) -> np.ndarray:
        """World position of a site or body (`kind` = "site" | "body" | None for either)."""

    def contacts(self) -> list[Contact]: ...
    def finite(self) -> bool: ...
    def native(self) -> tuple[Any, Any]:
        """Engine-specific handles passed to `on_step` hooks (MuJoCo: model, data)."""


BACKENDS = {
    "mujoco": "qddot.sim.backends.mujoco:MujocoBackend",
    "newton": "qddot.sim.backends.newton:NewtonBackend",
    "genesis": "qddot.sim.backends.genesis:GenesisBackend",
}


def load_backend(name: str, model_path: str) -> Backend:
    """Instantiate a registered backend for a model file."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(sorted(BACKENDS))}")
    module_name, cls_name = BACKENDS[name].split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"backend {name!r} needs an optional dependency: "
                          f"pip install 'qddot[{name}]' ({exc})") from exc
    return getattr(module, cls_name)(model_path)
