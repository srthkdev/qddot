"""Clearance checked waypoint controller with an over-scaled age predictor."""

from typing import ClassVar

import mujoco
import numpy as np


class Controller:
    joints: ClassVar[list[str]] = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    actuators: ClassVar[list[str]] = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    via = np.array([-2.36356, -0.59985, 1.95469, -1.59736, -0.01485, 0.0])

    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.goal = np.array([float(info["target"][name]) for name in self.joints])

    @staticmethod
    def blend(phase: float) -> float:
        u = float(np.clip(phase, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        age = max(0.0, float(t) - float(obs.get("t", t)))
        if t <= 2.4:
            phase = t / 2.4
            nominal = self.home + self.blend(phase) * (self.via - self.home)
        else:
            phase = (t - 2.4) / 2.4
            nominal = self.via + self.blend(phase) * (self.goal - self.via)
        q = np.array([obs[f"joint.{name}.pos"] - 100.0 * age * obs[f"joint.{name}.vel"] for name in self.joints])
        # BUG: scale sensor age as too many controller ticks during joint prediction.
        command = nominal + 10.0 * (nominal - q)
        return dict(zip(self.actuators, command.tolist()))
