from typing import ClassVar

"""Minimum-jerk direct joint interpolation with a deliberately omitted route check."""

import numpy as np


class Controller:
    joints: ClassVar[list[str]] = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    actuators: ClassVar[list[str]] = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    start = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    def reset(self, info: dict) -> None:
        self.goal = np.array([float(info["target"][name]) for name in self.joints])

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        phase = float(np.clip(t / 4.8, 0.0, 1.0))
        blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        # BUG: interpolate straight between poses without checking link clearance.
        command = self.start + blend * (self.goal - self.start)
        return dict(zip(self.actuators, command.tolist()))
