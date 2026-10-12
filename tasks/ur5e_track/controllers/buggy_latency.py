"""Minimum-jerk joint-space trajectory using the UR5e position actuators."""

import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.dt = float(info["control_dt"])
        self.start = {
            "shoulder_pan_joint": -1.5708,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
        }
        self.target = dict(info.get("target", {}))
        self.actuator = {
            "shoulder_pan_joint": "shoulder_pan",
            "shoulder_lift_joint": "shoulder_lift",
            "elbow_joint": "elbow",
        }

    def _nominal_step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        phase = float(np.clip(t / 2.0, 0.0, 1.0))
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        return {
            self.actuator[joint]: self.start[joint] + blend * (goal - self.start[joint])
            for joint, goal in self.target.items()
        }

    def step(self, obs, t):
        age = max(0.0, float(t) - float(obs.get("t", t)))
        # BUG: interprets sensor age in milliseconds as if it were seconds.
        return self._nominal_step(obs, t - (3000.0 * age if age > 0.005 else 0.0))
