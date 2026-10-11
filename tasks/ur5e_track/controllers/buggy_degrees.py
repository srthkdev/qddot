"""Trajectory controller that mistakenly treats radian targets as degrees."""

import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.names = ["shoulder_pan", "shoulder_lift", "elbow"]
        # BUG: radian-valued task targets are converted as though they were degrees.
        self.goals = np.deg2rad([-1.20, -1.35, 1.25])
        self.starts = np.array([-1.5708, -1.5708, 1.5708])

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        phase = float(np.clip(t / 2.0, 0.0, 1.0))
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        values = self.starts + blend * (self.goals - self.starts)
        return dict(zip(self.names, values.tolist()))
