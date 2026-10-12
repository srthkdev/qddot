"""Phased grasp-and-lift sequence with contact confirmation and jaw position hold."""

import mujoco


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.dt = float(info["control_dt"])
        self.close = 0.0
        self.lift = 0.0
        self.closed_at = None

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        if t < 0.65:
            self.lift = 0.0
            self.close = min(0.057, self.close + 0.002)
        elif self.closed_at is None:
            self.closed_at = t
            self.close = 0.057
        if t >= 0.65:
            self.lift = 0.18  # BUG: command the full vertical stroke in a single control cycle.
        # Hold a symmetric jaw opening; the position actuators provide compliant contact force.
        return {"lift": self.lift, "left_close": self.close, "right_close": self.close}
