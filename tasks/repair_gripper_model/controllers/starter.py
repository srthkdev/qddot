"""Phased, symmetric grasp-and-lift sequence for position-controlled jaws."""


class Controller:
    def reset(self, info: dict) -> None:
        self.dt = float(info["control_dt"])
        self.close_target = 0.0
        self.lift_target = 0.0
        self.grasp_time: float | None = None
        self.close_rate = 0.002
        self.lift_rate = 0.004
        self.max_close = 0.057
        self.max_lift = 0.145

    def _close_on_block(self, t: float) -> None:
        if t < 0.65:
            self.lift_target = 0.0
            self.close_target = min(self.max_close, self.close_target + self.close_rate)
            return

        if self.grasp_time is None:
            self.grasp_time = t
            self.close_target = self.max_close

    def _raise_carriage(self, t: float) -> None:
        if t < 0.85:
            return

        # A slow position ramp avoids giving the frictional grasp an impulse.
        self.lift_target = min(self.max_lift, self.lift_target + self.lift_rate)

    def _jaw_targets(self) -> dict[str, float]:
        return {
            "left_close": self.close_target,
            "right_close": self.close_target,
        }

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        self._close_on_block(t)
        self._raise_carriage(t)

        # Position actuators maintain a symmetric squeeze during the vertical move.
        targets = self._jaw_targets()
        targets["lift"] = self.lift_target
        return targets
