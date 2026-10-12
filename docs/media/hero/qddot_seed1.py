"""Position controller for the ALOHA cube handover.

The episode exposes joint positions/velocities, actuator controls/forces, cube
position, and cube-to-gripper distances.  The controller commands the seven
position actuators on each arm: waist, shoulder, elbow, forearm_roll,
wrist_angle, wrist_rotate, and gripper.  ``reset(info)`` receives actuator and
joint metadata, control_dt, target, and model_path.
"""


class Controller:
    # Arm configurations from the neutral keyframe, solved for the gripper
    # site locations in the task's MuJoCo model.
    _neutral = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
    _right_above = [0.00392, -0.82308, 1.42534, -0.00076, -0.17976, 0.00004]
    _right_pick = [0.00394, -0.33537, 1.33213, -0.00076, -0.13981, 0.00004]
    _right_lift = [0.00391, -1.04266, 1.39685, -0.00076, -0.19006, 0.00004]
    _right_handover = [0.00252, -0.35126, 0.93309, -0.00052, -0.2155, 0.00002]
    _right_retreat = [0.00392, -1.1455, 1.2637, -0.00068, -0.18446, 0.00004]
    _left_approach = [-0.00185, 0.14001, 0.52349, 0.0005, -0.31501, -0.00002]

    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]

    @staticmethod
    def _blend(a, b, u):
        # Smoothstep avoids abrupt changes to these relatively heavy arms.
        u = max(0.0, min(1.0, u))
        u = u * u * (3.0 - 2.0 * u)
        return [x + (y - x) * u for x, y in zip(a, b)]

    def _arm_pose(self, side, t):
        if side == "right":
            if t < 0.45:
                return self._blend(self._neutral, self._right_above, t / 0.45)
            if t < 0.9:
                return self._blend(self._right_above, self._right_pick, (t - 0.45) / 0.45)
            if t < 1.1:
                return self._right_pick
            if t < 1.65:
                return self._blend(self._right_pick, self._right_lift, (t - 1.1) / 0.55)
            if t < 2.45:
                return self._blend(self._right_lift, self._right_handover, (t - 1.65) / 0.8)
            if t < 3.55:
                return self._right_handover
            return self._blend(self._right_handover, self._right_retreat, (t - 3.55) / 0.65)

        if t < 1.65:
            return self._neutral
        if t < 2.65:
            return self._blend(self._neutral, self._left_approach, (t - 1.65) / 1.0)
        return self._left_approach

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        out = {}
        for side in ("left", "right"):
            pose = self._arm_pose(side, t)
            for name, value in zip(
                ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
                pose,
            ):
                out[f"{side}/{name}"] = value

        # The right hand closes around the cube after lowering. The left hand
        # closes only after reaching the shared pose, and the right hand opens
        # after the overlap has had time to establish contact.
        # Keep a small finger opening so the opposing finger pads bear on the
        # cube instead of contacting one another.
        out["right/gripper"] = 0.018 if 0.82 <= t < 3.4 else 0.037
        out["left/gripper"] = 0.012 if t >= 2.65 else 0.037
        return out
