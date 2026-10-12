"""Joint-space controller for the ALOHA cube handover.

Observations include t; joint.<name>.pos and joint.<name>.vel for all 16
joints; act.<name>.ctrl and act.<name>.force for 14 actuators; cube_x,
cube_y, cube_z, cube_right_pinch and cube_left_pinch. Actuator names are
left/{waist,shoulder,elbow,forearm_roll,wrist_angle,wrist_rotate,gripper}
and the same seven right/ names. reset(info) receives actuators, joints,
control_dt, target, and model_path.
"""


class Controller:
    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]
        self.names = ("waist", "shoulder", "elbow", "forearm_roll",
                      "wrist_angle", "wrist_rotate")
        self.handover = False
        self.handover_t = 99.0
        # IK solutions for sites above the cube, at the pickup pose, and for
        # each arm's side of the shared workspace. Values are radians.
        self.r = [
            [0.004, -0.829, 1.4014, -0.0008, -0.1137, 0.0],
            [0.004, -0.2238, 1.2875, -0.0008, -0.1021, 0.0],
            [0.0039, -1.0118, 1.4170, -0.0008, -0.2270, 0.0],
        ]
        self.l = [
            [-0.0014, 0.5263, -0.2916, 0.0005, -0.4678, 0.0],
            [-0.0014, 0.6510, -0.1407, 0.0005, -0.3342, 0.0],
            [-0.0014, 0.6022, -0.1551, 0.0005, -0.4080, 0.0],
        ]

    @staticmethod
    def blend(a, b, u):
        u = max(0.0, min(1.0, u))
        u = u * u * (3.0 - 2.0 * u)
        return [x + (y - x) * u for x, y in zip(a, b)]

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        if t > 4.0 and not self.handover and obs.get("cube_left_pinch", 1.0) < 0.060:
            self.handover = True
            self.handover_t = t
        # Right arm approaches from above, closes around the cube, and lifts.
        if t < 0.65:
            r = self.blend([0, -0.96, 1.16, 0, -0.3, 0], self.r[0], t / 0.65)
        elif t < 1.05:
            r = self.blend(self.r[0], self.r[1], (t - 0.65) / 0.4)
        elif t < 1.3:
            r = self.r[1]
        elif not self.handover:
            r = self.blend(self.r[1], self.r[2], (t - 1.3) / 0.65)
        else:
            r = self.blend(self.r[2], [0, -0.96, 1.16, 0, -0.3, 0], (t - 4.0) / 0.6)
        r = list(r)

        # Left arm waits clear of the pickup, then comes to the far side of
        # the cube for the overlap. The right gripper releases only after the
        # left gripper has reached the cube.
        if t < 2.15:
            l = [0, -0.96, 1.16, 0, -0.3, 0]
        elif t < 3.1:
            l = self.blend([0, -0.96, 1.16, 0, -0.3, 0], self.l[0], (t - 2.15) / 0.95)
        elif t < 4.0 or not self.handover:
            l = self.blend(self.l[0], self.l[1], (t - 3.1) / 0.9)
        elif t < self.handover_t + 0.18:
            l = list(self.l[1])
        else:
            l = self.blend(self.l[1], self.l[2], (t - self.handover_t - 0.18) / 0.6)

        if t < 0.55:
            rg = 0.032
        elif t < 0.95:
            rg = 0.032
        elif t < 3.65:
            rg = 0.020
        else:
            rg = 0.032 if self.handover else 0.020

        # Open left fingers until the final approach, then close and keep
        # holding. Mirrored finger joints are tied in the model.
        lg = 0.020 if self.handover else 0.032
        out = {}
        for arm, q, grip in (("right", r, rg), ("left", l, lg)):
            for name, value in zip(self.names, q):
                out[f"{arm}/{name}"] = value
            out[f"{arm}/gripper"] = grip
        return out
