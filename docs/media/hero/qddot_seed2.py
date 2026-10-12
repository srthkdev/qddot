"""Timed joint-position controller for a right-to-left ALOHA cube handover.

The task observations include t, joint.<name>.pos/vel, act.<name>.ctrl/force,
cube_x/y/z, and cube_right_pinch/cube_left_pinch. The seven position actuators
per side are waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate,
and gripper. reset(info) receives actuators, joints, control_dt, target, and
model_path.
"""

class Controller:
    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]

    @staticmethod
    def blend(a, b, t):
        u = max(0.0, min(1.0, t))
        u = u * u * (3.0 - 2.0 * u)
        return [x + (y - x) * u for x, y in zip(a, b)]

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        # Six-joint IK waypoints (waist through wrist rotate), computed for the
        # neutral keyframe and the cube/grasp points in the scene coordinates.
        neutral = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
        r_above = [0.004, -0.828, 1.428, -0.001, -0.182, 0.0]
        r_grasp = [0.004, -0.422, 1.355, -0.001, -0.145, 0.0]
        r_lift = [0.003, -0.582, 1.210, -0.001, -0.210, 0.0]
        r_center = [0.003, -0.582, 1.210, -0.001, -0.210, 0.0]
        r_retreat = [0.30, -0.96, 1.16, 0.0, -0.30, 0.0]
        l_handover = [-0.002, 0.533, 0.133, 0.0, -0.340, 0.0]
        l_raise = [-0.002, 0.395, 0.119, 0.0, -0.361, 0.0]

        def trajectory(now, segments, fallback):
            for start, end, q0, q1 in segments:
                if now < start:
                    return q0
                if now <= end:
                    return self.blend(q0, q1, (now - start) / (end - start))
            return segments[-1][3] if segments else fallback

        # Right hand descends onto the cube, closes, lifts, and presents it.
        rq = trajectory(t, [
            (0.0, 0.55, neutral, r_above),
            (0.55, 1.05, r_above, r_grasp),
            (1.25, 1.85, r_grasp, r_lift),
            (1.85, 2.55, r_lift, r_center),
            (4.35, 4.95, r_center, r_retreat),
        ], r_lift)

        # Left hand approaches the held cube, closes for the handover, then
        # lifts while the right hand releases and retreats.
        lq = trajectory(t, [
            (2.15, 3.35, neutral, l_handover),
            (4.75, 5.35, l_handover, l_raise),
        ], neutral)

        out = {}
        for side, q in (("right", rq), ("left", lq)):
            for joint, value in zip(
                ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"), q
            ):
                out[f"{side}/{joint}"] = value

        # Keep both grippers open for approach; overlap the grasp only after
        # the right gripper has lifted and the left has reached the cube.
        out["right/gripper"] = 0.002 if 1.05 <= t < 4.35 else 0.037
        out["left/gripper"] = 0.020 if 3.45 <= t < 6.0 else 0.037
        return out
