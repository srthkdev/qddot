"""Cartesian waypoint controller for handing the cube from right to left.

Observations include t; joint.<name>.pos and joint.<name>.vel for all 16
joints; act.<name>.ctrl and act.<name>.force for 14 actuators; cube_x,
cube_y, cube_z, cube_right_pinch and cube_left_pinch. Actuator names are
left/{waist,shoulder,elbow,forearm_roll,wrist_angle,wrist_rotate,gripper}
and the same seven right/ names. reset(info) receives actuators, joints,
control_dt, target, and model_path.
"""

import numpy as np
import mujoco


class Controller:
    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.names = ("waist", "shoulder", "elbow", "forearm_roll",
                      "wrist_angle", "wrist_rotate")
        self.jids = {}
        self.qadr = {}
        self.actids = {}
        for side in ("left", "right"):
            for name in self.names:
                full = f"{side}/{name}"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full)
                self.jids[full] = jid
                self.qadr[full] = int(self.model.jnt_qposadr[jid])
                self.actids[full] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
            self.actids[f"{side}/gripper"] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}/gripper")
        self.siteids = {
            s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}/gripper")
            for s in ("left", "right")
        }
        self.limits = {key: self.model.actuator_ctrlrange[aid].copy()
                       for key, aid in self.actids.items() if key.endswith(tuple(self.names))}
        self.seed = {}

    @staticmethod
    def blend(a, b, u):
        u = float(np.clip(u, 0.0, 1.0))
        u = u * u * (3.0 - 2.0 * u)
        return (1.0 - u) * np.asarray(a) + u * np.asarray(b)

    def waypoint(self, side, t):
        # Coordinates are gripper-site centers. The cube starts at (0.22,-.02,.024).
        if side == "right":
            seq = [
                (0.00, (0.188, -0.019, 0.325)),
                (0.52, (0.220, -0.020, 0.100)),
                (0.78, (0.220, -0.020, 0.046)),
                (1.28, (0.220, -0.020, 0.046)),
                (1.72, (0.220, -0.020, 0.205)),
                (2.55, (0.065, -0.020, 0.179)),
                (4.25, (0.065, -0.020, 0.179)),
                (4.75, (0.225, -0.020, 0.270)),
                (6.00, (0.225, -0.020, 0.270)),
            ]
        else:
            seq = [
                (0.00, (-0.188, -0.019, 0.325)),
                (1.60, (-0.188, -0.019, 0.325)),
                (2.30, (0.058, -0.020, 0.179)),
                (3.28, (0.058, -0.020, 0.179)),
                (4.50, (0.058, -0.020, 0.179)),
                (5.15, (0.080, -0.020, 0.275)),
                (6.00, (0.080, -0.020, 0.275)),
            ]
        for i in range(len(seq) - 1):
            ta, a = seq[i]
            tb, b = seq[i + 1]
            if t <= tb:
                return self.blend(a, b, (t - ta) / (tb - ta))
        return np.asarray(seq[-1][1])

    def solve(self, side, target, obs):
        # Use the observed state as the IK seed, which also tracks the arm smoothly
        # through each waypoint despite small actuator lag and observation noise.
        for full, adr in self.qadr.items():
            self.data.qpos[adr] = obs.get(f"joint.{full}.pos", 0.0)
        mujoco.mj_forward(self.model, self.data)
        ids = [self.jids[f"{side}/{n}"] for n in self.names]
        qadr = [self.qadr[f"{side}/{n}"] for n in self.names]
        site = self.siteids[side]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        # Damped least squares, seeded from the current posture for continuity.
        for _ in range(24):
            mujoco.mj_forward(self.model, self.data)
            err = target - self.data.site_xpos[site]
            if np.linalg.norm(err) < 0.0015:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
            dofs = [int(self.model.jnt_dofadr[j]) for j in ids]
            j = jacp[:, dofs]
            dq = j.T @ np.linalg.solve(j @ j.T + 0.0015 * np.eye(3), err)
            dq = np.clip(dq, -0.10, 0.10)
            for k, adr in enumerate(qadr):
                self.data.qpos[adr] += dq[k]
                full = f"{side}/{self.names[k]}"
                lo, hi = self.limits[full]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr], lo, hi)
        out = {}
        for name, adr in zip(self.names, qadr):
            full = f"{side}/{name}"
            out[full] = float(np.clip(self.data.qpos[adr], *self.limits[full]))
        return out

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        cmd = {}
        for side in ("left", "right"):
            cmd.update(self.solve(side, self.waypoint(side, t), obs))

        # Keep the left hand open until it is around the cube; grip before the
        # right hand releases, then let the right fingers clear the object.
        cmd["right/gripper"] = 0.002 if 0.78 <= t < 3.60 else 0.037
        cmd["left/gripper"] = 0.002 if t >= 2.85 else 0.037
        return cmd
