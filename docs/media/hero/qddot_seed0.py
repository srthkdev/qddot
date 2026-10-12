"""Cartesian waypoint controller for transferring the cube between ALOHA arms."""

import numpy as np
import mujoco


class Controller:
    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]
        path = info["model_path"]
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.arm_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        self.joint_ids = {}
        self.qadr = {}
        self.dadr = {}
        self.sites = {}
        for side in ("left", "right"):
            self.joint_ids[side] = []
            self.qadr[side] = []
            self.dadr[side] = []
            for name in self.arm_names:
                full = side + "/" + name
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, full)
                self.joint_ids[side].append(jid)
                self.qadr[side].append(self.model.jnt_qposadr[jid])
                self.dadr[side].append(self.model.jnt_dofadr[jid])
            self.sites[side] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, side + "/gripper")
        self.limits = {}
        for side in ("left", "right"):
            self.limits[side] = np.array([self.model.jnt_range[j] for j in self.joint_ids[side]])

    def _ik(self, side, target, obs):
        # Start from measured joints to tolerate actuator lag and perturbed observations.
        for s in ("left", "right"):
            for name, qa in zip(self.arm_names, self.qadr[s]):
                self.data.qpos[qa] = obs.get("joint." + s + "/" + name + ".pos", 0.0)
        qadr = self.qadr[side]
        dadr = self.dadr[side]
        site = self.sites[side]
        lo, hi = self.limits[side][:, 0], self.limits[side][:, 1]
        q = np.array([self.data.qpos[i] for i in qadr])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(12):
            for i, value in zip(qadr, q):
                self.data.qpos[i] = value
            mujoco.mj_forward(self.model, self.data)
            err = target - self.data.site_xpos[site]
            if np.linalg.norm(err) < 0.004:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
            J = jacp[:, dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.0025 * np.eye(3), err)
            q = np.clip(q + np.clip(dq, -0.12, 0.12), lo + 0.005, hi - 0.005)
        return q

    @staticmethod
    def _smooth(t, a, b):
        x = np.clip((t - a) / (b - a), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _blend(self, t, a, b, p0, p1):
        s = self._smooth(t, a, b)
        return (1.0 - s) * np.array(p0) + s * np.array(p1)

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        cube = np.array([obs["cube_x"], obs["cube_y"], obs["cube_z"]])
        # The right hand lifts vertically first, then presents the cube at the center.
        if t < 0.65:
            rtarget = np.array([0.22, -0.02, 0.115])
        elif t < 1.05:
            rtarget = self._blend(t, 0.65, 1.05, [0.22, -0.02, 0.115], [0.22, -0.02, 0.025])
        elif t < 1.35:
            rtarget = np.array([0.22, -0.02, 0.025])
        elif t < 1.85:
            rtarget = self._blend(t, 1.35, 1.85, [0.22, -0.02, 0.025], [0.20, -0.02, 0.22])
        elif t < 2.65:
            rtarget = self._blend(t, 1.85, 2.65, [0.20, -0.02, 0.22], [0.08, -0.02, 0.32])
        elif t < 3.9:
            rtarget = np.array([0.08, -0.02, 0.32])
        else:
            rtarget = self._blend(t, 3.9, 4.4, [0.08, -0.02, 0.32], [0.38, -0.02, 0.34])

        # The left gripper enters after the cube is clear of the table. It closes
        # before the right gripper releases, then carries the cube to the goal.
        if t < 2.5:
            ltarget = np.array([-0.18, -0.02, 0.32])
        elif t < 3.45:
            ltarget = self._blend(t, 2.5, 3.45, [-0.18, -0.02, 0.32], [0.03, -0.02, 0.32])
        elif t < 4.25:
            # Track the measured cube center while the two grippers exchange it.
            ltarget = cube.copy()
        else:
            ltarget = self._blend(t, 4.25, 5.05, cube, [0.22, -0.02, 0.18])

        targets = {}
        targets["right"] = self._ik("right", rtarget, obs)
        targets["left"] = self._ik("left", ltarget, obs)
        result = {}
        for side in ("left", "right"):
            for name, value in zip(self.arm_names, targets[side]):
                result[side + "/" + name] = float(value)
        # Turn the incoming left fingers across the right gripper's finger plane.
        result["left/wrist_rotate"] = 1.45 if t >= 2.5 else float(targets["left"][5])
        # Position actuator commands are symmetric; 0.002 is closed, 0.037 open.
        result["right/gripper"] = 0.037 if t >= 4.05 else (0.008 if t >= 1.12 else 0.037)
        result["left/gripper"] = 0.015 if t >= 3.75 else 0.037
        return result
