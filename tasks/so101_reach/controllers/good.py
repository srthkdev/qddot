"""Damped least-squares Cartesian position controller for the SO-101."""

import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.qpos_ids = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names]
        self.dof_ids = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names]
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        target = info.get("target", {})
        self.target = np.array([target.get(axis, default) for axis, default in zip("xyz", (0.16, 0.02, 0.20))], dtype=float)
        self.actuators = self.joint_names

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        for name, qpos_id in zip(self.joint_names, self.qpos_ids):
            self.data.qpos[qpos_id] = obs[f"joint.{name}.pos"]
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        error = self.target - self.data.site_xpos[self.site_id]
        jac = jacp[:, self.dof_ids]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.015**2 * np.eye(3), error)
        q = self.data.qpos[self.qpos_ids]
        limits = np.array([self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joint_names])
        # Leave enough clearance for position-actuator tracking overshoot so the
        # simulated joint never enters the runner's 1% limit-warning zone.
        command = np.clip(q + 0.55 * dq, limits[:, 0] + 0.20, limits[:, 1] - 0.20)
        return dict(zip(self.actuators, command.tolist()))
