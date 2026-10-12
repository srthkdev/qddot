"""Bimanual ALOHA cube transfer with Cartesian DLS and timed waypoints."""

from __future__ import annotations

import mujoco
import numpy as np


class Controller:
    """Position-actuator controller for a right-to-left cube handover."""

    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.arms = {}
        self.actuators: dict[str, int] = {}
        for side in ("left", "right"):
            joint_names = [f"{side}/{joint}" for joint in
                           ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")]
            joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in joint_names]
            self.arms[side] = {
                "qpos": np.array([self.model.jnt_qposadr[jid] for jid in joint_ids]),
                "dof": np.array([self.model.jnt_dofadr[jid] for jid in joint_ids]),
                "range": self.model.jnt_range[joint_ids].copy(),
                "site": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/gripper"),
                "actuators": joint_names,
            }
        self.gripper_names = {"left": "left/gripper", "right": "right/gripper"}
        self.cube_home = np.array([0.22, -0.02, 0.024])
        self.waypoints = [
            (0.25, "right", np.array([0.22, -0.02, 0.18])),  # approach
            (0.75, "right", np.array([0.22, -0.02, 0.050])), # descend
            (1.05, "right", np.array([0.22, -0.02, 0.050])), # close
            (1.55, "right", np.array([0.22, -0.02, 0.19])),  # lift
            (2.45, "right", np.array([0.035, -0.02, 0.30])), # meet
            (2.45, "left", np.array([0.030, -0.02, 0.285])),
            (3.55, "left", np.array([0.035, -0.02, 0.30])), # transfer
            (4.35, "right", np.array([0.22, -0.02, 0.32])), # retreat right after release
        ]
        self.start_sites: dict[str, np.ndarray] = {}

    @staticmethod
    def _minimum_jerk(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def _cartesian_target(self, side: str, t: float) -> np.ndarray:
        if side not in self.start_sites:
            raise RuntimeError("reset must initialize the private arm model")
        current_goal = self.start_sites[side]
        prior_time = 0.0
        for end_time, waypoint_side, point in self.waypoints:
            if waypoint_side != side:
                continue
            segment_start = prior_time
            if t <= end_time:
                blend = self._minimum_jerk((t - segment_start) / max(end_time - segment_start, 1e-6))
                return current_goal + blend * (point - current_goal)
            current_goal = point
            prior_time = end_time
        return current_goal

    def _ik(self, side: str, observation: dict[str, float], target: np.ndarray) -> np.ndarray:
        arm = self.arms[side]
        qpos_ids, dof_ids = arm["qpos"], arm["dof"]
        for index, qpos_id in enumerate(qpos_ids):
            self.data.qpos[qpos_id] = observation[f"joint.{arm['actuators'][index]}.pos"]
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros_like(jacp)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, arm["site"])
        jacobian = jacp[:, dof_ids]
        error = target - self.data.site_xpos[arm["site"]]
        # A bounded DLS step avoids large joint jumps near singular arm postures.
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.018**2 * np.eye(3), error)
        q = self.data.qpos[qpos_ids].copy()
        return np.clip(q + np.clip(delta, -0.12, 0.12), arm["range"][:, 0] + 0.08,
                       arm["range"][:, 1] - 0.08)

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        output: dict[str, float] = {}
        for side, arm in self.arms.items():
            if side not in self.start_sites:
                # Initialize interpolation from the private model's measured site pose.
                for idx, qpos_id in enumerate(arm["qpos"]):
                    self.data.qpos[qpos_id] = obs[f"joint.{arm['actuators'][idx]}.pos"]
                mujoco.mj_forward(self.model, self.data)
                self.start_sites[side] = self.data.site_xpos[arm["site"]].copy()
            goal = self._cartesian_target(side, t)
            command = self._ik(side, obs, goal)
            for name, value in zip(arm["actuators"], command):
                output[name] = float(value)
        right_grip = 0.008 if t >= 0.94 else 0.037  # BUG: clamp margin is marginal at reduced friction
        left_grip = 0.004 if t >= 3.0 else 0.037
        if t >= 3.55:
            right_grip = 0.037
        output[self.gripper_names["right"]] = right_grip
        output[self.gripper_names["left"]] = left_grip
        return output
