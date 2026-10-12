"""Cartesian waypoint controller for the ALOHA cube handover.

Observations contain ``t``; ``joint.<name>.pos`` and ``joint.<name>.vel`` for
all 16 joints; ``act.<name>.ctrl`` and ``act.<name>.force`` for all 14
actuators; and ``cube_x``, ``cube_y``, ``cube_z``, ``cube_right_pinch`` and
``cube_left_pinch``.  The position actuators are left/right ``waist``,
``shoulder``, ``elbow``, ``forearm_roll``, ``wrist_angle``, ``wrist_rotate``
and ``gripper``.  ``reset(info)`` receives ``actuators``, ``joints``,
``control_dt``, ``target`` and ``model_path``.  The controller targets the
gripper sites with damped least-squares inverse kinematics.
"""

import numpy as np
import mujoco


class Controller:
    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]
        model_path = info["model_path"]
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.arm_joints = {}
        self.arm_dofs = {}
        self.arm_sites = {}
        self.arm_qpos = {}
        for side in ("left", "right"):
            names = [f"{side}/{n}" for n in
                     ("waist", "shoulder", "elbow", "forearm_roll",
                      "wrist_angle", "wrist_rotate")]
            self.arm_joints[side] = names
            self.arm_qpos[side] = [int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                for n in names]
            self.arm_dofs[side] = [int(self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                for n in names]
            self.arm_sites[side] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/gripper")
            self._jacp = np.zeros((3, self.model.nv))
            self._jacr = np.zeros((3, self.model.nv))

    @staticmethod
    def _blend(a, b, u):
        u = float(np.clip(u, 0.0, 1.0))
        u = u * u * (3.0 - 2.0 * u)
        return (1.0 - u) * np.asarray(a) + u * np.asarray(b)

    def _cartesian(self, side, q, target):
        """Solve a small position-only IK problem, seeded from measured joints."""
        qadr = self.arm_qpos[side]
        dofs = self.arm_dofs[side]
        site = self.arm_sites[side]
        self.data.qpos[qadr] = q
        for _ in range(18):
            mujoco.mj_forward(self.model, self.data)
            err = np.asarray(target) - self.data.site_xpos[site]
            if np.linalg.norm(err) < 0.002:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jacp,
                              self._jacr, site)
            jac = self._jacp[:, dofs]
            # Damped least squares is stable near a straight elbow and also
            # keeps the redundant wrist joints close to their measured pose.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.012 * np.eye(3), err)
            dq = np.clip(dq, -0.12, 0.12)
            q = q + dq
            for i, name in enumerate(self.arm_joints[side]):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                lo, hi = self.model.jnt_range[jid]
                if self.model.jnt_limited[jid]:
                    q[i] = np.clip(q[i], lo + 0.005, hi - 0.005)
        return q

    def _right_target(self, t):
        # Pick from the table, lift, then carry into the shared workspace.
        if t < 0.35:
            return np.array([0.22, -0.02, 0.125])
        if t < 0.78:
            return self._blend([0.22, -0.02, 0.125],
                               [0.22, -0.02, 0.05], (t - 0.35) / 0.43)
        if t < 1.10:
            return np.array([0.22, -0.02, 0.05])
        if t < 1.55:
            return self._blend([0.22, -0.02, 0.05],
                               [0.22, -0.02, 0.17], (t - 1.10) / 0.45)
        if t < 2.65:
            return self._blend([0.22, -0.02, 0.17],
                               [0.22, -0.02, 0.17], (t - 1.55) / 1.10)
        if t < 3.85:
            return np.array([0.22, -0.02, 0.17])
        return self._blend([0.22, -0.02, 0.17],
                           [0.42, -0.02, 0.30], (t - 3.85) / 0.65)

    def _left_target(self, t):
        # Wait until the right hand has lifted the cube, then meet it from the
        # other side with a small separation between the two gripper bodies.
        if t < 1.65:
            return np.array([-0.188, -0.019, 0.325])
        if t < 3.10:
            return self._blend([-0.188, -0.019, 0.325],
                               [0.155, -0.02, 0.17], (t - 1.65) / 1.45)
        return np.array([0.155, -0.02, 0.17])

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        out = {}
        for side in ("left", "right"):
            q = np.array([obs[f"joint.{name}.pos"]
                          for name in self.arm_joints[side]], dtype=float)
            target = self._left_target(t) if side == "left" else self._right_target(t)
            qcmd = self._cartesian(side, q, target)
            for name, value in zip(self.arm_joints[side], qcmd):
                out[name] = float(value)

        # Fingers open for approach, close around the cube, and only release
        # the right gripper after the left fingers have had time to close.
        right_grip = 0.035 if t < 0.72 or t >= 3.85 else 0.002
        left_grip = 0.035 if t < 3.05 else 0.002
        out["right/gripper"] = right_grip
        out["left/gripper"] = left_grip
        return out
