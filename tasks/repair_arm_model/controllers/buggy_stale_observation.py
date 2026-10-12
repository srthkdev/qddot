"""Joint-space inverse-dynamics tracking for a deterministic minimum-jerk reach."""

import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = float(info["control_dt"])
        self.ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("shoulder_pan", "elbow_flex")
        ]
        self.qadr = self.model.jnt_qposadr[self.ids]
        self.dadr = self.model.jnt_dofadr[self.ids]
        self.act = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ("shoulder", "elbow")
        ]
        self.q0 = np.array([0.0, 0.0])
        self.qf = np.array([0.20, 0.80])
        self.kp = np.array([28.0, 18.0])
        self.kd = np.array([7.0, 5.0])
        self.history = []

    def step(
        self, obs: dict[str, float], t: float
    ) -> dict[str, float]:  # BUG: use a five-cycle-old joint sample.
        self.history.append(dict(obs))
        measured = self.history[max(0, len(self.history) - 6)]
        s = min(max(t / 1.25, 0.0), 1.0)
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        qref = self.q0 + (self.qf - self.q0) * blend
        dqref = (
            (self.qf - self.q0) * (30 * s**2 - 60 * s**3 + 30 * s**4) / 1.25
            if 0 < s < 1
            else np.zeros(2)
        )
        q = np.array([measured[f"joint.{n}.pos"] for n in ("shoulder_pan", "elbow_flex")])
        dq = np.array([measured[f"joint.{n}.vel"] for n in ("shoulder_pan", "elbow_flex")])
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.dadr] = dq
        mujoco.mj_forward(self.model, self.data)
        bias = self.data.qfrc_bias[self.dadr].copy()
        mass = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, mass)
        m = mass[np.ix_(self.dadr, self.dadr)]
        accel = 45 * (qref - q) + 12 * (dqref - dq)
        torque = bias + m @ accel
        return {"shoulder": float(torque[0]), "elbow": float(torque[1])}
