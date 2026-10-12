"""Discrete LQR stabilizer for the two-link pendulum."""
import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = float(info["control_dt"])
        self.joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("shoulder", "elbow")]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.joints], dtype=int)
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in self.joints], dtype=int)
        self.model.jnt_limited[self.joints] = False  # BUG: remove stops so the model can be linearized at hanging equilibrium.
        self.data.qpos[self.qadr] = [np.pi, 0.0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        a = np.zeros((4, 4)); b = np.zeros((4, self.model.nu))
        mujoco.mjd_transitionFD(self.model, self.data, 1e-6, True, a, b, None, None)
        q = np.diag([35.0, 30.0, 3.0, 2.5]); r = np.diag([0.10, 0.15]); p = q.copy()
        for _ in range(800):
            gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
            pn = q + a.T @ p @ a - a.T @ p @ b @ gain
            if np.max(np.abs(pn-p)) < 1e-9: p=pn; break
            p=pn
        self.gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        state = np.array([obs["joint.shoulder.pos"], obs["joint.elbow.pos"], obs["joint.shoulder.vel"], obs["joint.elbow.vel"]])
        return dict(zip(("shoulder_motor", "elbow_motor"), (-self.gain @ state).tolist()))
