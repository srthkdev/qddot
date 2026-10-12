"""Discrete full-state LQR designed from MuJoCo's finite-difference transition model."""
import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = float(info["control_dt"])
        self.joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                       for n in ("shoulder", "elbow")]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.joints], dtype=int)
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in self.joints], dtype=int)
        self.nx, self.nu = 4, self.model.nu
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        a = np.zeros((self.nx, self.nx))
        b = np.zeros((self.nx, self.nu))
        mujoco.mjd_transitionFD(self.model, self.data, 1e-6, True, a, b, None, None)
        q = np.diag([35.0, 30.0, 3.0, 2.5])
        r = np.diag([0.10, 0.15])
        p = q.copy()
        for _ in range(800):
            gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
            next_p = q + a.T @ p @ a - a.T @ p @ b @ gain
            if np.max(np.abs(next_p - p)) < 1e-9:
                p = next_p
                break
            p = next_p
        self.gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        self.gravity = np.zeros(self.nu)

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        state = np.array([obs["joint.shoulder.pos"], obs["joint.elbow.pos"],
                          obs["joint.shoulder.vel"], obs["joint.elbow.vel"]])
        return dict(zip(("shoulder_motor", "elbow_motor"), (-self.gain @ state).tolist()))
