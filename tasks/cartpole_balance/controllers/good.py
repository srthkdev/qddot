"""LQR cart-pole stabilizer; linearizes MuJoCo dynamics and solves DARE at reset."""

import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = float(info["control_dt"])
        self.state_keys = ["joint.cart.pos", "joint.pole.pos", "joint.cart.vel", "joint.pole.vel"]
        self.cart = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart")
        self.pole = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole")
        qids = [self.model.jnt_qposadr[self.cart], self.model.jnt_qposadr[self.pole]]
        dofs = [self.model.jnt_dofadr[self.cart], self.model.jnt_dofadr[self.pole]]
        eps = 1e-5
        a = np.zeros((4, 4))
        b = np.zeros(4)
        for index in range(4):
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            if index < 2:
                self.data.qpos[qids[index]] = eps
            else:
                self.data.qvel[dofs[index - 2]] = eps
            mujoco.mj_forward(self.model, self.data)
            a[2:, index] = self.data.qacc[dofs] / eps
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[0] = eps
        mujoco.mj_forward(self.model, self.data)
        b[2:] = self.data.qacc[dofs] / eps
        a[0, 2] = 1.0
        a[1, 3] = 1.0
        ac = np.eye(4) + self.dt * a
        bc = self.dt * b[:, None]
        q = np.diag([3.0, 35.0, 1.0, 2.0])
        r = np.array([[0.5]])
        p = q.copy()
        for _ in range(500):
            gain = np.linalg.solve(r + bc.T @ p @ bc, bc.T @ p @ ac)
            next_p = q + ac.T @ p @ ac - ac.T @ p @ bc @ gain
            if np.max(np.abs(next_p - p)) < 1e-10:
                p = next_p
                break
            p = next_p
        self.gain = np.linalg.solve(r + bc.T @ p @ bc, bc.T @ p @ ac).ravel()

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        state = np.array([obs[key] for key in self.state_keys])
        return {"push": float(-self.gain @ state)}
