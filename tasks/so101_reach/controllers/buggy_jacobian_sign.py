"""A plausible sign error in Cartesian feedback."""

import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]
        self.qids = [self.model.jnt_qposadr[j] for j in joints]
        self.dofs = [self.model.jnt_dofadr[j] for j in joints]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        for name, qid in zip(self.names, self.qids):
            self.data.qpos[qid] = obs[f"joint.{name}.pos"]
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, np.zeros_like(jacp), self.site)
        error = np.array([0.16, 0.02, 0.20]) - self.data.site_xpos[self.site]
        jac = jacp[:, self.dofs]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.015**2 * np.eye(3), error)
        # BUG: the Cartesian correction is negated, so the arm moves away from the target.
        q = self.data.qpos[self.qids] - 0.55 * dq
        return dict(zip(self.names, q.tolist()))
