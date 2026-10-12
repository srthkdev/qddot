"""Cartesian operational-space motion with damped least-squares IK."""
import mujoco
import numpy as np


class Controller:
    def reset(self, info):
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = info["control_dt"]
        self.names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.qids = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.names]
        self.dofs = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.names]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.target = info.get("target", {})
        self.last = np.zeros(3)

    def pose(self, xyz):
        self.data.qpos[self.qids] = self.last
        for _ in range(10):
            mujoco.mj_forward(self.model, self.data)
            jacp = np.zeros((3, self.model.nv)); jacr = np.zeros_like(jacp)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            error = xyz - self.data.site_xpos[self.site]
            jac = jacp[:, self.dofs]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02**2 * np.eye(3), error)
            update = np.clip(0.7*dq, -0.08, 0.08)
            self.last = np.clip(self.last + update, [-1.7,-1.55,-1.5,-1.45,-2.5], [1.7,1.55,1.5,1.45,2.5])
            self.data.qpos[self.qids] = self.last
        return dict(zip(self.names, self.last.tolist()))

    def _nominal_step(self, obs, t):
        self.last = np.array([obs[f"joint.{n}.pos"] for n in self.names])
        u = min(1.0, max(0.0, t/2.0)); blend = u*u*(3-2*u)
        xyz = np.array([0.391, -0.001 + 0.10*blend, 0.246])
        return self.pose(xyz)

    def step(self, obs, t):
        age = max(0.0, float(t) - float(obs.get("t", t)))
        estimate = dict(obs)
        if age > 0.005:
            for key in tuple(obs):
                if key.startswith("joint.") and key.endswith(".pos"):
                    velocity_key = key[:-4] + ".vel"
                    if velocity_key in obs:
                        # BUG: treats a seconds-based sensor age as milliseconds in a position estimate.
                        estimate[key] = float(obs[key]) - 1000.0 * age * float(obs[velocity_key])
        return self._nominal_step(estimate, t)
