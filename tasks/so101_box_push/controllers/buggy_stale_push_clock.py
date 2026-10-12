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
        age = max(0.0, float(t) - float(obs.get("t", t)))
        profile_t = max(0.0, t - 200.0 * age)
        approach = min(1.0, profile_t / 0.8)
        travel = min(1.0, max(0.0, (profile_t - 0.8) / 2.6))
        # BUG: run the contact plan on a timestamp retarded by the stale sensor age.
        x = 0.32 + 0.15 * travel
        z = 0.105 - 0.05 * approach
        output = self.pose(np.array([x, 0.0, z]))
        output["gripper"] = 0.65
        return output

    def step(self, obs, t):
        return self._nominal_step(obs, t)
