"""Resolved-rate Cartesian servo for the UR5e tool flange."""
import mujoco
import numpy as np


class Controller:
    def reset(self, info):
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.qids = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.names]
        self.dofs = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.names]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool_flange")
        self.target = np.array([info.get("target", {}).get(k, d) for k,d in zip("xyz", (-0.05,0.40,0.48))])

    def step(self, obs, t):
        q = np.array([obs[f"joint.{n}.pos"] for n in self.names])
        self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.model, self.data)
        jp = np.zeros((3,self.model.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
        jac = jp[:,self.dofs]
        error = self.target-self.data.site_xpos[self.site]
        dq = jac.T @ np.linalg.solve(jac@jac.T + 0.04**2*np.eye(3), error)
        # Bound each resolved-rate update and retain clearance from the position limits.
        limits = self.model.jnt_range[[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in self.names]]
        command = np.clip(q + np.clip(0.9*dq,-0.035,0.035),limits[:,0]+0.15,limits[:,1]-0.15)
        return dict(zip(["shoulder_pan","shoulder_lift","elbow","wrist_1","wrist_2","wrist_3"],command.tolist()))
