"""Gravity-compensated joint trajectory controller for the 2 kg load."""
from pathlib import Path

import mujoco
import numpy as np


class Controller:
    def reset(self, info):
        self.model = mujoco.MjModel.from_xml_path(str(Path(info["model_path"]).with_name("ur5e_no_payload.xml")))
        self.data = mujoco.MjData(self.model)
        self.names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.actuators = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        self.qids = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in self.names])
        self.dofs = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in self.names])
        self.start = np.array([-1.5708,-1.5708,1.5708,-1.5708,-1.5708,0.0])
        target = info.get("target", {})
        self.goal = np.array([target.get(n,v) for n,v in zip(self.names,[-1.25,-1.20,1.35,-1.70,-1.55,0.0])])
        self.kp = np.array([100., 120., 100., 35., 30., 20.])
        self.kd = np.array([24., 28., 20., 7., 6., 4.])

    # BUG: compensation uses the nominal arm model and omits the attached payload mass.
    def step(self, obs, t):
        q = np.array([obs[f"joint.{n}.pos"] for n in self.names])
        dq = np.array([obs[f"joint.{n}.vel"] for n in self.names])
        self.data.qpos[self.qids] = q
        self.data.qvel[self.dofs] = dq
        mujoco.mj_forward(self.model, self.data)
        phase = np.clip(t/1.3,0.0,1.0)
        blend = 10*phase**3-15*phase**4+6*phase**5
        desired = self.start + blend*(self.goal-self.start)
        # The model bias includes the payload mass and counteracts gravity at every sample.
        torque = self.data.qfrc_bias[self.dofs] + self.kp*(desired-q) - self.kd*dq
        return dict(zip(self.actuators,torque.tolist()))
