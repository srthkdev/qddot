from typing import ClassVar

"""Clearance-checked joint-space waypoint trajectory for the UR5e."""

import mujoco
import numpy as np


class Controller:
    joints: ClassVar[list[str]] = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    actuators: ClassVar[list[str]] = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    via = np.array([-2.36356, -0.59985, 1.95469, -1.59736, -0.01485, 0.0])

    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = float(info["control_dt"])
        self.goal = np.array([float(info["target"][name]) for name in self.joints])
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joints]
        self.qids = np.array([self.model.jnt_qposadr[jid] for jid in joint_ids])
        self.obstacle_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_box")
        if not self._clear_segment(self.home, self.via) or not self._clear_segment(self.via, self.goal):
            raise ValueError("configured waypoint route intersects the obstacle")

    def _clear_segment(self, start: np.ndarray, end: np.ndarray) -> bool:
        for phase in np.linspace(0.0, 1.0, 101):
            self.data.qpos[self.qids] = start + phase * (end - start)
            mujoco.mj_forward(self.model, self.data)
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                if self.obstacle_geom in (contact.geom1, contact.geom2):
                    return False
        return True

    @staticmethod
    def _blend(phase: float) -> float:
        u = float(np.clip(phase, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        first_duration = 2.4
        second_duration = 2.4
        if t <= first_duration:
            phase = t / first_duration
            command = self.home + self._blend(phase) * (self.via - self.home)
        else:
            phase = (t - first_duration) / second_duration
            command = self.via + self._blend(phase) * (self.goal - self.via)
        return dict(zip(self.actuators, command.tolist()))
