"""Gravity-compensated PID with realistic integral windup under sustained torque clipping."""
import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = float(info["control_dt"])
        self.target = float(info["target"].get("position", 1.4))
        self.jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
        self.qadr = int(self.model.jnt_qposadr[self.jid])
        self.vadr = int(self.model.jnt_dofadr[self.jid])
        self.integral = 0.0
        self.previous_velocity = 0.0
        self.kp, self.ki, self.kd = 30.0, 6.0, 7.5
        self.torque_limit = 6.0

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        q = obs["joint.shoulder.pos"]
        velocity = obs["joint.shoulder.vel"]
        error = self.target - q
        candidate = float(np.clip(self.integral + error * self.dt, -0.45, 0.45))
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = velocity
        mujoco.mj_forward(self.model, self.data)
        gravity = float(self.data.qfrc_bias[self.vadr])
        raw = gravity + self.kp * error + self.ki * candidate - self.kd * velocity
        # BUG: integrate through saturation in the same direction as the position error.
        self.integral = candidate
        self.previous_velocity = velocity
        return {"shoulder_torque": raw}
