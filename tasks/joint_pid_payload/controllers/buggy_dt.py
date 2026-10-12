"""Gravity-compensated PID with a finite-difference velocity estimate."""
import mujoco
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.model = mujoco.MjModel.from_xml_path(info["model_path"])
        self.data = mujoco.MjData(self.model)
        self.dt = 0.05  # BUG: assumes a 20 Hz control loop instead of the task 100 Hz loop.
        self.target = float(info["target"].get("position", 1.4))
        self.jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
        self.qadr = int(self.model.jnt_qposadr[self.jid])
        self.vadr = int(self.model.jnt_dofadr[self.jid])
        self.integral = 0.0
        self.previous_position = 0.0
        self.kp, self.ki, self.kd = 30.0, 2.0, 7.5
        self.torque_limit = 6.0

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        q = obs["joint.shoulder.pos"]
        velocity = (q - self.previous_position) / self.dt
        error = self.target - q
        candidate = float(np.clip(self.integral + error * self.dt, -0.45, 0.45))
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.vadr] = velocity
        mujoco.mj_forward(self.model, self.data)
        gravity = float(self.data.qfrc_bias[self.vadr])
        raw = gravity + self.kp * error + self.ki * candidate - self.kd * velocity
        command = float(np.clip(raw, -self.torque_limit, self.torque_limit))
        if abs(raw - command) < 1e-9 or np.sign(error) != np.sign(raw - command):
            self.integral = candidate
        self.previous_position = q
        return {"shoulder_torque": raw}
