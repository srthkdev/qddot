"""Cascaded position/pitch feedback with a smooth reference and common wheel torque."""
import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        self.target=float(info["target"].get("position",0.5)); self.dt=float(info["control_dt"])
        self.travel_kp=1.8; self.travel_kd=2.2; self.pitch_kp=150.0; self.pitch_kd=12.0
        self.max_force=12.5

    def step(self, obs: dict[str,float], t: float)->dict[str,float]:
        x=obs["joint.travel.pos"]; xd=obs["joint.travel.vel"]
        pitch=obs["joint.pitch.pos"]; pitch_rate=obs["joint.pitch.vel"]
        torque=4.0*(x-self.target)+self.pitch_kp*pitch+8.0*xd+self.pitch_kd*pitch_rate
        torque=float(np.clip(torque,-self.max_force,self.max_force))
        return {"left_drive":torque,"right_drive":torque}
