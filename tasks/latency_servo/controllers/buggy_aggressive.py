"""PD step tracking controller for the fast servo."""
import numpy as np


class Controller:
    def reset(self, info: dict)->None:
        self.target=float(info["target"].get("position",0.35)); self.dt=float(info["control_dt"])
        self.kp=58.0; self.kd=1.8; self.integral=0.0

    def step(self, obs: dict[str,float], t: float)->dict[str,float]:
        measured=obs["joint.servo.pos"]; velocity=obs["joint.servo.vel"]
        # BUG: gains assume fresh feedback; no prediction compensates the configured sensor delay.
        error=self.target-measured
        self.integral=float(np.clip(self.integral+error*self.dt,-0.12,0.12))
        return {"servo_motor":float(np.clip(self.kp*error-self.kd*velocity+1.2*self.integral,-8.0,8.0))}
