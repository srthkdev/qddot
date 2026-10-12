"""Damped servo control with a sensor-age predictor."""
import numpy as np


class Controller:
    def reset(self, info: dict)->None:
        self.target=float(info["target"].get("position",0.35)); self.dt=float(info["control_dt"])
        self.kp=22.0; self.kd=8.0; self.integral=0.0

    def step(self, obs: dict[str,float], t: float)->dict[str,float]:
        measured=obs["joint.servo.pos"]; velocity=obs["joint.servo.vel"]
        age=max(0.0,t-float(obs.get("t",t)))
        # BUG: converts the prediction horizon to milliseconds but uses it as seconds.
        predicted=measured+(age+0.5*self.dt)*1000.0*velocity
        error=self.target-predicted
        self.integral=float(np.clip(self.integral+error*self.dt,-0.10,0.10))
        raw=self.kp*error-self.kd*velocity+1.0*self.integral
        return {"servo_motor":raw}
