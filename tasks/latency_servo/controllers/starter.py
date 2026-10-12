"""Signals: joint.servo.pos, joint.servo.vel, act.servo_motor.ctrl/force. Actuator: servo_motor. info contains actuators, joints, control_dt, target, model_path."""
class Controller:
    def reset(self, info: dict) -> None: pass
    def step(self, obs: dict[str, float], t: float) -> dict[str, float]: return {}
