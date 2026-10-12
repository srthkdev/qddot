"""Signals: joint.shoulder.pos/vel, joint.elbow.pos/vel and both actuator ctrl/force signals. Actuators: shoulder_motor, elbow_motor. info contains actuators, joints, control_dt, target, model_path."""
class Controller:
    def reset(self, info: dict) -> None: pass
    def step(self, obs: dict[str, float], t: float) -> dict[str, float]: return {}
