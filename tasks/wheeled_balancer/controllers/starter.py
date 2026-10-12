"""Signals: joint.travel.pos/vel, joint.pitch.pos/vel, wheel spin states and actuator ctrl/force. Actuators: left_drive, right_drive. info contains actuators, joints, control_dt, target, model_path."""
class Controller:
    def reset(self, info: dict) -> None: pass
    def step(self, obs: dict[str, float], t: float) -> dict[str, float]: return {}
