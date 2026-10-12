"""Signals: joint.shoulder.pos, joint.shoulder.vel, act.shoulder_torque.ctrl, act.shoulder_torque.force. Actuator: shoulder_torque. info contains actuators, joints, control_dt, target, model_path."""
class Controller:
    def reset(self, info: dict) -> None:
        pass
    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
