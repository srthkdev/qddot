"""Signals: robot_x, robot_y, joint.yaw.pos/vel, joint.left_wheel_spin.pos/vel, joint.right_wheel_spin.pos/vel and actuator ctrl/force. Actuators: left_drive, right_drive, heading_trim. info contains actuators, joints, control_dt, target, model_path."""
class Controller:
    def reset(self, info: dict) -> None: pass
    def step(self, obs: dict[str, float], t: float) -> dict[str, float]: return {}
