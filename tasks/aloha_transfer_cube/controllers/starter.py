"""Starter controller for the ALOHA transfer task.

Observations include t; joint.<name>.pos and joint.<name>.vel for all 16
joints; act.<name>.ctrl and act.<name>.force for 14 actuators; cube_x,
cube_y, cube_z, cube_right_pinch and cube_left_pinch. Actuator names are
left/{waist,shoulder,elbow,forearm_roll,wrist_angle,wrist_rotate,gripper}
and the same seven right/ names. reset(info) receives actuators, joints,
control_dt, target, and model_path.
"""


class Controller:
    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
