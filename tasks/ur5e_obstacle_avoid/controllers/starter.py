"""Controller stub.

Observations include t and joint.<name>.pos/.vel for six UR5e joints; actuators
are shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, and wrist_3. Info
contains actuators, joints, control_dt, target, and model_path.
"""


class Controller:
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
