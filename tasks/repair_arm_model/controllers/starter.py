"""Observations: t; joint.shoulder_pan.pos/vel, joint.elbow_flex.pos/vel; act.shoulder.ctrl/force, act.elbow.ctrl/force; tool_x, tool_y, tool_error. Actuators: shoulder, elbow. info contains actuators, joints, control_dt, target, and model_path."""


class Controller:
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
