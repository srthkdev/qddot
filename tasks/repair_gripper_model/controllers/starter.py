"""Observations: t; joint.lift_z.pos/vel, joint.left_close.pos/vel, joint.right_close.pos/vel; act.lift.ctrl/force, act.left_close.ctrl/force, act.right_close.ctrl/force; block_z, col_pads_block. Actuators: lift, left_close, right_close. info contains actuators, joints, control_dt, target, and model_path."""


class Controller:
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
