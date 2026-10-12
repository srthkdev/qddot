class Controller:
    """Observations include joint position and velocity for shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll; dist_ee_target; actuator: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll."""
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
