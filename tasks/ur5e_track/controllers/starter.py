class Controller:
    """Observations include joint position and velocity for shoulder_pan_joint, shoulder_lift_joint, elbow_joint; actuator: shoulder_pan, shoulder_lift, elbow."""
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
