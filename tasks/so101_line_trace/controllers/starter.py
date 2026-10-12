"""Controller interface.

Observations: joint.<name>.pos/vel, act.<name>.ctrl/force, and task signals.
Actuators: read info["actuators"]. info also contains joints, control_dt, target,
and model_path.
"""
class Controller:
    def reset(self, info: dict) -> None:
        self.actuators = info["actuators"]

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
