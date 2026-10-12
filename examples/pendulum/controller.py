class Controller:
    def reset(self, info: dict) -> None:
        self.target = float(info["target"]["angle_rad"])

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        # The position actuator closes its own loop; send the desired angle.
        return {"hinge_target": self.target}
