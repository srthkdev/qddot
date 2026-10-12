class Controller:
    """Observations include joint.cart.pos, joint.pole.pos, joint.cart.vel, joint.pole.vel; actuator: push."""
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        return {}
