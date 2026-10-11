"""Cart-pole feedback with a sign error in angular rate."""

import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        # Standard linearized cart-pole LQR gains for the bundled model.
        self.gain = np.array([-2.35, -29.4, -3.7, -5.2])

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        # BUG: angular velocity has the opposite sign from the state used to tune K.
        state = np.array([obs["joint.cart.pos"], obs["joint.pole.pos"], obs["joint.cart.vel"], -obs["joint.pole.vel"]])
        return {"push": float(-self.gain @ state)}
