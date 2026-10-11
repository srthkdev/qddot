"""A balancing controller that misses a tight real-time deadline."""

import time


class Controller:
    def reset(self, info: dict) -> None:
        pass

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        # BUG: this delay exceeds the task's 10 ms controller budget every step.
        time.sleep(0.012)
        return {"push": -30.0 * obs["joint.pole.pos"] - 5.0 * obs["joint.pole.vel"]}
