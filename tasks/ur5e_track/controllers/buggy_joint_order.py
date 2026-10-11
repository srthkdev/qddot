"""Trajectory controller with a subtle actuator mapping swap."""


class Controller:
    def reset(self, info: dict) -> None:
        self.goals = [-1.20, -1.35, 1.25]
        self.starts = [-1.5708, -1.5708, 1.5708]

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        phase = min(max(t / 2.0, 0.0), 1.0)
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        positions = [a + blend * (b - a) for a, b in zip(self.starts, self.goals)]
        # BUG: shoulder-lift and elbow commands are assigned in the wrong order.
        return {"shoulder_pan": positions[0], "shoulder_lift": positions[2], "elbow": positions[1]}
