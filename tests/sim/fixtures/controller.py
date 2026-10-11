class Controller:
    def reset(self, info):
        self.actuators = info["actuators"]

    def step(self, obs, t):
        return {name: 0.0 for name in self.actuators}
