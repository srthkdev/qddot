"""qddot tasks as RL environments.

- `coding.evaluate_source`: grade controller *source code* (the agentic-coding reward).
- `gym.ControlEnv`: Gymnasium env; actions are actuator commands each control step.
- `openenv.CodingEnv`: OpenEnv env; actions are controller source, observations are verdicts.
- `verifiers.load_environment`: verifiers multi-turn env for prime-rl and the Environments Hub.
"""
