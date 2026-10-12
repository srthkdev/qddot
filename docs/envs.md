# qddot tasks as RL environments

Every qddot task can be used to train either a **policy** (actions are actuator commands) or a **coding model** (actions are controller source code). The physics, specs, events and fuzzing are the same ones the CLI uses.

| Export | Action | Observation | Reward | Use it for |
|---|---|---|---|---|
| `qddot.envs.gym.ControlEnv` | actuator vector per control step | task signals | spec robustness at episode end (sparse) | classic RL / imitation policies |
| `qddot.envs.openenv.CodingEnv` | `ControllerAction(code=...)` | verdict + first failing spec + diagnosis | 0–0.25 shaped on failure, 0.5 nominal pass, up to 1.0 at ≥90% fuzz pass | agentic-coding RL via OpenEnv |
| `qddot.envs.verifiers.load_environment` | a ```python block per turn | the same feedback text | `physics_reward` re-grades the final controller | prime-rl and the Environments Hub |
| `qddot.envs.coding.evaluate_source` | — | — | the reward function itself | any trainer |

```bash
pip install "qddot[envs]"   # gymnasium, openenv-core, verifiers
```

## Gymnasium

```python
from qddot.envs.gym import ControlEnv
env = ControlEnv("tasks/cartpole_balance/task.yaml", perturb=True)  # perturb: sample fuzz params each reset
obs, _ = env.reset(seed=0)
obs, reward, done, truncated, info = env.step(env.action_space.sample())
```

`env.obs_keys` and `env.actuators` give the names behind each vector index. At the end, `info["verdict"]` is the full verdict: specs, robustness, witness times and events.

## OpenEnv

```python
from qddot.envs.openenv import CodingEnv, ControllerAction
env = CodingEnv("tasks/so101_reach/task.yaml", max_turns=5)
obs = env.reset()                       # obs.task_prompt: PROMPT.md + task.yaml
obs = env.step(ControllerAction(code=source))
obs.feedback, obs.reward, obs.done      # done when solved (nominal pass + fuzz >= 90%) or out of turns
```

Every evaluation runs in its own temp directory, so the environment supports concurrent sessions. Serve it with OpenEnv's HTTP server like any other environment.

## verifiers / prime-rl

```python
from qddot.envs.verifiers import load_environment
env = load_environment("tasks", max_turns=5)   # one dataset row per task
```

The model replies with a controller in a ```python block, and the environment answers with the physics verdict and diagnosis. A rollout stops early once the controller is solved. The rubric re-grades the final controller from scratch.

## Notes

- **Reward hacking:** the reward comes from the hidden-seed fuzz run, so a controller tuned to the single nominal seed doesn't get full credit.
- **Determinism:** grading is deterministic per seed on a given machine and MuJoCo version.
