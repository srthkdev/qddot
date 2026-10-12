# Tasks and controllers

## Task YAML

Create `task.yaml` next to your model. Model paths are resolved relative to the YAML file.

```yaml
name: arm_reach
description: Move the end effector to the target and hold.
model: model.xml
duration: 4.0
control_dt: 0.02
init:
  keyframe: home
  qpos: {shoulder: -0.3}
target: {x: 0.2, y: 0.0, z: 0.15}
signals:
  - {name: ee_x, kind: site_pos, ref: gripperframe, axis: x}
  - {name: distance, kind: distance, ref: gripperframe, to: target}
contact_pairs:
  - {name: arm_table, a: [upper_arm, wrist], b: [table]}
perturb:
  friction_scale: [0.7, 1.3]
  mass_scale: [0.9, 1.1]
  obs_noise_std: 0.002
  latency_ms: [0, 40]
  jitter_ms: 5
  controller_budget_ms: 15
specs:
  - name: reach_and_hold
    stl: "eventually[0,2.5](always[0,1](distance < 0.02))"
```

Required keys are `name`, `model`, `duration`, and `control_dt`. `description`, `init`, `target`, `signals`, `contact_pairs`, `perturb`, and `specs` are optional. `control_dt` must be a multiple of the model's physics timestep. Signals support `site_pos`, `body_pos`, and `distance`; position signals require an `axis` of `x`, `y`, or `z`. For distance, `to` may be a named site/body or `target`.

Perturbation ranges use `[low, high]`; scalar values such as `obs_noise_std` and `jitter_ms` configure that parameter directly for fuzz sampling. The scalar `controller_budget_ms` is also used on nominal runs. Pass `--perturb controller_budget_ms=null` to disable it for a run.

## Controller API

A controller file defines `Controller`. `reset` is optional and receives model/task metadata. `step` runs at every control interval:

```python
class Controller:
    def reset(self, info: dict) -> None:
        self.actuator = info["actuators"][0]

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        error = 0.5 - obs["joint.elbow.pos"]
        return {self.actuator: 4.0 * error - 0.3 * obs["joint.elbow.vel"]}
```

`info` contains actuator and joint names, `control_dt`, the free-form task `target`, and `model_path`. `obs` contains recorded state and task-derived signals; event and contact-pair indicators are excluded. Perturbed runs can add observation noise and latency. Return actuator-name-to-control-value mappings. Omitted actuators retain their previous command; unknown actuator names are ignored; limited controls are clipped and reported as saturation events. Controller exceptions stop the episode and fail its verdict.

See [`examples/pendulum/`](../examples/pendulum/) for a complete task from scratch.
