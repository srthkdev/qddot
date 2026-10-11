---
name: qddot
description: Use qddot to run, check, and debug robot controllers, MuJoCo models, simulations, and physics-based control code.
---

# qddot: verify robot code in physics

Use this workflow whenever changing a robot controller, task, model, or physics simulation.

1. **Specify:** inspect the task and model. Write or confirm measurable STL specs. Run `qddot spec TASK` and read each English back-translation.
2. **Run:** run the controller and keep the episode id and verdict.
3. **Diagnose failures:** use `qddot why EPISODE`, then query signals/events and replay near the witness time.
4. **Fix:** change the controller or model based on evidence; rerun and compare.
5. **Fuzz:** run at least 50 perturbations. Require a pass rate of at least 90%.
6. **Report:** include the PASS verdict episode id and fuzz pass rate.

**Never claim done without a PASS verdict id plus a fuzz pass rate of at least 90%.** If fuzzing fails, continue diagnosis and fixes.

## Quick command reference

```sh
qddot check tasks/my_task/model.xml
qddot spec tasks/my_task/task.yaml
qddot run tasks/my_task/task.yaml --controller controller.py --seed 0
qddot why .qddot/runs/EPISODE
qddot query "select t, joint.elbow.pos from signals where episode='EPISODE' order by t"
qddot replay .qddot/runs/EPISODE --at 1.2 --frame
qddot fuzz tasks/my_task/task.yaml --controller controller.py -n 50 --workers 4 --seed 0
qddot view .qddot/runs/EPISODE --open
```

Use `--json` on machine-oriented commands. `qddot query` supports `--limit` (maximum 50 rows through MCP). Install this guide with `qddot skill install --codex`, `--claude`, or `--project`; use `--force` to replace an installed copy.

## Controller API

A controller file defines a class with `step`; `reset` is optional:

```python
class Controller:
    def reset(self, info: dict) -> None:
        # info: actuators, joints, control_dt, target, model_path
        ...

    def step(self, obs: dict[str, float], t: float) -> dict[str, float]:
        # Return actuator-name -> control value. Missing actuators hold.
        ...
```

`obs` contains recorded state and task signals. `ev.*` and `col_*` are excluded. Perturbed runs may apply observation noise and latency. Controls are clipped to actuator `ctrlrange`; clipping emits a saturation event. Exceptions stop the episode and fail the verdict.

## STL cheatsheet

- Comparisons: `x < c`, `x <= c`, `x > c`, `x >= c`, `x == c`; use `abs(x)` for magnitude.
- Arithmetic: signal `a - b`, `a + b`, or constant `k * a`.
- Boolean: `not`, `and`, `or`, `implies`, parentheses.
- Temporal: `always[0,1](...)`, `eventually[0,2](...)`, `a until[0,3] b`. Bounds are seconds; omitted bounds run to trace end.
- Example: `eventually[0,2.5](always[0,1](distance < 0.02))` means the distance stays below 2 cm for a full second, starting within 2.5 seconds.

## Event kinds

`contact_start`, `contact_end`, `slip`, `collision`, `joint_limit`, `saturation`, `deadline_miss`, `nan_state`, and `controller_error`. Events mark state changes; contact end is debounced. A deadline miss delays when the new command takes effect. See [events and signals](reference/events.md).

## Common failures and fixes

| Symptom | Check | Likely fix |
|---|---|---|
| Error grows from the start | Signal sign, coordinate frame, target units | Correct sign/frame/units; verify with `qddot query` |
| Oscillation grows | Position/velocity signals, saturation | Reduce gains, add damping, respect actuator limits |
| Joint limit event | Joint position and declared range | Add limit-aware targets or trajectory constraints |
| Slip or contact end before failure | Contacts, grip force, object motion | Improve grasp geometry/force and avoid abrupt motion |
| Deadline misses | Event times and controller execution | Reduce per-step work or simplify control |
| Only perturbed runs fail | Perturb values and observation delay | Improve robustness to friction, mass, noise, or latency |
| NaN state | Last finite signals and controller outputs | Bound calculations and avoid unstable numerical operations |

For detailed event semantics, signal names, and query recipes, see [events and signals](reference/events.md). For debugging and controller design patterns, see [debugging](reference/debugging.md).
