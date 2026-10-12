# Pendulum swing-to-target

This minimal example contains a MuJoCo model, qddot task, and controller. The position actuator moves the pendulum from −0.7 rad to a 0.55 rad target and holds it there.

From the repository root:

```sh
uv run qddot check examples/pendulum/model.xml
uv run qddot spec examples/pendulum/task.yaml
uv run qddot run examples/pendulum/task.yaml \
  --controller examples/pendulum/controller.py --seed 0
```

The task requires entering an 0.08 rad target band and holding it for 0.8 seconds within the first 2 seconds, while staying inside a broader angle limit. After running, use the printed episode ID with `qddot why`, `qddot replay`, or `qddot view`. The controller reads its target from `info` rather than duplicating task configuration.
