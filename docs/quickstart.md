# Quickstart

This guide runs the bundled cart-pole controller, inspects its requirements, then diagnoses a deliberately faulty controller.

## Install from source

```sh
uv sync
uv run qddot --help
```

Python 3.12 is required. qddot runs MuJoCo in-process and stores run artifacts under `.qddot/` in the current project directory.

## Inspect the model and requirements

```sh
uv run qddot check tasks/cartpole_balance/model.xml
uv run qddot spec tasks/cartpole_balance/task.yaml
```

To start from another robot model, scaffold a project first:

```sh
uv run qddot init my_arm --robot ur5e
uv run qddot spec my_arm/task.yaml --signals
uv run qddot spec my_arm/task.yaml --check "always(ev.joint_limit < 0.5)"
uv run qddot run my_arm/task.yaml --controller my_arm/controller.py
```

`--robot` accepts the vendored `so101`, `ur5e`, or `aloha` Menagerie names, an MJCF `.xml` path, or a URDF `.urdf` path. Source checkouts include the assets. Pip users should pass a model path or a Menagerie checkout path using `--assets`. URDF conversion creates an MJCF beside the copied URDF, with a floor, light, and default position actuators when needed.

Use `qddot spec TASK --eval "STL" --episode EPISODE` to evaluate a formula against an existing run's recorded signals while iterating on requirements.

Example output from the current bundled task:

```text
upright_and_hold: eventually[0,2](always[0,2](abs(joint.pole.pos) < 0.05))
  within 2 s, |joint.pole.pos| drops below 0.05 and stays there for 2 s
stay_on_rail: always(abs(joint.cart.pos) < 0.8)
  joint.cart.pos stays below 0.8 for the whole run
avoid_saturation: always[0,4](ev.saturation < 0.5)
  no saturation occurs for 4 s
```

## Run and diagnose

```sh
uv run qddot run tasks/cartpole_balance/task.yaml \
  --controller tasks/cartpole_balance/controllers/buggy_velocity_sign.py --seed 0
uv run qddot why .qddot/runs/cartpole_balance-buggy_velocity_sign-s0-94cfad
```

The real CLI reports a failed pole-hold requirement at 2.29 s and points out growing tracking error and a joint limit. The output includes SQL and replay commands for the episode. Run the supplied corrected controller:

```sh
uv run qddot run tasks/cartpole_balance/task.yaml \
  --controller tasks/cartpole_balance/controllers/good.py --seed 0
```

In the captured run, this produced `PASS cartpole_balance` with robustness `0.044`. Episode IDs include task, controller stem, seed, and a perturbation hash; if your task or perturbations differ, use the ID printed by your own run.

## Check robustness

```sh
uv run qddot fuzz tasks/cartpole_balance/task.yaml \
  --controller tasks/cartpole_balance/controllers/good.py -n 50 --workers 1 --seed 0
```

The captured run passed 50/50 samples. Fuzz results depend on the configured perturbation ranges and environment; they are not a guarantee beyond those samples.

Continue with [Tasks and controllers](tasks-and-controllers.md), [Specs](specs.md), and [CLI](cli.md).
