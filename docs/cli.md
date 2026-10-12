# CLI reference

Run `qddot --help` for the command list and `<command> --help` for option details. The CLI uses exit status 0 for successful commands, 1 for a failed run/check, and 2 for command errors. `--json` is available on run, fuzz, check, why, diff, replay, query, remember, and similar.

## Commands and flags

| Command | Flags and behavior |
|---|---|
| `run TASK --controller FILE` | `--seed N` (default 0), repeatable `--perturb key=value`, `--json`; runs once and records an episode. |
| `fuzz TASK --controller FILE` | `-n N` (default 50), `--workers W`, `--seed N`, `--json`; reports pass rate, worst five, and parameter correlation. |
| `check MODEL` | `--json`; static physical/model setup checks. |
| `init [DIR] --robot NAME_OR_PATH` | Creates a runnable project with task, starter controller, sample specs, and `.qddot/` ignored. Supports Menagerie `so101`, `ur5e`, `aloha`, MJCF `.xml`, or URDF `.urdf`; `--task NAME` sets the task name and `--assets PATH` locates vendored Menagerie files. |
| `spec TASK` | Prints each STL spec and English read-back. `--signals` lists available signals by group; `--check "STL"` validates syntax and names; `--eval "STL" --episode ID` evaluates recorded signals. |
| `why EPISODE` | `--frames`, `--vlm`, `--json`; failure diagnosis, optional re-simulated frames and visual diff. |
| `diff A B` | `--tolerance FLOAT` (default `1e-6`), `--json`; compares episodes. |
| `replay EPISODE --at T` | `--frame`, `--json`; saves integration state at T and prints joint state/contact summary. |
| `view EPISODE` | `--open`; writes `replay.gif` and `replay.html`. |
| `query SQL` | `--root DIR` (default `.qddot`), `--limit N` (default 50), `--json`; queries `signals`, `events`, and `verdicts` views. |
| `runs` | `--last N` (default 10), `--root DIR`; lists newest verdict files. |
| `remember [EPISODE]` | `--all`, `--global`, `--root DIR`, `--json`; indexes failure cards. Use either an episode path or `--all`. |
| `similar EPISODE` | `-k N` (default 3), `--global`, `--root DIR`, `--json`; retrieves related failure cards. |
| `skill install` | `--claude`, `--codex` (default), or `--project`; `--force` replaces an existing different installation. |
| `mcp` | Launches the qddot MCP server over stdio. |

### Initialize a project and work with specs

```sh
qddot init my_robot --robot so101
qddot spec my_robot/task.yaml --signals
qddot spec my_robot/task.yaml --check "always(ev.joint_limit < 0.5)"
qddot run my_robot/task.yaml --controller my_robot/controller.py
qddot spec my_robot/task.yaml --eval "always(ev.joint_limit < 0.5)" --episode EPISODE
```

`init` copies the Menagerie model folder to the project so local mesh and include references remain valid. Menagerie assets are present in a source checkout; pip users should pass an MJCF/URDF model path or use `--assets` with a Menagerie checkout. A URDF is copied into the project and converted to MJCF next to that copy; the converted model receives a floor, light, and position actuators for its joints when it has no actuators.

`spec --check` loads the task model to discover built-in joint and actuator signals, then adds task signals, contact indicators, and event indicators. It does not run a simulation. `spec --eval` reads `signals.parquet` from the existing episode and does not re-run the controller.

## Run and diagnose

Captured CLI output for the bundled cart-pole task:

```text
$ qddot run tasks/cartpole_balance/task.yaml --controller tasks/cartpole_balance/controllers/buggy_velocity_sign.py --seed 0
FAIL cartpole_balance  robustness -1.748  episode cartpole_balance-buggy_velocity_sign-s0-94cfad
  upright_and_hold     FAIL  rob -1.748  worst at t=2.29s (window 0.00-2.00s)
  stay_on_rail         FAIL  rob -0.237  worst at t=1.24s (window 0.00-4.00s)
  avoid_saturation     FAIL  rob -0.500  worst at t=0.15s (window 0.00-4.00s)
  events: contact_start x6 (first floor @0.00s), contact_end x4 (first floor @0.52s), slip x4 (first floor @0.02s)
  why: tracking error grows from the start; check sign or coordinate frame [med]
  files: .qddot/runs/cartpole_balance-buggy_velocity_sign-s0-94cfad/  (signals.parquet events.parquet verdict.json log.mcap)
  next: qddot query "select ..." | qddot fuzz ...

$ qddot why .qddot/runs/cartpole_balance-buggy_velocity_sign-s0-94cfad
FAIL episode cartpole_balance-buggy_velocity_sign-s0-94cfad
first failing spec: upright_and_hold at 2.29s window 0.00-2.00s
  within 2 s, |joint.pole.pos| drops below 0.05 and stays there for 2 s
joint.pole.pos: -1.798 (window -1.819..0.1), falling
first events: slip floor @0.022s; saturation push @0.140s; contact_end floor @0.524s; joint_limit cart @0.526s
likely causes: tracking error grows from the start; check sign or coordinate frame [med]; joint range was reached [high]
  evidence: diverging signal(s): joint.pole.pos; saturation began at 0.140s
  evidence: joint_limit on cart at 0.526s
next: qddot query "select t, \"joint.pole.pos\" from signals where episode = 'cartpole_balance-buggy_velocity_sign-s0-94cfad' order by t limit 20"
next: qddot replay .qddot/runs/cartpole_balance-buggy_velocity_sign-s0-94cfad --at 2.290
```

## Run and fuzz

```text
$ qddot run tasks/cartpole_balance/task.yaml --controller tasks/cartpole_balance/controllers/good.py --seed 0
PASS cartpole_balance  robustness 0.044  episode cartpole_balance-good-s0-94cfad
  upright_and_hold     PASS  rob 0.044  worst at t=2.76s (window 0.00-2.00s)
  stay_on_rail         PASS  rob 0.594  worst at t=1.04s (window 0.00-4.00s)
  avoid_saturation     PASS  rob 0.500  worst at t=0.00s (window 0.00-4.00s)
  events: slip x2 (first floor @0.02s), contact_start x1 (first floor @0.00s)
  files: .qddot/runs/cartpole_balance-good-s0-94cfad/  (signals.parquet events.parquet verdict.json log.mcap)
  next: qddot query "select ..." | qddot fuzz ...

$ qddot fuzz tasks/cartpole_balance/task.yaml --controller tasks/cartpole_balance/controllers/good.py -n 50 --workers 1 --seed 0
fuzz: 50/50 passed  pass rate 100.0%
  cartpole_balance-good-s17-4c132b  robustness 0.044
  cartpole_balance-good-s46-31c3da  robustness 0.044
  cartpole_balance-good-s31-eb2c42  robustness 0.044
  cartpole_balance-good-s42-366e1e  robustness 0.044
  cartpole_balance-good-s10-3dfd4e  robustness 0.044
  most correlated: controller_budget_ms (pass_corr n/a)
```

The run and fuzz outputs are verbatim observations for this checkout and machine; episode IDs are derived from run inputs and must be taken from your own output.

## Inspect data and compare episodes

The query below was run against the captured demo project. DuckDB's text table formatting can wrap columns based on terminal width.

```text
$ qddot query "select episode, task, passed, robustness from verdicts order by episode" --root .qddot --limit 3
 episode                          task              passed  robustness
 cartpole_balance-buggy_velocity  cartpole_balance  False   -1.7477498282945596
 _sign-s0-94cfad
 cartpole_balance-buggy_velocity  cartpole_balance  False   -1.7477499716025047
 _sign-s0-f24b51
 cartpole_balance-buggy_velocity  cartpole_balance  False   -1.7477498406655876
 _sign-s1-2bdf0d
showing first 3 rows; results truncated (use --limit to change)
```

Compare two recorded runs with:

```text
$ qddot diff EPISODE_A EPISODE_B
episodes: cartpole_balance-buggy_velocity_sign-s0-94cfad vs cartpole_balance-good-s0-94cfad
signal divergences: 9/10
  act.push.ctrl first differs at 0.010s (max 10.54)
  act.push.force first differs at 0.010s (max 10.54)
  ev.collision first differs at 0.510s (max 1)
  ev.joint_limit first differs at 0.530s (max 1)
  ev.saturation first differs at 0.150s (max 1)
  joint.cart.pos first differs at 0.010s (max 1.237)
  joint.cart.vel first differs at 0.010s (max 5.534)
  joint.pole.pos first differs at 0.010s (max 1.804)
  joint.pole.vel first differs at 0.010s (max 8.335)
  events only A/B: 16/0
  ...
```

`diff` prints at most 12 signal divergences in text mode; the excerpt above omits the remaining real output lines. `qddot replay EPISODE --at 0.5` reports joint positions/velocities, active contacts, and the saved `state_500.npy` path. `qddot view EPISODE` writes a GIF and HTML report; it requires MuJoCo rendering support on the current machine.

For scripting, pass `--json` and use `--root` on data-inspection commands to point them at a particular project store.
