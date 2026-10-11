# qddot — engineering contract (read fully before coding)

qddot lets a coding agent **run, check and debug robot code in physics** before claiming it is done.

- **Delivery:** one Python package that is a library and a CLI (`qddot ...`).
- **No servers:**
  - MuJoCo runs in-process.
  - DuckDB is embedded.
  - All data lives in `.qddot/` inside the user's project.

## Rules for every contributor

- **Python environment:**
  - Python 3.12, managed by `uv`.
  - Run things with `uv run ...`.
  - Run tests with `uv run pytest -q`.
- **Dependencies:**
  - Already installed: mujoco 3.15, mcap, pyarrow, duckdb, numpy, typer, pyyaml, rtamt, rich, pillow, pytest, ruff.
  - **Do not add or upgrade dependencies, and do not edit `pyproject.toml`.** The network is unavailable.
  - If you truly need a new dependency, say so in your final message.
- **Ownership:** only create or edit the files your task assigns to you. Other agents are working in parallel on other modules.
  - If you need something from another module that doesn't exist yet, code against the contract below.
  - In tests, use a small local fake.
- **Code quality:**
  - Write production-quality code: type hints, small functions, clear names.
  - Docstrings only where they help.
  - No dead code.
  - `uv run ruff check` must be clean for your files.
- **Testing:**
  - Every module ships with pytest tests under `tests/` (file names given per task).
  - Tests must be fast (< 20 s total per module) and deterministic.
- **Output:** be concise and agent-friendly.
  - Short text by default.
  - `--json` for machine output.
  - Never dump raw arrays.
- **Determinism:** same task + controller + seed + perturb ⇒ bitwise-identical signals on the same machine and MuJoCo version.
  - Seed every RNG explicitly; use `numpy.random.default_rng(seed)`.
- **Repository:** do not commit. The orchestrator commits.

## Layout

```
src/qddot/
  __init__.py          (orchestrator)
  task.py              Task model + YAML loader                [runner agent]
  sim/runner.py        run_episode()                           [runner agent]
  sim/events.py        per-step event detection                [runner agent]
  sim/perturb.py       physics + timing perturbations          [runner agent]
  sim/fuzz.py          fuzz()                                  [runner agent]
  record/writer.py     Parquet + MCAP + verdict writers         [runner agent]
  spec/                STL parser, evaluator, robustness, english  [spec agent]
  check/static.py      check_model()                           [cli agent]
  store.py             DuckDB views over .qddot/runs         [cli agent]
  cli.py               Typer app, entry point `qddot`        [cli agent]
tasks/<task_name>/     built-in benchmark tasks                 [tasks agent]
assets/                vendored MuJoCo Menagerie models (SO-101, UR5e). Read-only.
tests/                 tests/<module>/test_*.py
```

## Contracts

### 1. Task file (`tasks/<name>/task.yaml`)

Paths are relative to the task file.

```yaml
name: so101_reach
description: Move the gripper to the target and hold.
model: ../../assets/robotstudio_so101/scene.xml   # or a model.xml next to the task
duration: 4.0            # simulated seconds
control_dt: 0.02         # controller period in seconds (must be a multiple of model timestep)
init:                    # optional
  keyframe: home         # optional keyframe name
  qpos: {shoulder_lift: -0.5}   # optional per-joint overrides (joint name -> value)
target: {x: 0.2, y: 0.0, z: 0.15}   # optional free-form dict passed to the controller as info["target"]
signals:                 # extra derived signals (built-in signals are always recorded)
  - {name: ee_x, kind: site_pos, ref: gripperframe, axis: x}
  - {name: dist_ee_target, kind: distance, ref: gripperframe, to: target}   # `to` = a site/body name or "target"
  - {name: cube_z, kind: body_pos, ref: box, axis: z}
contact_pairs:           # optional named pairs that produce collision indicator signals
  - {name: arm_table, a: [upper_arm, lower_arm, wrist, gripper], b: [floor]}   # body names
perturb:                 # default ranges used by `fuzz` (all optional)
  friction_scale: [0.7, 1.3]
  mass_scale: [0.9, 1.1]
  obs_noise_std: 0.002
  latency_ms: [0, 40]          # sensor-to-controller delay, rounded to control steps
  jitter_ms: 5                 # std of extra actuation delay, rounded to physics steps
  controller_budget_ms: 15     # wall-clock compute budget per control step (deadline)
specs:
  - name: reach
    stl: "eventually[0,2.5](always[0,1](dist_ee_target < 0.02))"
  - name: no_table_collision
    stl: "always(col_arm_table < 0.5)"
```

`task.py` exposes:

- `@dataclass Task` holding those fields, plus `path: Path` and `model_path: Path`.
- `load_task(path: str | Path) -> Task`.
  - Validates the fields.
  - Raises `TaskError` with a clear message on a bad file.

### 2. Signal naming

Signals are wide-format columns. They are all float64 and sampled once per **control step**, and include t=0.

| Signal | Name |
|---|---|
| Time | `t` |
| Joints (each named hinge/slide joint) | `joint.<name>.pos`, `joint.<name>.vel` |
| Actuators | `act.<name>.ctrl` (applied), `act.<name>.force` |
| Task-declared signals | their `name` |
| Contact pairs | `col_<pair_name>`: 1.0 if any contact between the groups at that step, else 0.0 |
| Event indicators | `ev.collision`, `ev.joint_limit`, `ev.saturation`, `ev.deadline_miss`: 1.0 if that event kind occurred during that control step |

### 3. Controller API (user code)

A Python file that defines `class Controller`:

```python
class Controller:
    def reset(self, info: dict) -> None: ...      # optional
    def step(self, obs: dict[str, float], t: float) -> dict[str, float]: ...
```

`info` contains:

- `actuators`: list of names
- `joints`: list of names
- `control_dt`
- `target`: the task's target dict, or `{}`
- `model_path`: str

`obs` maps every signal name (except `ev.*` and `col_*`) to its current value. Latency and noise apply when perturbed.

`step` returns `{actuator_name: ctrl}`. Missing actuators hold their previous ctrl. Values are clipped to `ctrlrange` (emitting a `saturation` event when clipped).

Controller exceptions are caught: the episode stops, `verdict.errors` records the traceback (last 20 lines), and `passed=False`.

### 4. Episode outputs: `.qddot/runs/<episode_id>/`

`episode_id` = `<task>-<controller_stem>-s<seed>-<6 hex of params hash>`.

| File | Contents |
|---|---|
| `signals.parquet` | wide table (section 2) |
| `events.parquet` | see the schema below |
| `verdict.json` | see the schema below |
| `log.mcap` | JSON-encoded channels `/signals` (one message per control step) and `/events` |
| `meta.json` | task path, controller path + sha256, seed, perturb params actually sampled, env info |

`events.parquet` columns:

| Column | Type | Meaning |
|---|---|---|
| `t` | float64 | time of the event |
| `kind` | str | one of the kinds below |
| `a` | str | primary body/geom/joint/actuator |
| `b` | str or null | second party, if any |
| `value` | float64 | normal force N / slip speed m/s / penetration m / overshoot rad / ms |
| `detail` | str | small JSON |

Event kinds:

- **`contact_start` / `contact_end`:** body pair, deduplicated per body pair; `value` = normal force at start.
- **`slip`:** a persisting contact whose tangential relative speed > 0.01 m/s for at least 2 consecutive control steps.
- **`collision`:** contact_start between bodies in a declared `contact_pairs` group, or `penetration` > 5 mm.
- **`joint_limit`:** a limited joint within 1% of its range of a limit, or beyond it; emitted on entry.
- **`saturation`:** controller output clipped to ctrlrange; emitted on entry per actuator.
- **`deadline_miss`:** controller wall time > `controller_budget_ms`; `value` = ms.
- **`nan_state`:** non-finite qpos/qvel. The episode stops.
- **`controller_error`:** the controller raised.

Events are emitted on **state change**, not every step, to keep tables small.

`verdict.json`:

```json
{
  "episode": "...", "task": "...", "controller": "path", "seed": 0,
  "passed": false,
  "robustness": -0.013,
  "specs": [{"name": "reach", "stl": "...", "english": "...", "passed": false,
             "robustness": -0.013, "witness_t": 1.84, "window": [1.5, 2.5]}],
  "first_violation": {"spec": "reach", "t": 1.84},
  "events_summary": {"contact_start": 3, "slip": 1},
  "errors": [],
  "perturb": {...},
  "env": {"mujoco": "3.15.0", "python": "3.12.x", "platform": "macOS-arm64", "qddot": "0.1.0"},
  "stats": {"control_steps": 200, "physics_steps": 1000, "wall_s": 0.21}
}
```

`passed` = all specs passed and no errors. `robustness` = min over specs.

### 5. Spec package API (`qddot.spec`)

- `parse(stl: str) -> Node`: raises `SpecError` with position info.
- `to_english(stl: str) -> str`: deterministic, template-based back-translation.
- `evaluate(stl: str, t: np.ndarray, signals: Mapping[str, np.ndarray]) -> SpecResult`. The dataclass has:
  - `passed`
  - `robustness: float` (the robustness at time 0)
  - `witness_t: float | None`: the time that determined the robustness (the critical sample)
  - `window: tuple[float, float] | None`: the time interval the critical operator ranged over
- `evaluate_specs(specs: list[dict], t, signals) -> list[dict]`: dicts shaped like `verdict.specs` entries.

Grammar (discrete time, samples at control steps, bounds in seconds):

- Atoms: `sig < c`, `sig <= c`, `sig > c`, `sig >= c`, `sig == c` (with an `abs(...)` helper), and simple arithmetic `a - b`, `a + b`, `k * a` over signals and constants.
- Boolean: `not`, `and`, `or`, `implies`, and parentheses.
- Temporal: `always`, `eventually`, `until`, each with an optional `[a,b]` bound. With no bound they cover the rest of the trace.
- Unknown signal names raise `SpecError`, listing the available names closest to the unknown one.

Quantitative semantics are the standard STL robustness (Donzé–Maler); bounded windows are truncated at the trace end.

### 6. Other APIs

- `qddot.sim.runner.run_episode(task: Task, controller_path: Path, seed: int = 0, perturb: dict | None = None, root: Path = Path(".qddot"), record_mcap: bool = True) -> EpisodeResult`, where `EpisodeResult` has `episode_dir: Path` and `verdict: dict`.
  - `perturb=None` means nominal (no perturbation).
  - A dict holds explicit sampled values, e.g. `{"friction_scale": 0.8, "mass_scale": 1.0, "obs_noise_std": 0.0, "latency_ms": 20, "jitter_ms": 0, "controller_budget_ms": null}`.
- `qddot.sim.perturb.sample(task: Task, rng) -> dict`: samples concrete values from `task.perturb` ranges.
- `qddot.sim.fuzz.fuzz(task, controller_path, n: int, workers: int = os.cpu_count(), seed: int = 0, root=...) -> dict`.
  - Uses a process pool; episode i uses seed `seed + i` and `sample(task, default_rng(seed + i))`.
  - Returns `{"n", "passed", "pass_rate", "worst": [top 5 by robustness with episode ids + perturb], "by_param": {param: {"pass_corr": float}}}`.
- `qddot.check.static.check_model(path) -> list[dict]`. Each dict is `{severity: error|warning|info, code, message, element}`. Checks:
  - load errors
  - bodies with mass ≤ 1e-6 or invalid inertia (triangle inequality)
  - hinge/slide joints that are actuated but unlimited
  - actuator ctrlrange outside the joint range for position actuators
  - initial penetration > 1 mm at the default/first keyframe
  - timestep > 0.01
  - geoms with zero size
  - free bodies starting in the air (info)
- `qddot.store.connect(root=Path(".qddot")) -> duckdb.DuckDBPyConnection` with views:
  - `signals`: all episodes, plus an `episode` column
  - `events`: all episodes, plus an `episode` column
  - `verdicts`: one row per episode, with flattened top-level fields and `specs` as JSON

## CLI (Typer, `qddot`)

| Command | Does |
|---|---|
| `run TASK --controller FILE [--seed N] [--perturb k=v ...] [--json]` | Prints a ≤ 12-line verdict: PASS/FAIL, robustness, per-spec line with witness time, top events, episode id |
| `fuzz TASK --controller FILE [-n 50] [--workers W] [--seed 0] [--json]` | Prints pass rate, worst 5, and the param most correlated with failure |
| `check MODEL [--json]` | Static model checks |
| `spec TASK` | Lists specs with their English back-translations |
| `query "SQL" [--json] [--limit 50]` | DuckDB over the views |
| `runs [--last 10]` | Lists recent episodes |

Phase 2 commands (`why`, `diff`, `replay`, `view`, `similar`) are not in scope yet. Leave clean extension points.
