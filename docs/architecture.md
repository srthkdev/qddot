# Architecture

```text
 task.yaml ────────┐
 MJCF model ───────┼──> qddot run ──> in-process MuJoCo + Controller.step()
 controller.py ────┘                          │
                                              ├── signals + event detection
                                              ├── STL evaluation / verdict
                                              └── episode files in .qddot/
                                                       │
                             ┌─────────────────────────┼──────────────────────┐
                             v                         v                      v
                         qddot why                 qddot query          replay / diff / view
                     diagnosis evidence         embedded DuckDB       recorded-control replay
```

MuJoCo is loaded and stepped inside the Python process. DuckDB reads local Parquet/JSON artifacts and exposes project views; there is no qddot server or external database service.

## Project data

```text
.qddot/
├── runs/
│   └── <task>-<controller>-s<seed>-<hash>/
│       ├── signals.parquet
│       ├── events.parquet
│       ├── verdict.json
│       ├── meta.json
│       └── log.mcap
└── memory/                 # optional local failure-card index
```

`signals.parquet` is a wide control-step trace; `events.parquet` stores event transitions; `verdict.json` stores spec results; `meta.json` records inputs, hashes, parameters, and environment; `log.mcap` contains JSON-encoded `/signals` and `/events` channels. `query` creates DuckDB views over the run artifacts.

## Determinism and timing

qddot seeds its perturb/noise random generator from the run seed. For a fixed task, controller, seed, perturbation, MuJoCo version, and platform, the intended result is repeatable. Bitwise identity is not promised across MuJoCo releases or platforms. Wall-clock controller timing can vary with machine load and drives deadline misses.

When configured, a controller budget is active even for a nominal run. A late command is applied after a modeled delay; jitter is quantized to physics steps. Fuzz samples task perturbation ranges with per-run seeds derived from the requested seed.

## Re-simulation

`replay`, `view`, and keyframe generation replay the applied controls recorded in the signal trace into a temporary output root, so they do not overwrite the original episode. This supports repeatable inspection without executing a possibly stateful controller again. The logged controls are sampled at control boundaries, so sub-step actuation jitter may leave small state differences. Deadline effects are represented by the resulting applied command sequence.
