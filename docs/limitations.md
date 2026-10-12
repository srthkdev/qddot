# Limitations

- **Simulation is not hardware verification.** “Sim-checked” describes the supplied model and simulated conditions. Contact, friction, and especially grasping fidelity are limited; hardware calibration and safety validation remain necessary.
- **Determinism is scoped.** Repeatability is intended for the same machine, platform, MuJoCo version, task, controller, seed, and perturbation. Cross-version and cross-platform bitwise identity is not guaranteed.
- **Deadlines use wall time.** Controller compute time is measured on the host. CPU contention and machine speed can change whether a budget is missed. Replay uses recorded applied controls to avoid remeasuring controller time.
- **No firmware/timing emulation.** qddot does not run embedded firmware, model bus timing, or replace hardware-in-the-loop testing.
- **Task metrics depend on task authorship.** A passing spec only means the recorded trace satisfied that formula. A misspecified or incomplete set of specs can produce misleading confidence.
- **Static model checks are heuristic.** `qddot check` identifies common setup issues; a clean result does not establish model correctness.
- **Benchmark metrics have limits.** Overshoot/settling calculations are best-effort and rely on task target names matching recorded signals.

Planned integrations such as ROS 2, Isaac/Newton, Renode, WASM replays, and Yantra RL environments are not current capabilities.
