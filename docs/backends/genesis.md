# Genesis backend

Genesis is an optional, in-process CPU backend. Keep it in a separate Python
3.12 environment because its torch stack can conflict with qddot's optional
memory dependencies. Install qddot in that environment, then install Genesis
1.4.4:

```sh
python3.12 -m venv .venv-genesis
. .venv-genesis/bin/activate
python -m pip install -e .
python -m pip install genesis-world==1.4.4
```

Select it on a loaded task before calling `run_episode`:

```python
task = load_task("tasks/cartpole_balance/task.yaml")
task.backend = "genesis"
result = run_episode(task, controller_path)
```

Genesis initialization is guarded and uses its CPU backend with deterministic
algorithms enabled. MJCF is loaded through `gs.morphs.MJCF`. The backend uses
MuJoCo to read source MJCF metadata and preserve joint, actuator, site, and
keyframe names and ordering; dynamics and collision resolution are Genesis.

## Supported MJCF

The implementation is intended for rigid-body robot models using hinge and
slide joints, standard primitive collision geoms, named bodies and sites,
gravity, joint damping, motor actuators, and affine position actuators attached
to 1-DoF joints. It applies reset keyframes and per-joint position overrides,
scales geom sliding friction, and scales link masses and inertia tensors.
Actuator forces are applied as generalized forces/torques. Contact records are
aggregated per link pair; Genesis contact force is projected onto its contact
normal, slip speed is estimated from the two links' velocities at the contact
point, and penetration is taken from the engine's collision result.

Genesis' MJCF importer does not reproduce every MuJoCo feature. Unsupported or
not fully equivalent features include free and ball-joint state mapping,
multiple actuators driving one joint, tendon and site transmissions, actuator
internal dynamics, non-fixed actuator gain/bias types, full MuJoCo solver
parameters, equality constraints, mocap bodies, flex/soft bodies, and MuJoCo's
visual-only asset semantics. Mesh geometry may be convexified by Genesis.
Position actuator behavior is represented using the source model's compiled
affine gain and bias coefficients. Genesis' solver and collision response are
not bitwise equivalent to MuJoCo.

The backend is headless and does not render. Genesis 1.4.4 initializes its
OpenGL rasterizer as part of scene construction, so the backend skips rasterizer
construction; this does not affect its physics scene.

## Measured comparison

Measured on the repository's 4.0 s `cartpole_balance` task, seed 0, the good
controller, and explicit nominal perturbations (noise, latency, jitter, and
controller deadline disabled):

| Backend | Verdict | Runner wall time |
|---|---:|---:|
| MuJoCo | PASS | 0.155 s |
| Genesis 1.4.4 CPU | PASS | 7.718 s |

The Genesis wall time includes scene setup and first-use kernel compilation.
On this machine it took about 50 times as long as MuJoCo for this short run.

On `tests/backends/fixtures/box_drop.xml`, after 1.0 simulated second, Genesis
reported 19.6200 N summed normal force for the resting 2 kg box; MuJoCo reported
19.6200 N summed across its four contact points. The relative force difference
was below 0.001%. Genesis combines the four manifold contacts into one
body-pair record. This comparison is fixture-specific; contact solver forces
can differ more for impacts, frictional contacts, or other solver settings.

Runs are repeatable within one process and machine with the same Genesis,
torch, and model versions. CPU execution and deterministic algorithms are
selected explicitly, but cross-version and cross-machine bitwise identity is
not guaranteed.
