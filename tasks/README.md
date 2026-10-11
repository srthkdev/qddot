# Benchmark tasks

| Task | Goal and specs | Controllers |
|---|---|---|
| `so101_reach` | Move the SO-101 gripper frame to the target, hold within 2.5 cm, avoid joint limits and floor contact. | `good.py` damped least-squares IK; `buggy_jacobian_sign.py`; `buggy_limit_avoidance.py` |
| `cartpole_balance` | Recover from a 0.1 rad tilt, keep the pole within 0.05 rad for a hold window, keep the cart within 0.8 m and avoid saturation. Includes a 10 ms controller budget. | `good.py` LQR with a MuJoCo linearization and discrete Riccati solve; `buggy_velocity_sign.py`; `buggy_slow.py` |
| `ur5e_track` | Smoothly track three requested joint targets over 2 s, settle each within 0.04 rad, and limit shoulder-pan overshoot. | `good.py` minimum-jerk joint trajectory; `buggy_joint_order.py`; `buggy_degrees.py` |

Each task folder contains `task.yaml`, controllers and `KNOWN_FAILURES.md` with nominal failure ground truth.
