# Performance

Measured with `python -m bench.perf` on arm64 15.7.7, 10 cores, MuJoCo 3.15.0, median of 3 runs.

| Task | DoF | Sim time | Wall time | Real-time factor | Physics steps/s |
|---|---:|---:|---:|---:|---:|
| aloha_transfer_cube | 22 | 6.0 s | 0.45 s | 13x | 6,718 |
| cartpole_balance | 2 | 4.0 s | 0.08 s | 48x | 23,917 |
| diffdrive_waypoints | 5 | 6.0 s | 0.11 s | 57x | 28,296 |
| double_pendulum_lqr | 2 | 4.0 s | 0.03 s | 118x | 58,918 |
| joint_pid_payload | 1 | 4.0 s | 0.02 s | 168x | 84,100 |
| latency_servo | 1 | 3.3 s | 0.02 s | 133x | 132,729 |
| repair_arm_model | 2 | 3.0 s | 0.02 s | 152x | 76,192 |
| repair_gripper_model | 9 | 3.0 s | 0.09 s | 35x | 17,275 |
| so101_box_push | 12 | 4.5 s | 0.37 s | 12x | 2,458 |
| so101_line_trace | 12 | 3.5 s | 0.11 s | 32x | 6,414 |
| so101_reach | 6 | 4.0 s | 0.07 s | 57x | 11,307 |
| ur5e_cartesian_reach | 6 | 5.0 s | 0.06 s | 78x | 38,945 |
| ur5e_obstacle_avoid | 6 | 5.5 s | 0.08 s | 69x | 34,323 |
| ur5e_payload_hold | 6 | 5.0 s | 0.06 s | 81x | 40,615 |
| ur5e_track | 6 | 4.0 s | 0.04 s | 91x | 45,438 |
| wheeled_balancer | 4 | 5.0 s | 0.07 s | 67x | 33,499 |

| Command | Latency |
|---|---:|
| `qddot --help (startup)` | 0.19 s |
| `run (4 s cart-pole episode)` | 0.41 s |
| `why` | 0.30 s |
| `why --frames` | 0.67 s |
| `query (events by kind)` | 0.17 s |
| `diff` | 0.31 s |
| `fuzz -n 20 (all cores)` | 1.25 s |
| `similar (warm memory)` | 0.58 s |
| `view (GIF + HTML)` | 0.96 s |
