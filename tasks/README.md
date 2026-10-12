# Public benchmark tasks

The public suite contains 15 MuJoCo tasks. Every task includes its YAML specification, prompt, starter controller, a good controller, and realistic fault controllers. The **Fuzz-only bug** column identifies controllers that pass nominally and are expected to fail under the task perturbations.

| Task | Category | Robot | Specs | Controllers | Fuzz-only bug |
|---|---|---|---|---|---|
| `cartpole_balance` | inverted pendulum | cart-pole | upright_and_hold, stay_on_rail, avoid_saturation | `good`, buggy_latency, buggy_slow, buggy_velocity_sign | `buggy_latency` |
| `diffdrive_waypoints` | mobile navigation | differential drive base | waypoint_one, waypoint_two, waypoint_three_and_safe | `good`, buggy_atan2, buggy_latency, buggy_wrap | `buggy_latency` |
| `double_pendulum_lqr` | balance control | double pendulum | capture_upright, stay_within_workspace, no_saturation | `good`, buggy_latency, buggy_mapping, buggy_wrong_equilibrium | `buggy_latency` |
| `joint_pid_payload` | PID tuning under payload | single-joint arm | overshoot_bound, settle_and_hold, avoid_prolonged_saturation | `good`, buggy_derivative_kick, buggy_dt, buggy_latency, buggy_windup | `buggy_latency` |
| `latency_servo` | timing robustness | rotary servo | track_step, bounded_overshoot, no_joint_stop | `good`, buggy_aggressive, buggy_dt | `buggy_aggressive`, `buggy_dt` |
| `repair_arm_model` | model repair | planar two-link arm | target_reached, joint_safe, no_saturation | `good`, buggy_joint_order, buggy_stale_observation | `buggy_stale_observation` |
| `repair_gripper_model` | model repair and grasping | parallel-jaw gripper | lift_block, establish_grasp, no_actuator_saturation | `good`, buggy_finger_mapping, buggy_friction_assumption | `buggy_friction_assumption` |
| `so101_box_push` | contact manipulation | SO-101 | box_in_goal, box_not_tipped, no_arm_floor_collision | `good`, buggy_latency, buggy_units, buggy_wrong_frame | `buggy_units`, `buggy_latency` |
| `so101_lift_coarse` | contact manipulation | SO-101 | box_in_goal, box_not_tipped, no_arm_floor_collision | `good`, buggy_early_close, buggy_latency, buggy_weak_grip | `buggy_latency` |
| `so101_line_trace` | Cartesian trajectory | SO-101 | start_point, midline_waypoint, final_hold, no_joint_limit | `good`, buggy_latency, buggy_units, buggy_wrong_frame | `buggy_latency` |
| `so101_reach` | Cartesian reach | SO-101 | reach_and_hold, no_joint_limit, no_floor_collision | `good`, buggy_jacobian_sign, buggy_latency, buggy_limit_avoidance | `buggy_latency` |
| `ur5e_cartesian_reach` | Cartesian reach | UR5e | reach_hold, no_joint_limit, no_table_collision | `good`, buggy_frame, buggy_latency, buggy_units | `buggy_units`, `buggy_latency` |
| `ur5e_payload_hold` | payload motion | UR5e | payload_pose, payload_stability, no_joint_limit | `good`, buggy_fragile_gain, buggy_missing_compensation | `buggy_fragile_gain` |
| `ur5e_track` | joint trajectory | UR5e | pan_settle, lift_settle, elbow_settle, no_tracking_overshoot | `good`, buggy_degrees, buggy_joint_order, buggy_latency | `buggy_latency` |
| `wheeled_balancer` | balance control | wheeled base | balance, reach_and_stop, avoid_saturation | `good`, buggy_gain_instability, buggy_latency, buggy_mapping | `buggy_latency` |

Held-out fixtures live in the gitignored `tasks_private/` directory. They cover contact pushing, five-bar reach, pan-tilt tracking, uneven-terrain rover control, and thrust-vector hover. Their task prompts and controller references are intentionally excluded from this public task list.

For model-repair tasks, `fixed_model` identifies the repaired MJCF used to assess the good controller.
