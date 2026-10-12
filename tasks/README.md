# Public benchmark tasks

The public suite contains 15 MuJoCo tasks. Each task includes a YAML specification, prompt, starter, good controller, and realistic fault controllers. Fuzz-only faults pass nominally and fail under perturbation.

| Task | Category | Robot | Specs | Controllers | Fuzz-only bug |
|---|---|---|---|---|---|
| `cartpole_balance` | inverted pendulum | cart-pole | upright_and_hold, stay_on_rail, avoid_saturation | `buggy_latency, buggy_slow, buggy_velocity_sign, good` | `buggy_latency` |
| `diffdrive_waypoints` | mobile navigation | differential drive base | waypoint_one, waypoint_two, waypoint_three_and_safe | `buggy_atan2, buggy_latency, buggy_wrap, good` | `buggy_latency` |
| `double_pendulum_lqr` | balance control | double pendulum | capture_upright, stay_within_workspace, no_saturation | `buggy_mapping, buggy_mass_tuned, buggy_wrong_equilibrium, good` | `buggy_mass_tuned` |
| `joint_pid_payload` | payload PID tuning | single-joint arm | overshoot_bound, settle_and_hold, avoid_prolonged_saturation | `buggy_derivative_kick, buggy_dt, buggy_unfiltered_derivative, buggy_windup, good` | `buggy_unfiltered_derivative` |
| `latency_servo` | timing robustness | rotary servo | track_step, bounded_overshoot, no_joint_stop | `buggy_aggressive, buggy_dt, good` | `buggy_aggressive, buggy_dt` |
| `repair_arm_model` | model repair | planar two-link arm | target_reached, joint_safe, no_saturation | `buggy_joint_order, buggy_stale_observation, good` | `buggy_stale_observation` |
| `repair_gripper_model` | model repair and grasping | parallel-jaw gripper | lift_block, establish_grasp, no_actuator_saturation | `buggy_finger_mapping, buggy_friction_assumption, good` | `buggy_friction_assumption` |
| `so101_box_push` | contact manipulation | SO-101 | box_in_goal, box_not_tipped, no_arm_floor_collision | `buggy_stale_push_clock, buggy_wrong_frame, good` | `buggy_stale_push_clock` |
| `so101_line_trace` | Cartesian trajectory | SO-101 | start_point, midline_waypoint, final_hold, no_joint_limit | `buggy_noise_amplification, buggy_units, buggy_wrong_frame, good` | `buggy_noise_amplification` |
| `so101_reach` | Cartesian reach | SO-101 | reach_and_hold, no_joint_limit, no_floor_collision | `buggy_jacobian_sign, buggy_latency, buggy_limit_avoidance, good` | `buggy_latency` |
| `ur5e_cartesian_reach` | Cartesian reach | UR5e | reach_hold, no_joint_limit, no_table_collision | `buggy_feedback_gain, buggy_frame, good` | `buggy_feedback_gain` |
| `ur5e_obstacle_avoid` | obstacle avoidance | UR5e | reach_goal, collision_free, no_joint_limit | `buggy_age_scale, buggy_direct_path, good` | `buggy_age_scale` |
| `ur5e_payload_hold` | payload motion | UR5e | payload_pose, payload_stability, no_joint_limit | `buggy_fragile_gain, buggy_missing_compensation, good` | `buggy_fragile_gain` |
| `ur5e_track` | joint trajectory | UR5e | pan_settle, lift_settle, elbow_settle, no_tracking_overshoot | `buggy_degrees, buggy_joint_order, buggy_phase_margin, good` | `buggy_phase_margin` |
| `wheeled_balancer` | balance control | wheeled base | balance, reach_and_stop, avoid_saturation | `buggy_damping_margin, buggy_gain_instability, buggy_mapping, good` | `buggy_damping_margin` |

Held-out fixtures live in the gitignored `tasks_private/` directory. They cover contact pushing, five-bar reach, pan-tilt tracking, uneven-terrain rover control, and thrust-vector hover. Their task prompts and controller references are intentionally excluded from this public task list.

For model-repair tasks, `fixed_model` identifies the repaired MJCF used to assess the good controller.
