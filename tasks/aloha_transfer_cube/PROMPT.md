# ALOHA cube handover

Implement a controller for the ALOHA 2 dual-arm scene that transfers the red cube from the right gripper to the left gripper. Start from the provided neutral keyframe, approach and lift the cube from the table, bring it to the shared workspace, complete a two-hand handover, and finish with the left arm holding the cube clear of the table.

Use the available position actuators and remain within their ranges. The episode is six simulated seconds. The controller receives joint and actuator observations, cube position signals, and distances from the cube to each gripper. Verify the required motion and collision constraints using the task's specifications.
