# Known controller and model failures

The challenge model has polished, nearly frictionless pads with one-dimensional contact. The block is physically pinched, but the contact model cannot provide the tangential support needed during the lift. The fixed model restores compliant frictional pads with a four-dimensional contact basis. The other variants independently break the right finger range or apply an excessive actuator gear.

| Variant | Expected behavior | Minimal repair |
|---|---|---|
| `buggy_friction_assumption.py` | Nominally lifts on the edge of the friction cone; small friction and latency perturbations cause slip and `lift_block` failure, with `contact_end` after the block falls. | Ramp the lift more gently and preserve a wider, stable grasp. |
| `buggy_finger_mapping.py` | The jaw mismatch leaves a weak, off-center grasp; `lift_block` fails and contact ends during the lift. | Calibrate the two symmetric jaw commands identically. |
| `models/buggy_friction.xml` | Good sequence loses the block during lift. | Restore pad friction and tangential contact dimensions. |
| `models/buggy_range.xml` | The inverted right-finger range causes the compiler to drop its limit, which `qddot check` warns about. The fixed grasp trajectory stays inside the travel anyway. | Restore the intended nonnegative closing coordinate range. |
| `models/buggy_gear.xml` | The right jaw produces excess contact force and the grasp is unstable. | Restore the matched actuator gear. |
