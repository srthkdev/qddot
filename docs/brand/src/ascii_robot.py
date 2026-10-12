"""Render a MuJoCo robot to edge-aware ASCII art (used for the qddot logo)."""
import sys
import mujoco, numpy as np
spec = mujoco.MjSpec.from_file('assets/robotstudio_so101/scene.xml')
spec.visual.global_.offwidth = 1600; spec.visual.global_.offheight = 1600
m = spec.compile(); d = mujoco.MjData(m)
J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
for n, v in dict(shoulder_pan=0.4, shoulder_lift=-0.4, elbow_flex=-0.6, wrist_flex=0.7, gripper=0.9).items():
    d.qpos[J(n)] = v
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 1400, 1050)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.05, 0, 0.2]; cam.distance = 0.75; cam.azimuth = 120; cam.elevation = -8
r.update_scene(d, cam); lum = r.render().astype(float).mean(axis=2) / 255
r.enable_segmentation_rendering(); r.update_scene(d, cam); seg = r.render()[..., 0]
mask = np.isin(seg, [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]).astype(float)
ys, xs = np.nonzero(mask); pad = 10
y0, y1, x0, x1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad), max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
cols = int(sys.argv[1]) if len(sys.argv) > 1 else 64
cw = (x1 - x0) / cols; ch = cw * 2.1; rows = int((y1 - y0) / ch)
gy, gx = np.gradient(mask)          # silhouette edges
ly, lx = np.gradient(lum * mask)    # internal shading edges
ramp = " .,:;=+*#%@"
out = []
for i in range(rows):
    line = ""
    for j in range(cols):
        ya, yb, xa, xb = int(y0 + i * ch), int(y0 + (i + 1) * ch), int(x0 + j * cw), int(x0 + (j + 1) * cw)
        cell = mask[ya:yb, xa:xb]
        cov = cell.mean()
        if cell.size == 0 or cov < 0.08:
            line += " "; continue
        ex, ey = gx[ya:yb, xa:xb].sum(), gy[ya:yb, xa:xb].sum()
        mag = np.hypot(ex, ey) / cell.size
        if cov < 0.7 and mag > 0.04:   # boundary cell: draw the edge direction
            ang = (np.degrees(np.arctan2(-ey, ex)) + 90) % 180  # edge runs perpendicular to the gradient
            line += "|" if 67 <= ang < 113 else "_" if ang < 22 or ang >= 158 else "/" if ang < 67 else "\\"
            continue
        v = lum[ya:yb, xa:xb][cell > 0].mean()
        sx, sy = lx[ya:yb, xa:xb].sum(), ly[ya:yb, xa:xb].sum()
        if np.hypot(sx, sy) / cell.size > 0.03:   # strong internal crease (panel lines, joints)
            line += "#"
            continue
        line += ramp[int(np.clip(v, 0, 0.999) * (len(ramp) - 1)) + 0]
    out.append(line.rstrip())
while out and not out[0].strip(): out.pop(0)
print("\n".join(out))
