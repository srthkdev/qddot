"""Create a self-contained offline animation and episode report."""

from __future__ import annotations

import base64
import html
import json
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import mujoco
from PIL import Image, ImageDraw


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _render_frames(episode_dir: Path) -> list[Image.Image]:
    meta = _read_json(episode_dir / "meta.json")
    task_path = Path(meta["task_path"])
    controller_path = Path(meta["controller_path"])
    if not task_path.is_absolute():
        task_path = (episode_dir / task_path).resolve()
    if not controller_path.is_absolute():
        controller_path = (episode_dir / controller_path).resolve()
    from qddot.sim.runner import run_episode
    from qddot.task import load_task

    frames: list[Image.Image] = []
    next_frame_t = 0.0
    renderer: mujoco.Renderer | None = None
    renderer_failed = False
    camera = mujoco.MjvCamera()

    def capture(_index: int, t: float, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        nonlocal next_frame_t, renderer, renderer_failed
        if t + 1e-9 < next_frame_t:
            return
        if not renderer_failed:
            try:
                if renderer is None:
                    renderer = mujoco.Renderer(model, height=240, width=320)
                    mujoco.mjv_defaultCamera(camera)
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = model.stat.center
                    camera.distance = max(2.0, float(model.stat.extent) * 2.5)
                    camera.azimuth = 135.0
                    camera.elevation = -20.0
                renderer.update_scene(data, camera)
                frames.append(Image.fromarray(renderer.render()).convert("RGB"))
            except Exception:  # noqa: BLE001 - Some headless systems lack a GL context.
                renderer_failed = True
                if renderer is not None:
                    renderer.close()
                    renderer = None
                frames.append(_skeleton_frame(model, data, t))
        else:
            frames.append(_skeleton_frame(model, data, t))
        next_frame_t = t + 0.05

    try:
        with tempfile.TemporaryDirectory(prefix="qddot-view-") as temp_root:
            run_episode(
                load_task(task_path), controller_path, seed=int(meta.get("seed", 0)),
                perturb=meta.get("perturb"), root=Path(temp_root), record_mcap=False,
                on_step=capture, compute_ms_override=meta.get("compute_ms"),
            )
    finally:
        if renderer is not None:
            renderer.close()
    if not frames:
        raise RuntimeError("re-simulation produced no renderable frames")
    return frames


def _skeleton_frame(model: mujoco.MjModel, data: mujoco.MjData, t: float) -> Image.Image:
    """Draw body positions as a portable fallback when no OpenGL context exists."""
    image = Image.new("RGB", (320, 240), "#101827")
    draw = ImageDraw.Draw(image)
    if model.nbody <= 1:
        draw.text((16, 16), "MuJoCo model", fill="#e7edf7")
        return image
    positions = data.xpos[1:model.nbody]
    xs = positions[:, 0]
    zs = positions[:, 2]
    x_span = max(float(xs.max() - xs.min()), 0.1)
    z_span = max(float(zs.max() - zs.min()), 0.1)
    scale = min(270.0 / x_span, 170.0 / z_span)
    center_x = float(xs.min() + xs.max()) / 2
    center_z = float(zs.min() + zs.max()) / 2

    def pixel(body_id: int) -> tuple[float, float]:
        pos = data.xpos[body_id]
        return 160 + (float(pos[0]) - center_x) * scale, 125 - (float(pos[2]) - center_z) * scale

    for body_id in range(1, model.nbody):
        parent = int(model.body_parentid[body_id])
        if parent > 0:
            draw.line((pixel(parent), pixel(body_id)), fill="#65b8ff", width=5)
    for body_id in range(1, model.nbody):
        x, y = pixel(body_id)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body {body_id}"
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#ffbd69", outline="#fff1d6")
        draw.text((x + 8, y - 8), name[:16], fill="#dbe7f6")
    draw.text((12, 12), f"headless re-simulation · {t:.2f}s", fill="#e7edf7")
    return image


def _events(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "events.parquet"
    if not path.exists():
        return []
    rows = duckdb.connect(database=":memory:").execute(
        "SELECT t, kind, a, b FROM read_parquet(?) ORDER BY t", [str(path)]
    ).fetchall()
    return [{"t": float(t), "kind": str(kind), "a": str(a), "b": b} for t, kind, a, b in rows]


def _timeline(events: list[dict[str, Any]], verdict: dict[str, Any], end_time: float) -> str:
    width, height, left, right = 760, 160, 92, 18
    span = max(float(end_time), 1e-9)

    def x_at(t: float) -> float:
        return left + (min(max(t, 0.0), span) / span) * (width - left - right)

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Episode events and spec windows">',
             '<line x1="92" y1="126" x2="742" y2="126" stroke="#526078" stroke-width="2"/>']
    for tick in range(5):
        value = span * tick / 4
        xpos = x_at(value)
        parts.append(f'<line x1="{xpos:.1f}" y1="122" x2="{xpos:.1f}" y2="131" stroke="#526078"/>')
        parts.append(f'<text x="{xpos:.1f}" y="148" text-anchor="middle">{value:.2f}s</text>')
    for idx, item in enumerate(verdict.get("specs", [])):
        y = 24 + idx * 24
        window = item.get("window")
        if window:
            x1, x2 = x_at(float(window[0])), x_at(float(window[1]))
            parts.append(f'<rect x="{x1:.1f}" y="{y}" width="{max(x2-x1, 2):.1f}" height="11" rx="4" fill="#496b96" opacity=".9"/>')
        label = html.escape(str(item.get("name", "spec")))
        parts.append(f'<text x="{left-10}" y="{y+10}" text-anchor="end">{label}</text>')
        witness = item.get("witness_t")
        if witness is not None:
            wx = x_at(float(witness))
            parts.append(f'<line x1="{wx:.1f}" y1="{y-4}" x2="{wx:.1f}" y2="{y+15}" stroke="#ffbd69" stroke-width="2"/>')
    event_y = 115
    colors = {"collision": "#fb7185", "contact_start": "#f97316", "slip": "#facc15",
              "deadline_miss": "#c084fc", "joint_limit": "#38bdf8", "saturation": "#fb923c"}
    for event in events:
        xpos = x_at(event["t"])
        color = colors.get(event["kind"], "#94a3b8")
        parts.append(f'<circle cx="{xpos:.1f}" cy="{event_y}" r="4" fill="{color}"><title>{html.escape(event["kind"])} at {event["t"]:.3f}s: {html.escape(event["a"])}</title></circle>')
    parts.extend(['<text x="14" y="118">events</text>',
                  '<text x="98" y="15" fill="#ffbd69">│ witness</text>', '</svg>'])
    return "".join(parts)


def _build_html(episode_dir: Path, gif_path: Path, verdict: dict[str, Any],
                events: list[dict[str, Any]]) -> str:
    encoded = base64.b64encode(gif_path.read_bytes()).decode("ascii")
    end_time = float(verdict.get("stats", {}).get("control_steps", 1))
    signals_path = episode_dir / "signals.parquet"
    if signals_path.is_file():
        end_time = float(duckdb.connect(database=":memory:").execute(
            "SELECT max(t) FROM read_parquet(?)", [str(signals_path)]
        ).fetchone()[0] or end_time)
    passed = bool(verdict.get("passed"))
    specs = "".join(
        f'<li><span class="pill {"pass" if item.get("passed") else "fail"}">{"PASS" if item.get("passed") else "FAIL"}</span> '
        f'<b>{html.escape(str(item.get("name", "spec")))}</b> · robustness '
        f'{html.escape(str(item.get("robustness", "n/a")))}</li>'
        for item in verdict.get("specs", [])
    ) or '<li class="muted">No formal specs recorded</li>'
    timeline = _timeline(events, verdict, end_time)
    episode = html.escape(str(verdict.get("episode", episode_dir.name)))
    status = "PASS" if passed else "FAIL"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>qddot replay · {episode}</title><style>
:root{{color-scheme:dark;background:#0b1020;color:#e7edf7;font:15px/1.5 system-ui,sans-serif}}
body{{max-width:1060px;margin:36px auto;padding:0 22px}}header,.panel{{background:#121a2b;border:1px solid #27334a;border-radius:14px;padding:22px;margin:18px 0}}
h1{{margin:0;font-size:24px}}.sub,.muted{{color:#9aa9c1}}.status{{float:right;font-weight:700;color:{'#55d6a2' if passed else '#fb7185'}}}
img{{display:block;width:min(100%,760px);margin:14px auto;border-radius:10px;background:#080d18}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}ul{{padding-left:0;list-style:none}}li{{margin:9px 0}}.pill{{font-size:11px;font-weight:700;padding:3px 7px;border-radius:20px}}.pass{{background:#153c32;color:#69e3b2}}.fail{{background:#49222b;color:#ff8797}}
svg{{width:100%;height:auto}}svg text{{font:11px system-ui,sans-serif;fill:#a8b5ca}}.timeline{{overflow-x:auto}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}body{{margin:14px auto}}}}
</style></head><body>
<header><span class="status">{status}</span><h1>Episode replay</h1><div class="sub">{episode} · {html.escape(str(verdict.get('task', 'task')))} · seed {html.escape(str(verdict.get('seed', '—')))}</div></header>
<section class="panel"><img alt="Re-simulated MuJoCo episode animation" src="data:image/gif;base64,{encoded}"></section>
<div class="grid"><section class="panel"><h2>Verdict</h2><p>Overall robustness: <b>{html.escape(str(verdict.get('robustness', 'n/a')))}</b></p><ul>{specs}</ul></section>
<section class="panel"><h2>Run details</h2><p>Control steps: {html.escape(str(verdict.get('stats', {}).get('control_steps', '—')))}<br>Physics steps: {html.escape(str(verdict.get('stats', {}).get('physics_steps', '—')))}<br>Events: {len(events)}</p></section></div>
<section class="panel timeline"><h2>Events and spec windows</h2>{timeline}</section>
</body></html>'''


def create_view(episode_dir: str | Path, open_browser: bool = False) -> tuple[Path, Path]:
    """Re-simulate and write a GIF plus a self-contained HTML report."""
    episode_dir = Path(episode_dir).resolve()
    verdict = _read_json(episode_dir / "verdict.json")
    frames = _render_frames(episode_dir)
    gif_path = episode_dir / "replay.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50,
                   loop=0, optimize=False, disposal=2)
    events = _events(episode_dir)
    html_path = episode_dir / "replay.html"
    html_path.write_text(_build_html(episode_dir, gif_path, verdict, events), encoding="utf-8")
    if open_browser:
        import webbrowser

        webbrowser.open(html_path.as_uri())
    return gif_path, html_path
