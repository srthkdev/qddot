"""Render the hero demo: the same agent on the ALOHA handover without and with qddot.

    python scripts/make_hero_video.py LEFT.py RIGHT.py --out docs/media/hero/aloha.mp4

Both controllers are re-simulated deterministically on the bundled task; each
panel shows the live spec status and ends on the real verdict.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from qddot.sim.runner import run_episode
from qddot.task import load_task

TASK = Path(__file__).resolve().parents[1] / "tasks" / "aloha_transfer_cube" / "task.yaml"
W, H, PANEL_W, PANEL_H, FPS = 1920, 1080, 940, 705, 30
CAMERA = {"lookat": [0.0, -0.02, 0.15], "distance": 0.85, "azimuth": 105, "elevation": -22}
BG, INK, DIM, OK, BAD, ACCENT = "#06090e", "#e8eef7", "#7f8da3", "#5fe08a", "#ff5f6d", "#ff9f43"
FONT = "/System/Library/Fonts/Menlo.ttc"


def render(controller: Path) -> tuple[list[np.ndarray], dict, list[dict]]:
    task = load_task(TASK)
    frames: list[np.ndarray] = []
    renderer: list[mujoco.Renderer] = []
    every = max(1, round(1 / (FPS * task.control_dt)))

    def capture(index: int, t: float, model, data) -> None:
        if not renderer:
            renderer.append(mujoco.Renderer(model, PANEL_H, PANEL_W))
        if (index + 1) % every == 0:
            cam = mujoco.MjvCamera()
            cam.lookat[:], cam.distance = CAMERA["lookat"], CAMERA["distance"]
            cam.azimuth, cam.elevation = CAMERA["azimuth"], CAMERA["elevation"]
            renderer[0].update_scene(data, cam)
            frames.append(renderer[0].render().copy())

    result = run_episode(task, controller.resolve(), root=Path(tempfile.mkdtemp()), on_step=capture,
                         record_mcap=False)
    import pyarrow.parquet as pq

    events = pq.read_table(result.episode_dir / "events.parquet").to_pylist()
    return frames, result.verdict, [e for e in events if e["kind"] == "collision"]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def compose(left: tuple, right: tuple, labels: tuple[str, str], out: Path) -> None:
    (lf, lv, le), (rf, rv, re_) = left, right
    n = max(len(lf), len(rf))
    hold = FPS * 2  # linger on the verdict
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                             "yuv420p", "-crf", "18", str(out)], stdin=subprocess.PIPE)
    for i in range(n + hold):
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 28), "qddot", font=font(40), fill=INK)
        for dx in (54, 74):  # the double dot of q-double-dot
            draw.ellipse((40 + dx - 52, 22, 40 + dx - 44, 30), fill=ACCENT)
        draw.text((215, 40), "ALOHA 2 bimanual handover · same agent: Codex GPT-6 Luna (high)",
                  font=font(22), fill=DIM)
        for k, (frames, verdict, events, label) in enumerate(((lf, lv, le, labels[0]), (rf, rv, re_, labels[1]))):
            x = 20 + k * (PANEL_W + 20)
            canvas.paste(Image.fromarray(frames[min(i, len(frames) - 1)]), (x, 120))
            draw.rectangle((x, 120, x + PANEL_W - 1, 120 + PANEL_H - 1), outline="#26324a", width=2)
            draw.text((x, 845), label, font=font(24), fill=INK)
            t = min(i, len(frames) - 1) / FPS
            draw.text((x, 885), f"t = {t:4.1f} s", font=font(20), fill=DIM)
            hits = [e for e in events if e["t"] <= t]
            if hits:  # a real collision event from the simulation, kept on screen once it happens
                first = hits[0]
                text = f"COLLISION  {first['a']} x {first['b']}  t={first['t']:.2f}s"
                width = draw.textlength(text, font=font(22))
                draw.rectangle((x + 16, 136, x + 44 + width, 186), fill="#3a0d14", outline=BAD, width=2)
                draw.text((x + 30, 146), text, font=font(22), fill=BAD)
            if i >= len(frames):  # verdict card
                passed = verdict["passed"]
                draw.rectangle((x, 925, x + PANEL_W - 1, 1050), fill="#0c1220", outline=OK if passed else BAD,
                               width=3)
                draw.text((x + 24, 940), "PASS" if passed else "FAIL", font=font(44), fill=OK if passed else BAD)
                ordered = sorted(verdict["specs"], key=lambda spec: spec["passed"])  # failures first
                for row, spec in enumerate(ordered[:3]):
                    mark = "✓" if spec["passed"] else "✗"
                    draw.text((x + 200, 940 + row * 34), f"{mark} {spec['name']}  ({spec['robustness']:+.3f})",
                              font=font(20), fill=OK if spec["passed"] else BAD)
        draw.text((W - 40 - draw.textlength("github.com/srthkdev/qddot", font=font(22)), 40),
                  "github.com/srthkdev/qddot", font=font(22), fill=ACCENT)
        proc.stdin.write(np.asarray(canvas).tobytes())
    proc.stdin.close()
    proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="without qddot: one blind run")
    parser.add_argument("--right-label", default="with qddot: run · why · fuzz")
    parser.add_argument("--out", type=Path, default=Path("docs/media/hero/aloha.mp4"))
    args = parser.parse_args()
    compose(render(args.left), render(args.right), (args.left_label, args.right_label), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
