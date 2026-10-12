"""Run bundled controller examples and create replay GIFs under docs/media."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
QDDOT = ROOT / ".venv" / "bin" / "qddot"


def run_cli(args: list[str], cwd: Path, *, allow_failure: bool = False) -> str:
    result = subprocess.run(
        [str(QDDOT), *args], cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    print(f"$ qddot {' '.join(args)}")
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode and not allow_failure:
        raise RuntimeError(f"qddot {' '.join(args)} exited with {result.returncode}")
    return result.stdout


def episode_from_output(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("  files: "):
            return line.split("  files: ", 1)[1].split("/  (", 1)[0]
    raise RuntimeError("could not find episode path in qddot run output")


def make_comparison(bad_gif: Path, good_gif: Path, output: Path) -> None:
    with Image.open(bad_gif) as bad_image, Image.open(good_gif) as good_image:
        bad_frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(bad_image)]
        good_frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(good_image)]
    if not bad_frames or not good_frames:
        raise RuntimeError("replay GIF has no frames")
    count = max(len(bad_frames), len(good_frames))
    width = max(frame.width for frame in bad_frames + good_frames)
    height = max(frame.height for frame in bad_frames + good_frames)
    composed: list[Image.Image] = []
    for index in range(count):
        frames = [bad_frames[min(index, len(bad_frames) - 1)],
                  good_frames[min(index, len(good_frames) - 1)]]
        resized = [ImageOps.pad(frame, (width, height)) for frame in frames]
        canvas = Image.new("RGB", (width * 2, height), "white")
        canvas.paste(resized[0], (0, 0))
        canvas.paste(resized[1], (width, 0))
        composed.append(canvas)
    composed[0].save(output, save_all=True, append_images=composed[1:],
                      duration=50, loop=0, optimize=False)


def demo_pair(name: str, task: Path, bad: Path, good: Path, output_dir: Path,
              workdir: Path) -> tuple[Path, Path]:
    bad_output = run_cli(["run", str(task), "--controller", str(bad), "--seed", "0"],
                         workdir, allow_failure=True)
    bad_episode = episode_from_output(bad_output)
    run_cli(["why", bad_episode], workdir)
    good_output = run_cli(["run", str(task), "--controller", str(good), "--seed", "0"], workdir)
    good_episode = episode_from_output(good_output)
    bad_view = run_cli(["view", bad_episode], workdir, allow_failure=True)
    good_view = run_cli(["view", good_episode], workdir, allow_failure=True)
    bad_gif = workdir / bad_episode / "replay.gif"
    good_gif = workdir / good_episode / "replay.gif"
    saved_bad = output_dir / f"{name}-bad.gif"
    saved_good = output_dir / f"{name}-good.gif"
    comparison = output_dir / f"{name}-before-after.gif"
    if "wrote" not in bad_view or "wrote" not in good_view or not bad_gif.is_file() or not good_gif.is_file():
        print(f"warning: replay rendering unavailable for {name}; skipping GIF copies", file=sys.stderr)
        return saved_bad, saved_good
    shutil.copy2(bad_gif, saved_bad)
    shutil.copy2(good_gif, saved_good)
    make_comparison(bad_gif, good_gif, comparison)
    print(f"wrote {comparison}")
    return saved_bad, saved_good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "media")
    args = parser.parse_args()
    if not QDDOT.is_file():
        parser.error(f"qddot executable not found at {QDDOT}; create the repo .venv first")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cartpole = ROOT / "tasks" / "cartpole_balance"
    reach = ROOT / "tasks" / "so101_reach"
    with tempfile.TemporaryDirectory(prefix="qddot-demo-") as temporary:
        workdir = Path(temporary)
        demo_pair("cartpole", cartpole / "task.yaml",
                  cartpole / "controllers" / "buggy_velocity_sign.py",
                  cartpole / "controllers" / "good.py", output_dir, workdir)
        run_cli(["fuzz", str(cartpole / "task.yaml"), "--controller",
                 str(cartpole / "controllers" / "good.py"), "-n", "50",
                 "--workers", "1", "--seed", "0"], workdir)
        demo_pair("so101-reach", reach / "task.yaml",
                  reach / "controllers" / "buggy_jacobian_sign.py",
                  reach / "controllers" / "good.py", output_dir, workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
