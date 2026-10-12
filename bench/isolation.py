"""Keep agents away from the answers during benchmark runs.

Three layers: agents get qddot from a venv outside the repository (so no tool
path reveals where the reference solutions live); secret files in the repo are
made unreadable for the duration of a run; every transcript is audited for
any sign the agent reached them anyway.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path

from bench.workdir import is_secret

TOOLS_VENV = Path.home() / ".cache" / "qddot" / "bench-venv"


def tools_dir(repo_root: Path) -> Path:
    """A non-editable qddot install outside the repo, rebuilt when the source changes."""
    digest = hashlib.sha256()
    for path in sorted((repo_root / "src").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(path.relative_to(repo_root).as_posix().encode() + path.read_bytes())
    stamp = TOOLS_VENV / ".qddot-src"
    if not stamp.is_file() or stamp.read_text() != digest.hexdigest():
        subprocess.run(["uv", "venv", "-q", "--allow-existing", "--python", "3.12", str(TOOLS_VENV)], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--reinstall-package", "qddot",
                        "--python", str(TOOLS_VENV / "bin" / "python"), str(repo_root)], check=True)
        stamp.write_text(digest.hexdigest())
    return TOOLS_VENV / "bin"


def secret_files(repo_root: Path) -> list[Path]:
    roots = [repo_root / "tasks", repo_root / "tasks_private"]
    return [p for root in roots if root.is_dir() for p in root.rglob("*") if p.is_file() and is_secret(p)]


@contextmanager
def secrets_locked(repo_root: Path):
    """chmod 000 every secret file for the duration, restoring modes even on signals."""
    modes = {path: path.stat().st_mode for path in secret_files(repo_root)}

    def restore(*_args) -> None:
        for path, mode in modes.items():
            if path.exists():
                os.chmod(path, mode)

    previous = {sig: signal.signal(sig, lambda s, f: (restore(), os._exit(128 + s)))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    atexit.register(restore)
    try:
        for path in modes:
            os.chmod(path, 0)
        yield list(modes)
    finally:
        restore()
        atexit.unregister(restore)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def audit(transcript: str, repo_root: Path) -> list[str]:
    """Evidence an agent reached outside its workdir toward the answers."""
    hits = []
    if str(repo_root) in transcript:
        hits.append(f"mentions repo path {repo_root}")
    hits += [f"mentions {name}" for name in ("good.py", "KNOWN_FAILURES", "expected.yaml", "model_fixed")
             if name in transcript]
    if "buggy_" in transcript:
        hits.append("mentions buggy_ controllers")
    return hits
