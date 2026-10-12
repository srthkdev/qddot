"""Build compact, source-backed memory cards from recorded episodes."""

from __future__ import annotations

import difflib
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qddot.diag import why
from qddot.diag._data import read_json, read_table
from qddot.spec import SpecError, to_english


def _controller_source(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _sha(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _controller_sha(source: str, meta: dict[str, Any]) -> str:
    return str(meta.get("controller_sha256") or _sha(source))


def _fix_for(episode_dir: Path, task: str, controller_path: str,
             controller_sha: str, source: str) -> str | None:
    runs_dir = episode_dir.parent
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    current_mtime = (episode_dir / "verdict.json").stat().st_mtime
    for verdict_path in runs_dir.glob("*/verdict.json"):
        candidate_dir = verdict_path.parent
        if candidate_dir == episode_dir:
            continue
        try:
            meta = read_json(candidate_dir / "meta.json")
            verdict = read_json(verdict_path)
            mtime = verdict_path.stat().st_mtime
        except (OSError, ValueError):
            continue
        if (verdict.get("passed") and meta.get("task_path")
                and verdict.get("task") == task
                and str(meta.get("controller_path")) == controller_path
                and str(meta.get("controller_sha256")) != controller_sha
                and mtime >= current_mtime):
            candidates.append((mtime, candidate_dir, meta))
    if not candidates:
        return None
    _, _, passing_meta = min(candidates, key=lambda item: item[0])
    passing_source = _controller_source(passing_meta.get("controller_path", ""))
    if not passing_source:
        return None
    return "".join(difflib.unified_diff(
        source.splitlines(keepends=True), passing_source.splitlines(keepends=True),
        fromfile="failing/controller.py", tofile="passing/controller.py",
    ))


def link_fix(episode_dir: str | Path, controller_src: str) -> str | None:
    """Find a later passing run and diff it against a stored failing snapshot."""
    directory = Path(episode_dir).resolve()
    verdict = read_json(directory / "verdict.json")
    meta = read_json(directory / "meta.json")
    controller_path = str(Path(meta.get("controller_path", verdict.get("controller", ""))).resolve())
    controller_sha = str(meta.get("controller_sha256") or _sha(controller_src))
    return _fix_for(directory, str(verdict.get("task", "")), controller_path,
                    controller_sha, controller_src)


def card_from_episode(episode_dir: str | Path) -> dict[str, Any]:
    """Return a compact card and a controller snapshot for one episode."""
    directory = Path(episode_dir).resolve()
    verdict = read_json(directory / "verdict.json")
    meta = read_json(directory / "meta.json")
    controller_path = str(Path(meta.get("controller_path", verdict.get("controller", ""))).resolve())
    controller_src = _controller_source(controller_path)
    controller_sha = _controller_sha(controller_src, meta)
    specs = []
    for spec in verdict.get("specs", []):
        if spec.get("passed"):
            continue
        stl = str(spec.get("stl", ""))
        try:
            english = to_english(stl)
        except SpecError:
            english = stl
        specs.append(f"{spec.get('name', 'spec')}: {english}")
    diagnosis = why(directory)
    causes = list(dict.fromkeys(
        str(cause.get("cause", "")) for cause in diagnosis.get("likely_causes", [])
        if cause.get("cause")
    ))
    _, events = read_table(directory / "events.parquet")
    event_lines = [
        f"{event.get('kind')} {event.get('a', '')}"
        + (f" / {event.get('b')}" if event.get("b") else "")
        + f" at {float(event.get('t', 0.0)):.3f}s"
        for event in events[:8]
    ]
    task = str(verdict.get("task", ""))
    state = "PASS" if verdict.get("passed") else "FAIL"
    sections = [f"{state} {task}"]
    sections.append("Failing specs: " + ("; ".join(specs) if specs else "none"))
    sections.append("Likely causes: " + ("; ".join(causes) if causes else "none reported"))
    sections.append("Key events: " + ("; ".join(event_lines) if event_lines else "none"))
    sections.append(f"Controller sha256: {controller_sha}")
    fix_diff = None
    if not verdict.get("passed"):
        fix_diff = _fix_for(directory, task, controller_path, controller_sha, controller_src)
    return {
        "id": str(directory), "task": task, "episode": str(verdict.get("episode", directory.name)),
        "created": datetime.fromtimestamp((directory / "verdict.json").stat().st_mtime,
                                           tz=UTC).isoformat(),
        "passed": bool(verdict.get("passed")), "text": "\n".join(sections),
        "causes": causes, "fix_diff": fix_diff, "controller_src": controller_src,
        "controller_sha": controller_sha,
    }
