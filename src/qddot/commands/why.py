"""Explain the first failed specification in an episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ._out import console, emit_json, exit_error


def _lines(result: dict[str, Any], frames: bool) -> list[str]:
    lines = [f"{result['verdict']} episode {result.get('episode')}"]
    failed = result.get("first_failing_spec")
    if failed:
        window = failed.get("window")
        window_text = f" window {window[0]:.2f}-{window[1]:.2f}s" if window else ""
        lines.append(f"first failing spec: {failed['name']} at {failed['witness_t']:.2f}s{window_text}")
        lines.append(f"  {failed.get('english', failed.get('stl'))}")
    for item in result.get("signals", []):
        lines.append(f"{item['signal']}: {item['value']:.4g} (window {item['min']:.4g}..{item['max']:.4g}), {item['trend']}")
    for event in result.get("nearby_events", [])[:5]:
        lines.append(f"event {event['kind']} {event.get('a')} @ {float(event['t']):.3f}s")
    first = result.get("first_events", {})
    if first:
        parts = [f"{kind} {event.get('a')} @{float(event['t']):.3f}s"
                 for kind, event in first.items()]
        lines.append("first events: " + "; ".join(parts))
    causes = result.get("likely_causes", [])
    lines.append("likely causes: " + ("; ".join(
        f"{item['cause']} [{item['confidence']}]" for item in causes[:4]) if causes else "none"))
    for cause in causes[:3]:
        lines.append("  evidence: " + "; ".join(cause["evidence"]))
    lines.extend(f"next: {command}" for command in result.get("next_commands", [])[:2])
    if frames and result.get("visual_diff"):
        for item in result["visual_diff"]:
            moved = ", ".join(f"{body['body']} ({body['distance_m'] * 100:.1f}cm)"
                               for body in item["moved_bodies"][:4]) or "none"
            rotated = ", ".join(f"{body['body']} ({body['angle_deg']:.1f}°)"
                                 for body in item.get("rotated_bodies", [])[:4]) or "none"
            lines.append(f"visual {item['baseline']} {item['from_t']:.2f}-{item['to_t']:.2f}s: moved {moved}; rotated {rotated}; contacts {len(item['contacts_before'])}->{len(item['contacts_after'])}")
    return lines[:25]


def register(app: typer.Typer) -> None:
    @app.command()
    def why(
        episode: Path = typer.Argument(...),  # noqa: B008
        frames: bool = typer.Option(False, "--frames"),
        vlm: bool = typer.Option(False, "--vlm"),
        json_output: bool = typer.Option(False, "--json"),
    ) -> None:
        """Diagnose an episode and suggest focused next commands."""
        try:
            from qddot.diag import why as diagnose
            from qddot.diag.replay import keyframes, visual_diff

            result: dict[str, Any] = diagnose(episode)
            if frames and result.get("first_failing_spec"):
                witness = result["first_failing_spec"]["witness_t"]
                from qddot.diag._data import read_json, read_table

                task_path = read_json(episode / "meta.json").get("task_path")
                from qddot.task import load_task

                task = load_task(task_path)
                _, events = read_table(episode / "events.parquet")
                earlier = [float(event["t"]) for event in events if float(event["t"]) < witness]
                first_event = min(earlier) if earlier else max(0.0, witness - 0.5)
                times = [0.0, first_event, witness, task.duration]
                outputs = keyframes(episode, times)
                result["frames"] = [str(path) for path in outputs]
                result["visual_diff"] = visual_diff(episode, times)
            if vlm:
                from qddot.diag import vlm as vlm_module

                result["visual_description"] = vlm_module.describe(result.get("frames", []))
        except Exception as exc:  # noqa: BLE001
            exit_error(exc, json_output)
        if json_output:
            emit_json(result)
        else:
            for line in _lines(result, frames):
                console.print(line)
