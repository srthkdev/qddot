"""Shared subprocess and JSONL helpers."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from bench.agents import AgentResult


def run_command(command: list[str], workdir: Path, prompt: str, budget: dict[str, Any], transcript: Path) -> tuple[str, float, int]:
    timeout = float(budget.get("timeout_s", 900))
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else "")
        code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "") + "\n[timeout]\n"
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        code = 124
    transcript.write_text(output)
    return output, time.monotonic() - started, code


def jsonl_events(output: str) -> list[dict[str, Any]]:
    events = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def claude_parse(output: str) -> dict[str, Any]:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        try:
            data, _ = json.JSONDecoder().raw_decode(output.lstrip())
        except json.JSONDecodeError:
            return {"final_message": output.strip()}
    usage = data.get("usage") or {}
    return {
        "final_message": data.get("result", ""),
        "tokens_in": int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0),
        "tokens_out": int(usage.get("output_tokens", usage.get("outputTokens", 0)) or 0),
        "cost_usd": data.get("total_cost_usd", data.get("cost_usd")),
    }


def codex_parse(output: str) -> dict[str, Any]:
    events = jsonl_events(output)
    final = ""
    in_tokens = out_tokens = 0
    for event in events:
        kind = event.get("type") or event.get("event")
        payload = event.get("item", event.get("data", event))
        if (kind in {"item.completed", "agent_message"} and isinstance(payload, dict)
                and (payload.get("type") in {"agent_message", "message"} or kind == "agent_message")):
            final = payload.get("text", payload.get("content", final))
        if kind in {"turn.completed", "turn_complete"} and isinstance(payload, dict):
            usage = payload.get("usage") or payload.get("token_usage") or {}
            in_tokens = int(usage.get("input_tokens", usage.get("input", in_tokens)) or 0)
            out_tokens = int(usage.get("output_tokens", usage.get("output", out_tokens)) or 0)
    return {"final_message": final, "tokens_in": in_tokens, "tokens_out": out_tokens}


def opencode_parse(output: str) -> dict[str, Any]:
    events = jsonl_events(output)
    text_parts: list[str] = []
    in_tokens = out_tokens = 0
    for event in events:
        part = event.get("part") or event.get("data") or event
        if isinstance(part, dict):
            if part.get("type") in {"text", "text-delta", "message"}:
                chunk = part.get("text", part.get("content", ""))
                if chunk:
                    text_parts.append(str(chunk))
            usage = part.get("tokens") or part.get("usage") or {}
            in_tokens = int(usage.get("input", usage.get("input_tokens", in_tokens)) or 0)
            out_tokens = int(usage.get("output", usage.get("output_tokens", out_tokens)) or 0)
    return {"final_message": "".join(text_parts).strip(), "tokens_in": in_tokens, "tokens_out": out_tokens}


def result_for(command: list[str], workdir: Path, prompt: str, budget: dict[str, Any], parser, transcript: Path) -> AgentResult:
    output, elapsed, code = run_command(command, workdir, prompt, budget, transcript)
    parsed = parser(output)
    return AgentResult(
        final_message=str(parsed.get("final_message", "")),
        transcript_path=transcript,
        wall_s=elapsed,
        tokens_in=int(parsed.get("tokens_in", 0)),
        tokens_out=int(parsed.get("tokens_out", 0)),
        cost_usd=parsed.get("cost_usd"),
        exit_code=code,
    )
