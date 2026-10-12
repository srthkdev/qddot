"""Cached end-to-end cause accuracy check for all public benchmark controllers."""

import pytest

from bench.why_eval import DEFAULT_CACHE, PUBLIC_TASKS, evaluate

# Public run achieved 38.5%; keep the regression floor five percentage points lower.
PUBLIC_TOP1_FLOOR = 0.335


def test_public_ground_truth_accuracy_and_good_runs() -> None:
    report = evaluate(PUBLIC_TASKS, DEFAULT_CACHE, run_missing=True)
    assert report["top1"] >= PUBLIC_TOP1_FLOOR
    assert len(report["tasks"]) >= 15
    assert report["good_failures"] == []


def test_cached_public_hypotheses_use_the_fixed_taxonomy() -> None:
    taxonomy = {
        "sign_or_frame", "units", "gain_instability", "saturation", "joint_limit", "windup",
        "latency_sensitivity", "deadline", "lost_contact", "wrong_equilibrium",
        "missing_compensation", "wrong_mapping", "model_error",
    }
    episodes = list(DEFAULT_CACHE.glob("*/buggy_*/nominal/**/verdict.json"))
    if not episodes:
        pytest.skip("public accuracy test populated no nominal episode cache")
    from qddot.diag import why

    for verdict_path in episodes:
        diagnosis = why(verdict_path.parent)
        assert all(item.get("category") in taxonomy for item in diagnosis["likely_causes"])
