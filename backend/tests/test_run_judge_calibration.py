import asyncio
import hashlib
import json

import pytest

from evaluation import run_judge_calibration as runner
from evaluation.evaluator import QualityScores


def test_checkpoints_and_stops_on_first_failure(monkeypatch, tmp_path):
    calls = []
    options = {}

    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)

        async def close(self):
            pass

    class Judge:
        JUDGE_PROMPT = "fixed test prompt"

        def __init__(self, client, model):
            pass

        async def judge(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                return QualityScores(0.5, 0.5, 0.5, 0.5, judge_failed=True, error="quota exhausted")
            return QualityScores(0.8, 0.7, 0.6, 0.9)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model")
    monkeypatch.setattr(runner, "AsyncAnthropic", Client)
    monkeypatch.setattr(runner, "LLMJudge", Judge)
    payload = {"cases": [{"case_id": str(i), "question": "q", "response": "a"} for i in range(3)]}
    output = tmp_path / "checkpoint.json"
    assert asyncio.run(runner.score_workbook(payload, 3, output)) == 2
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["cases"][0]["judge_scores"]["accuracy"] == 0.7
    assert saved["cases"][1]["judge_scores"]["judge_failed"]
    assert "judge_scores" not in saved["cases"][2]
    assert saved["metadata"]["status"] == "judge_scoring_stopped_on_failure"
    assert options["max_retries"] == 0


def test_resume_preserves_failure_history_and_completed_scores(monkeypatch, tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "scored.json"
    source.write_text(json.dumps({"cases": [{"case_id": "ok"}, {"case_id": "retry"}]}), encoding="utf-8")
    success = {"accuracy": 0.75, "judge_failed": False}
    failed = {"accuracy": 0.5, "judge_failed": True, "error": "Connection error"}
    checkpoint = {
        "metadata": {"judge_input_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "cases": [{"case_id": "ok", "judge_scores": success}, {"case_id": "retry", "judge_scores": failed}],
    }
    output.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="--retry-failed"):
        runner.main([str(source), str(output), "--execute"])

    async def inspect_resume(payload, max_cases, output_path):
        assert payload["cases"][0]["judge_scores"] == success
        assert payload["cases"][1]["judge_scores"] is None
        assert payload["cases"][1]["judge_failed_attempts"] == [failed]
        return 0

    monkeypatch.setattr(runner, "score_workbook", inspect_resume)
    assert runner.main([str(source), str(output), "--execute", "--retry-failed"]) == 0
