"""为校准工作表生成 LLM-as-Judge 分数。

该脚本会产生真实模型调用。只有显式传入 ``--execute`` 时才运行。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from evaluation.evaluator import LLMJudge


def _context_text(case: Dict[str, Any]) -> str:
    return json.dumps(case.get("reference_context") or {}, ensure_ascii=False, sort_keys=True)


def _checkpoint(payload: Dict[str, Any], output: Optional[pathlib.Path]) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)


async def score_workbook(
    payload: Dict[str, Any], max_cases: int, output: Optional[pathlib.Path] = None
) -> int:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    if not api_key or not model:
        raise RuntimeError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required")
    options: Dict[str, Any] = {"api_key": api_key, "max_retries": 0, "timeout": 60.0}
    if base_url:
        options["base_url"] = base_url
    metadata = payload.setdefault("metadata", {})
    prompt_hash = hashlib.sha256(LLMJudge.JUDGE_PROMPT.encode("utf-8")).hexdigest()
    if metadata.get("judge_model", model) != model or metadata.get("judge_prompt_sha256", prompt_hash) != prompt_hash:
        raise ValueError("Model or judge prompt changed; use a separate evaluation output")
    metadata.update(judge_model=model, judge_prompt_sha256=prompt_hash, sdk_max_retries=0)
    metadata["status"] = "judge_scoring_in_progress"
    _checkpoint(payload, output)
    client = AsyncAnthropic(**options)
    judge = LLMJudge(client, model)

    scored = 0
    for case in payload.get("cases") or []:
        if case.get("judge_scores") is not None:
            continue
        if scored >= max_cases:
            break
        result = await judge.judge(
            question=str(case.get("question", "")),
            response=str(case.get("response", "")),
            context=_context_text(case),
        )
        case["judge_scores"] = {
            "relevance": result.relevance,
            "accuracy": result.accuracy,
            "completeness": result.completeness,
            "helpfulness": result.helpfulness,
            "judge_failed": result.judge_failed,
            "error": result.error,
        }
        if not result.judge_failed and not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (result.relevance, result.accuracy, result.completeness, result.helpfulness)
        ):
            case["judge_scores"].update(judge_failed=True, error="Judge scores outside [0, 1]")
        scored += 1
        metadata["judge_scored_at"] = datetime.now(timezone.utc).isoformat()
        failed = case["judge_scores"]["judge_failed"]
        metadata["status"] = "judge_scoring_stopped_on_failure" if failed else "judge_scoring_in_progress"
        _checkpoint(payload, output)
        print(f"{case.get('case_id')}: {'FAILED; stopping' if failed else 'scored'}", flush=True)
        if failed:
            break
    if all(c.get("judge_scores") is not None and not c["judge_scores"].get("judge_failed") for c in payload.get("cases") or []):
        metadata["status"] = "judge_scoring_complete"
    _checkpoint(payload, output)
    await client.close()
    return scored


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run real LLM judge calls for calibration")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Preserve failed attempts and retry only failed/unscored cases")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.execute:
        raise RuntimeError("real model calls require the explicit --execute flag")
    if args.max_cases < 1:
        raise ValueError("--max-cases must be positive")

    root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env.local", override=True)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("metadata", {}).get("judge_input_sha256") != input_hash:
            raise ValueError("Existing output belongs to a different input")
        failures = [c for c in existing.get("cases") or [] if (c.get("judge_scores") or {}).get("judge_failed")]
        if failures and not args.retry_failed:
            raise ValueError("Output contains a failed attempt; review it before --retry-failed")
        for case in failures:
            case.setdefault("judge_failed_attempts", []).append(case["judge_scores"])
            case["judge_scores"] = None
        payload = existing
    payload.setdefault("metadata", {})["judge_input_sha256"] = input_hash
    scored = asyncio.run(score_workbook(payload, args.max_cases, args.output))
    print(f"scored {scored} cases; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
