"""为校准工作表生成 LLM-as-Judge 分数。

该脚本会产生真实模型调用。只有显式传入 ``--execute`` 时才运行。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from evaluation.evaluator import LLMJudge


def _context_text(case: Dict[str, Any]) -> str:
    return json.dumps(case.get("reference_context") or {}, ensure_ascii=False, sort_keys=True)


async def score_workbook(payload: Dict[str, Any], max_cases: int) -> int:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    if not api_key or not model:
        raise RuntimeError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required")
    options: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        options["base_url"] = base_url
    judge = LLMJudge(AsyncAnthropic(**options), model)

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
        scored += 1
    payload.setdefault("metadata", {})["judge_model"] = model
    payload["metadata"]["judge_scored_at"] = datetime.now(timezone.utc).isoformat()
    return scored


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run real LLM judge calls for calibration")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.execute:
        raise RuntimeError("real model calls require the explicit --execute flag")
    if args.max_cases < 1:
        raise ValueError("--max-cases must be positive")

    root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env.local", override=True)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    scored = asyncio.run(score_workbook(payload, args.max_cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scored {scored} cases; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
