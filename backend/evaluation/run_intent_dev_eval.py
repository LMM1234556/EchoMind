"""Run a reviewed development intent dataset without touching frozen test data."""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
from datetime import datetime, timezone

from dotenv import load_dotenv

from core.intent_recognizer import IntentRecognizer
from evaluation.intent_dataset import load_dataset
from evaluation.run_frozen_intent_test import _evaluate_one, _write_json, compute_metrics


async def run(args: argparse.Namespace) -> int:
    metadata, all_cases = load_dataset(args.dataset)
    cases = [
        case for case in all_cases
        if case.label_status == "verified" and case.split == "dev"
    ]
    if not cases:
        raise RuntimeError("development evaluation requires verified/dev cases")

    load_dotenv(args.env_file, override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required")

    recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
    completed = []
    print(
        f"dev_eval_start total={len(cases)} excluded={len(all_cases) - len(cases)} "
        f"model={model}",
        flush=True,
    )
    for offset in range(0, len(cases), args.concurrency):
        batch = cases[offset:offset + args.concurrency]
        rows = await asyncio.gather(*(
            _evaluate_one(recognizer, case, args.timeout) for case in batch
        ))
        completed.extend(rows)
        failures = [row for row in rows if row["failed"]]
        print(
            f"progress={len(completed)}/{len(cases)} batch_failures={len(failures)}",
            flush=True,
        )
        if failures:
            _write_json(args.report, {
                "name": metadata.get("name", args.dataset.stem),
                "run_status": "aborted",
                "model": model,
                "provider_failures": len(failures),
                "cases": completed,
            })
            return 2

    metrics = compute_metrics(completed)
    _write_json(args.report, {
        "name": metadata.get("name", args.dataset.stem),
        "run_status": "complete",
        "dataset": str(args.dataset),
        "model": model,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provider_failures": 0,
        "metrics": metrics,
        "cases": completed,
    })
    print(
        f"complete accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env.local"))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
