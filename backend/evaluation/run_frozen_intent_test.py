"""Run a frozen intent test once with hash verification and resumable checkpoints."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

from core.intent_recognizer import IntentRecognizer
from evaluation.intent_dataset import load_dataset


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    if any(row.get("failed") for row in rows):
        raise ValueError("failed provider calls must not be scored")
    labels = sorted({str(row["expected"]) for row in rows})
    correct = sum(row["predicted"] == row["expected"] for row in rows)
    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(r["predicted"] == label and r["expected"] == label for r in rows)
        fp = sum(r["predicted"] == label and r["expected"] != label for r in rows)
        fn = sum(r["predicted"] != label and r["expected"] == label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(r["expected"] == label for r in rows),
        }
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "macro_f1": round(macro_f1, 4),
        "predicted_counts": dict(sorted(Counter(r["predicted"] for r in rows).items())),
        "per_class": per_class,
    }


async def _evaluate_one(
    recognizer: IntentRecognizer,
    case,
    timeout_s: float,
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(recognizer.recognize(case.message), timeout=timeout_s)
        source_scores = dict(result.source_scores)
        llm_failed = bool(source_scores.get("llm_failed", 0.0))
        return {
            "id": case.id,
            "message": case.message,
            "expected": case.expected_intent,
            "predicted": result.intent.value,
            "confidence": round(float(result.confidence), 4),
            "reasoning": result.reasoning,
            "source_scores": source_scores,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "failed": llm_failed,
            "error": "LLM provider call failed; fallback result is not scoreable" if llm_failed else None,
        }
    except asyncio.TimeoutError:
        return {
            "id": case.id,
            "message": case.message,
            "expected": case.expected_intent,
            "predicted": None,
            "confidence": 0.0,
            "reasoning": "",
            "source_scores": {},
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "failed": True,
            "error": f"request exceeded {timeout_s:.1f}s",
        }
    except Exception as ex:
        return {
            "id": case.id,
            "message": case.message,
            "expected": case.expected_intent,
            "predicted": None,
            "confidence": 0.0,
            "reasoning": "",
            "source_scores": {},
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "failed": True,
            "error": f"{type(ex).__name__}: {ex}",
        }


async def run(args: argparse.Namespace) -> int:
    dataset_path = args.dataset.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = _sha256(dataset_path)
    if dataset_hash != manifest["dataset_sha256"]:
        raise RuntimeError("dataset SHA-256 does not match freeze manifest")

    metadata, cases = load_dataset(dataset_path)
    if metadata.get("frozen") is not True:
        raise RuntimeError("dataset is not frozen")

    load_dotenv(args.env_file, override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required")

    checkpoint: Dict[str, Any] = {
        "dataset_sha256": dataset_hash,
        "baseline_commit": metadata["baseline_commit"],
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "successful_cases": [],
        "last_failures": [],
    }
    if args.checkpoint.exists():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("dataset_sha256") != dataset_hash:
            raise RuntimeError("checkpoint belongs to a different dataset")
        if checkpoint.get("model") != model:
            raise RuntimeError("checkpoint belongs to a different model")

    successful = {row["id"]: row for row in checkpoint.get("successful_cases", [])}
    pending = [case for case in cases if case.id not in successful]
    recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
    print(
        f"frozen_test_start total={len(cases)} completed={len(successful)} "
        f"pending={len(pending)} model={model}",
        flush=True,
    )

    for offset in range(0, len(pending), args.concurrency):
        batch = pending[offset:offset + args.concurrency]
        batch_results = await asyncio.gather(*(
            _evaluate_one(recognizer, case, args.timeout) for case in batch
        ))
        failures = [row for row in batch_results if row["failed"]]
        for row in batch_results:
            if not row["failed"]:
                successful[row["id"]] = row
        checkpoint["successful_cases"] = [
            successful[case.id] for case in cases if case.id in successful
        ]
        checkpoint["last_failures"] = failures
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(args.checkpoint, checkpoint)
        print(
            f"progress={len(successful)}/{len(cases)} "
            f"batch_failures={len(failures)}",
            flush=True,
        )
        if failures:
            print("stopped_on_provider_failure=true", flush=True)
            return 2

    ordered = [successful[case.id] for case in cases]
    report = {
        "name": "intent-test-v1-first-run",
        "run_status": "complete",
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "baseline_commit": metadata["baseline_commit"],
        "model": model,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provider_failures": 0,
        "metrics": compute_metrics(ordered),
        "cases": ordered,
    }
    _write_json(args.report, report)
    print(
        f"complete accuracy={report['metrics']['accuracy']:.4f} "
        f"macro_f1={report['metrics']['macro_f1']:.4f}",
        flush=True,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
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
