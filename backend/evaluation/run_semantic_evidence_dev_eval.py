"""Evaluate a constrained Qwen evidence judge on development data only."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from core.llm_utils import extract_text_content, tracked_messages_create
from evaluation.evidence_gate_evaluator import load_evidence_gate_dataset
from evaluation.retrieval_comparison import load_corpus
from evaluation.semantic_evidence_judge import (
    SYSTEM_PROMPT,
    build_batch_items,
    build_user_prompt,
    parse_judge_response,
    semantic_judge_metrics,
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _load_rankings(
    dataset_path: pathlib.Path,
    retrieval_report_path: pathlib.Path,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    _, cases = load_evidence_gate_dataset(dataset_path)
    report = json.loads(retrieval_report_path.read_text(encoding="utf-8"))
    rows = list(report.get("cases") or [])
    case_by_id = {case.id: case for case in cases}
    if set(case_by_id) != {str(row.get("id")) for row in rows}:
        raise ValueError("retrieval report rows do not match development dataset")
    enriched = []
    for row in rows:
        case = case_by_id[str(row["id"])]
        enriched.append({
            **row,
            "expected_supported": case.expected_supported,
            "candidate_title": case.candidate_title,
            "label_basis": case.label_basis,
        })
    return report, enriched


async def run(args: argparse.Namespace) -> None:
    dataset_path = args.dataset.resolve()
    corpus_path = args.corpus.resolve()
    retrieval_report_path = args.retrieval_report.resolve()
    report_path = args.report.resolve()
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else report_path.with_suffix(report_path.suffix + ".checkpoint")
    )
    if report_path.exists() and not args.dry_run:
        raise RuntimeError("semantic evidence report already exists")

    _, rows = _load_rankings(dataset_path, retrieval_report_path)
    _, documents = load_corpus(corpus_path)
    documents_by_title = {item["title"]: item["content"] for item in documents}
    batches = [rows[index:index + args.batch_size] for index in range(0, len(rows), args.batch_size)]
    prompts = [build_user_prompt(build_batch_items(batch, documents_by_title)) for batch in batches]
    estimated_characters = sum(len(SYSTEM_PROMPT) + len(prompt) for prompt in prompts)

    if args.dry_run:
        print(f"cases={len(rows)}", flush=True)
        print(f"batch_size={args.batch_size}", flush=True)
        print(f"api_calls={len(batches)}", flush=True)
        print(f"prompt_characters={estimated_characters}", flush=True)
        print("labels_sent_to_model=false", flush=True)
        return

    load_dotenv(args.env_file.resolve(), override=True)
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    if not api_key or not model:
        raise RuntimeError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required")

    checkpoint: Dict[str, Any] = {
        "dataset_sha256": sha256(dataset_path),
        "retrieval_report_sha256": sha256(retrieval_report_path),
        "model": model,
        "decisions": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "completed_batches": [],
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("dataset_sha256") != sha256(dataset_path):
            raise RuntimeError("checkpoint dataset hash mismatch")
        if checkpoint.get("retrieval_report_sha256") != sha256(retrieval_report_path):
            raise RuntimeError("checkpoint retrieval report hash mismatch")
        if checkpoint.get("model") != model:
            raise RuntimeError("checkpoint model mismatch")

    client_options: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    client = AsyncAnthropic(**client_options)

    for batch_index, (batch, prompt) in enumerate(zip(batches, prompts), start=1):
        if batch_index in checkpoint["completed_batches"]:
            continue
        response = await tracked_messages_create(
            client,
            "evaluation.evidence_judge",
            model=model,
            max_tokens=args.max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = extract_text_content(response.content)
        expected_ids = [str(row["id"]) for row in batch]
        decisions = parse_judge_response(text, expected_ids)
        for decision in decisions:
            checkpoint["decisions"][decision["id"]] = decision
        usage = _usage(response)
        checkpoint["usage"]["input_tokens"] += usage["input_tokens"]
        checkpoint["usage"]["output_tokens"] += usage["output_tokens"]
        checkpoint["completed_batches"].append(batch_index)
        write_json(checkpoint_path, checkpoint)
        print(
            f"batch={batch_index}/{len(batches)} cases={len(batch)} "
            f"input_tokens={usage['input_tokens']} output_tokens={usage['output_tokens']}",
            flush=True,
        )

    decisions = checkpoint["decisions"]
    metrics = semantic_judge_metrics(rows, decisions)
    case_results = []
    for row in rows:
        decision = decisions[str(row["id"])]
        case_results.append({
            "id": row["id"],
            "query": row["query"],
            "expected_supported": row["expected_supported"],
            "predicted_supported": decision["supported"],
            "evidence": decision["evidence"],
            "candidate_title": row["candidate_title"],
            "top_title": row["top_title"],
            "top_score": row["top_score"],
        })
    payload = {
        "name": "semantic-evidence-judge-dev-evaluation-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "warning": "development-set result; not an independent test result",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "retrieval_report": str(retrieval_report_path),
        "retrieval_report_sha256": sha256(retrieval_report_path),
        "model": model,
        "base_url_configured": bool(base_url),
        "batch_size": args.batch_size,
        "api_calls": len(batches),
        "usage": checkpoint["usage"],
        "metrics": metrics,
        "cases": case_results,
    }
    write_json(report_path, payload)
    checkpoint_path.unlink(missing_ok=True)
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"strict_accuracy={metrics['strict_decision_accuracy']:.4f} "
        f"rejection={metrics['correct_rejection_rate']:.4f}",
        flush=True,
    )
    print(f"report={report_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--retrieval-report", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env.local"))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_tokens < 100:
        parser.error("--max-tokens must be at least 100")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
