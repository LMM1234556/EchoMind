"""Evaluate and calibrate a local BGE evidence gate on the reviewed dev set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

from evaluation.evidence_gate_evaluator import (
    annotate_rankings,
    evidence_gate_metrics,
    load_evidence_gate_dataset,
)
from evaluation.retrieval_comparison import (
    LocalCosineRetriever,
    calibrate_threshold,
    collect_rankings,
    load_corpus,
)
from mcp.knowledge_base import BGE_ZH_EMBEDDING_MODEL, KnowledgeBase


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _score_summary(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "min": round(ordered[0], 8),
        "median": round(median, 8),
        "max": round(ordered[-1], 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--fastembed-cache", type=pathlib.Path, required=True)
    parser.add_argument("--existing-threshold", type=float, default=0.46692987)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.top_k < 2:
        parser.error("--top-k must be at least 2")
    if not args.fastembed_cache.is_dir():
        parser.error("--fastembed-cache must be an existing directory")

    os.environ["FASTEMBED_CACHE_PATH"] = str(args.fastembed_cache.resolve())
    os.environ["FASTEMBED_LOCAL_FILES_ONLY"] = "1"
    dataset_path = args.dataset.resolve()
    corpus_path = args.corpus.resolve()
    metadata, cases = load_evidence_gate_dataset(dataset_path)
    _, documents = load_corpus(corpus_path)
    corpus_titles = {item["title"] for item in documents}
    unknown_titles = sorted({case.candidate_title for case in cases} - corpus_titles)
    if unknown_titles:
        raise ValueError(f"candidate titles missing from corpus: {unknown_titles}")

    embedding_function = KnowledgeBase._build_embedding_function(BGE_ZH_EMBEDDING_MODEL)
    retriever = LocalCosineRetriever(documents, embedding_function)
    retriever.search("warmup", top_k=1)
    base_rows = collect_rankings(
        retriever.search,
        [case.as_retrieval_case() for case in cases],
        top_k=args.top_k,
    )
    rows = annotate_rankings(base_rows, cases)
    calibration = calibrate_threshold(rows)
    calibrated_threshold = float(calibration["threshold"])
    supported_scores = [float(row["top_score"]) for row in rows if row["expected_supported"]]
    gap_scores = [float(row["top_score"]) for row in rows if not row["expected_supported"]]
    false_accepts = [
        row["id"] for row in rows
        if not row["expected_supported"] and float(row["top_score"]) >= calibrated_threshold
    ]
    false_rejects = [
        row["id"] for row in rows
        if row["expected_supported"] and float(row["top_score"]) < calibrated_threshold
    ]

    payload = {
        "name": "evidence-gate-dev-evaluation-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "warning": "development-set result; not an independent test result",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_metadata": metadata,
        "corpus": str(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "model": BGE_ZH_EMBEDDING_MODEL,
        "evaluation_engine": "direct_local_embeddings",
        "similarity": "cosine",
        "top_k": args.top_k,
        "existing_threshold_metrics": evidence_gate_metrics(rows, args.existing_threshold),
        "calibration": calibration,
        "calibrated_metrics": evidence_gate_metrics(rows, calibrated_threshold),
        "score_distribution": {
            "supported": _score_summary(supported_scores),
            "knowledge_gap": _score_summary(gap_scores),
        },
        "calibrated_errors": {
            "false_accept_ids": false_accepts,
            "false_reject_ids": false_rejects,
        },
        "cases": rows,
    }
    _write_json(args.report, payload)
    print(
        f"existing_threshold={args.existing_threshold:.8f} "
        f"macro_f1={payload['existing_threshold_metrics']['macro_f1']:.4f} "
        f"rejection={payload['existing_threshold_metrics']['correct_rejection_rate']:.4f}",
        flush=True,
    )
    print(
        f"calibrated_threshold={calibrated_threshold:.8f} "
        f"macro_f1={payload['calibrated_metrics']['macro_f1']:.4f} "
        f"rejection={payload['calibrated_metrics']['correct_rejection_rate']:.4f}",
        flush=True,
    )
    print(f"report={args.report}", flush=True)


if __name__ == "__main__":
    main()
