"""Calibrate MiniLM and BGE rejection thresholds on existing development cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv

from evaluation.retrieval_comparison import (
    LabelledRetrievalCase,
    LocalCosineRetriever,
    calibrate_threshold,
    collect_rankings,
    load_corpus,
    summarize_retrieval,
)
from evaluation.retrieval_evaluator import (
    CHALLENGE_RETRIEVAL_CASES,
    NEGATIVE_RETRIEVAL_CASES,
    SEED_RETRIEVAL_CASES,
)
from mcp.knowledge_base import (
    BGE_ZH_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    KnowledgeBase,
)


MODEL_CONFIGS = {
    "baseline_minilm": {
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "collection": "retrieval_eval_minilm_v1",
    },
    "candidate_bge": {
        "embedding_model": BGE_ZH_EMBEDDING_MODEL,
        "collection": "retrieval_eval_bge_v1",
    },
}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _development_cases() -> List[LabelledRetrievalCase]:
    source = [
        ("seed", SEED_RETRIEVAL_CASES),
        ("challenge", CHALLENGE_RETRIEVAL_CASES),
        ("negative", NEGATIVE_RETRIEVAL_CASES),
    ]
    rows: List[LabelledRetrievalCase] = []
    for group, cases in source:
        for index, case in enumerate(cases, start=1):
            rows.append(LabelledRetrievalCase(
                id=f"dev-{group}-{index:03d}",
                query=case.query,
                expected_title=case.expected_title,
                case_type="positive" if case.expected_title is not None else "out_of_domain",
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env.local"))
    parser.add_argument("--minilm-cache", type=pathlib.Path, required=True)
    parser.add_argument("--fastembed-cache", type=pathlib.Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.top_k < 3:
        parser.error("--top-k must be at least 3")

    load_dotenv(args.env_file, override=True)
    os.environ["CHROMA_EMBEDDING_CACHE"] = str(args.minilm_cache.resolve())
    os.environ["FASTEMBED_CACHE_PATH"] = str(args.fastembed_cache.resolve())
    corpus_path = args.corpus.resolve()
    _, corpus_documents = load_corpus(corpus_path)
    cases = _development_cases()
    reports: Dict[str, Any] = {}

    for name, config in MODEL_CONFIGS.items():
        embedding_function = KnowledgeBase._build_embedding_function(
            config["embedding_model"]
        )
        retriever = LocalCosineRetriever(corpus_documents, embedding_function)
        retriever.search("warmup", top_k=1)
        rows = collect_rankings(retriever.search, cases, top_k=args.top_k)
        calibration = calibrate_threshold(rows)
        reports[name] = {
            **config,
            "evaluation_engine": "direct_local_embeddings",
            "similarity": "cosine",
            "calibration": calibration,
            "development_metrics": summarize_retrieval(rows, calibration["threshold"]),
            "cases": rows,
        }
        print(
            f"calibrated {name} threshold={calibration['threshold']:.8f} "
            f"macro_f1={calibration['macro_f1']:.4f}",
            flush=True,
        )

    payload = {
        "name": "retrieval-threshold-calibration-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "development_scope": {
            "total": len(cases),
            "positive": sum(case.expected_title is not None for case in cases),
            "negative": sum(case.expected_title is None for case in cases),
            "warning": "development cases only; not an independent test result",
        },
        "top_k": args.top_k,
        "models": reports,
    }
    _write_json(args.report, payload)
    print(f"report={args.report}", flush=True)


if __name__ == "__main__":
    main()
