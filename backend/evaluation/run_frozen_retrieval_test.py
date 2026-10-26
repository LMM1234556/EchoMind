"""Run the paired frozen retrieval benchmark exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv

from evaluation.retrieval_comparison import (
    LocalCosineRetriever,
    collect_rankings,
    load_corpus,
    load_test_dataset,
    summarize_retrieval,
)
from mcp.knowledge_base import KnowledgeBase


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def validate_freeze_manifest(
    manifest: Dict[str, Any],
    dataset_path: pathlib.Path,
    corpus_path: pathlib.Path,
    calibration_path: pathlib.Path,
) -> None:
    if manifest.get("first_evaluation_run") is True:
        raise RuntimeError("freeze manifest already records a completed first evaluation")
    expected = {
        "dataset_sha256": sha256(dataset_path),
        "corpus_sha256": sha256(corpus_path),
        "calibration_report_sha256": sha256(calibration_path),
    }
    for field, actual in expected.items():
        if manifest.get(field) != actual:
            raise RuntimeError(f"{field} does not match freeze manifest")
    if manifest.get("top_k", 0) < 3:
        raise RuntimeError("freeze manifest top_k must be at least 3")
    models = manifest.get("models") or {}
    if set(models) != {"baseline_minilm", "candidate_bge"}:
        raise RuntimeError("freeze manifest must define the two fixed models")
    for name, config in models.items():
        if not config.get("embedding_model") or not isinstance(config.get("threshold"), (int, float)):
            raise RuntimeError(f"invalid fixed model config: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--calibration-report", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env.local"))
    parser.add_argument("--minilm-cache", type=pathlib.Path, required=True)
    parser.add_argument("--fastembed-cache", type=pathlib.Path, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    manifest_path = args.manifest.resolve()
    corpus_path = args.corpus.resolve()
    calibration_path = args.calibration_report.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_freeze_manifest(manifest, dataset_path, corpus_path, calibration_path)
    metadata, cases = load_test_dataset(dataset_path, require_frozen=True)
    _, corpus_documents = load_corpus(corpus_path)
    if metadata.get("baseline_commit") != manifest.get("evaluation_code_commit"):
        raise RuntimeError("dataset baseline_commit does not match evaluation code commit")

    load_dotenv(args.env_file, override=True)
    os.environ["CHROMA_EMBEDDING_CACHE"] = str(args.minilm_cache.resolve())
    os.environ["FASTEMBED_CACHE_PATH"] = str(args.fastembed_cache.resolve())

    model_reports: Dict[str, Any] = {}
    for name in ("baseline_minilm", "candidate_bge"):
        config = manifest["models"][name]
        embedding_function = KnowledgeBase._build_embedding_function(config["embedding_model"])
        retriever = LocalCosineRetriever(corpus_documents, embedding_function)
        retriever.search("warmup", top_k=1)
        rows = collect_rankings(retriever.search, cases, top_k=int(manifest["top_k"]))
        model_reports[name] = {
            "embedding_model": config["embedding_model"],
            "threshold": config["threshold"],
            "evaluation_engine": "direct_local_embeddings",
            "similarity": "cosine",
            "metrics": summarize_retrieval(rows, float(config["threshold"])),
            "cases": rows,
        }
        print(
            f"evaluated {name} total={len(rows)} "
            f"decision_accuracy={model_reports[name]['metrics']['decision_accuracy']:.4f}",
            flush=True,
        )

    report = {
        "name": "retrieval-test-v1-first-run",
        "run_status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "corpus": str(corpus_path),
        "corpus_sha256": sha256(corpus_path),
        "calibration_report": str(calibration_path),
        "calibration_report_sha256": sha256(calibration_path),
        "evaluation_code_commit": manifest["evaluation_code_commit"],
        "top_k": manifest["top_k"],
        "models": model_reports,
    }
    write_json(args.report, report)
    print(f"report={args.report}", flush=True)


if __name__ == "__main__":
    main()
