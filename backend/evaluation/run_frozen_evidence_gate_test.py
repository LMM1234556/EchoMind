"""Run the fixed evidence-gate benchmark exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from evaluation.evidence_gate_evaluator import (
    add_lexical_coverage,
    annotate_rankings,
    evidence_gate_metrics,
    load_evidence_gate_dataset,
)
from evaluation.retrieval_comparison import LocalCosineRetriever, collect_rankings, load_corpus
from mcp.knowledge_base import BGE_ZH_EMBEDDING_MODEL, KnowledgeBase


REQUIRED_RULES = {
    "original_score_gate",
    "calibrated_score_gate",
    "score_and_coverage_gate",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def validate_freeze_manifest(
    manifest: Mapping[str, Any],
    dataset_path: pathlib.Path,
    corpus_path: pathlib.Path,
    development_dataset_path: pathlib.Path,
    development_report_path: pathlib.Path,
    *,
    allow_completed: bool = False,
) -> None:
    if manifest.get("first_evaluation_run") is True and not allow_completed:
        raise RuntimeError("freeze manifest already records a completed first evaluation")
    expected_hashes = {
        "dataset_sha256": sha256(dataset_path),
        "corpus_sha256": sha256(corpus_path),
        "development_dataset_sha256": sha256(development_dataset_path),
        "development_report_sha256": sha256(development_report_path),
    }
    for field, actual in expected_hashes.items():
        if manifest.get(field) != actual:
            raise RuntimeError(f"{field} does not match freeze manifest")
    if int(manifest.get("top_k", 0)) < 2:
        raise RuntimeError("freeze manifest top_k must be at least 2")
    if manifest.get("embedding_model") != BGE_ZH_EMBEDDING_MODEL:
        raise RuntimeError("freeze manifest embedding model is not the fixed BGE model")
    rules = manifest.get("fixed_rules") or {}
    if set(rules) != REQUIRED_RULES:
        raise RuntimeError("freeze manifest does not define the three fixed gate rules")
    for name in ("original_score_gate", "calibrated_score_gate"):
        if not isinstance((rules.get(name) or {}).get("threshold"), (int, float)):
            raise RuntimeError(f"invalid score threshold for {name}")
    combined = rules.get("score_and_coverage_gate") or {}
    if not isinstance(combined.get("threshold"), (int, float)):
        raise RuntimeError("invalid combined score threshold")
    if not isinstance(combined.get("coverage_threshold"), (int, float)):
        raise RuntimeError("invalid combined coverage threshold")
    if str(combined.get("operator", "")).lower() != "and":
        raise RuntimeError("combined gate operator must be AND")


def evaluate_rules(
    rows: Sequence[Dict[str, Any]],
    fixed_rules: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    for name in ("original_score_gate", "calibrated_score_gate", "score_and_coverage_gate"):
        rule = fixed_rules[name]
        threshold = float(rule["threshold"])
        coverage_threshold = (
            float(rule["coverage_threshold"])
            if name == "score_and_coverage_gate"
            else None
        )
        metrics = evidence_gate_metrics(rows, threshold, coverage_threshold)
        errors = []
        for row in rows:
            accepted = float(row["top_score"]) >= threshold
            if coverage_threshold is not None:
                accepted = accepted and float(row["lexical_coverage"]) >= coverage_threshold
            expected = bool(row["expected_supported"])
            strict_correct = (
                accepted and expected and row.get("top_title") == row.get("candidate_title")
            ) or (not accepted and not expected)
            if not strict_correct:
                errors.append({
                    "id": row["id"],
                    "expected_supported": expected,
                    "predicted_supported": accepted,
                    "candidate_title": row.get("candidate_title"),
                    "top_title": row.get("top_title"),
                    "top_score": row.get("top_score"),
                    "lexical_coverage": row.get("lexical_coverage"),
                })
        reports[name] = {
            "rule": dict(rule),
            "metrics": metrics,
            "strict_errors": errors,
        }
    return reports


def repository_relative(path: pathlib.Path) -> str:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            return resolved.relative_to(parent).as_posix()
    return str(resolved)


def finalize_first_run_manifest(
    manifest_path: pathlib.Path,
    manifest: Mapping[str, Any],
    report_path: pathlib.Path,
    completed_at: str,
) -> Dict[str, Any]:
    completed = {
        **manifest,
        "first_evaluation_run": True,
        "first_evaluation_completed_at": completed_at,
        "first_evaluation_report": repository_relative(report_path),
        "first_evaluation_report_sha256": sha256(report_path),
    }
    write_json(manifest_path, completed)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--development-dataset", type=pathlib.Path, required=True)
    parser.add_argument("--development-report", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--fastembed-cache", type=pathlib.Path, required=True)
    args = parser.parse_args()

    paths = {
        "dataset": args.dataset.resolve(),
        "manifest": args.manifest.resolve(),
        "corpus": args.corpus.resolve(),
        "development_dataset": args.development_dataset.resolve(),
        "development_report": args.development_report.resolve(),
        "report": args.report.resolve(),
    }
    if paths["report"].exists():
        raise RuntimeError("refusing to run because the first-run report path already exists")
    if not args.fastembed_cache.is_dir():
        parser.error("--fastembed-cache must be an existing directory")

    lock_path = paths["manifest"].with_suffix(paths["manifest"].suffix + ".run.lock")
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        validate_freeze_manifest(
            manifest,
            paths["dataset"],
            paths["corpus"],
            paths["development_dataset"],
            paths["development_report"],
        )
        metadata, cases = load_evidence_gate_dataset(paths["dataset"])
        if metadata.get("frozen") is not True or metadata.get("evaluation_status") != "frozen_not_run":
            raise RuntimeError("evidence gate dataset is not frozen for its first run")
        if len(cases) != int(manifest["total_cases"]):
            raise RuntimeError("dataset case count does not match freeze manifest")
        _, documents = load_corpus(paths["corpus"])
        corpus_titles = {item["title"] for item in documents}
        if {case.candidate_title for case in cases} - corpus_titles:
            raise RuntimeError("evidence gate candidate title is missing from corpus")

        os.environ["FASTEMBED_CACHE_PATH"] = str(args.fastembed_cache.resolve())
        os.environ["FASTEMBED_LOCAL_FILES_ONLY"] = "1"
        embedding_function = KnowledgeBase._build_embedding_function(BGE_ZH_EMBEDDING_MODEL)
        retriever = LocalCosineRetriever(documents, embedding_function)
        retriever.search("warmup", top_k=1)
        base_rows = collect_rankings(
            retriever.search,
            [case.as_retrieval_case() for case in cases],
            top_k=int(manifest["top_k"]),
        )
        rows = add_lexical_coverage(annotate_rankings(base_rows, cases), documents)
        rule_reports = evaluate_rules(rows, manifest["fixed_rules"])
        completed_at = datetime.now(timezone.utc).isoformat()
        report = {
            "name": "evidence-gate-test-v1-first-run",
            "run_status": "complete",
            "completed_at": completed_at,
            "warning": "first and only untouched frozen-test result",
            "dataset": repository_relative(paths["dataset"]),
            "dataset_sha256": sha256(paths["dataset"]),
            "corpus": repository_relative(paths["corpus"]),
            "corpus_sha256": sha256(paths["corpus"]),
            "gate_rules_commit": manifest["gate_rules_commit"],
            "evaluation_code_commit": manifest.get("evaluation_code_commit"),
            "embedding_model": manifest["embedding_model"],
            "top_k": manifest["top_k"],
            "rules": rule_reports,
            "cases": rows,
        }
        write_json(paths["report"], report)
        finalize_first_run_manifest(
            paths["manifest"], manifest, paths["report"], completed_at
        )
        for name, result in rule_reports.items():
            metrics = result["metrics"]
            print(
                f"{name} strict_accuracy={metrics['strict_decision_accuracy']:.4f} "
                f"rejection={metrics['correct_rejection_rate']:.4f}",
                flush=True,
            )
        print(f"report={paths['report']}", flush=True)
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    main()
