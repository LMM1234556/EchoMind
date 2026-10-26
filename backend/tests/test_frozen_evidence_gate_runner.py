import hashlib
import json
import pathlib

import pytest

from evaluation.run_frozen_evidence_gate_test import (
    evaluate_rules,
    finalize_first_run_manifest,
    validate_freeze_manifest,
)


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(paths):
    return {
        "first_evaluation_run": False,
        "dataset_sha256": _write(paths[0], "dataset"),
        "corpus_sha256": _write(paths[1], "corpus"),
        "development_dataset_sha256": _write(paths[2], "development dataset"),
        "development_report_sha256": _write(paths[3], "development report"),
        "top_k": 6,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "fixed_rules": {
            "original_score_gate": {"threshold": 0.4},
            "calibrated_score_gate": {"threshold": 0.6},
            "score_and_coverage_gate": {
                "threshold": 0.5,
                "coverage_threshold": 0.3,
                "operator": "and",
            },
        },
    }


def test_manifest_validation_checks_every_frozen_hash(tmp_path):
    paths = [tmp_path / name for name in ("dataset", "corpus", "dev", "report")]
    manifest = _manifest(paths)

    validate_freeze_manifest(manifest, *paths)

    paths[2].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="development_dataset_sha256"):
        validate_freeze_manifest(manifest, *paths)


def test_manifest_validation_refuses_completed_first_run(tmp_path):
    paths = [tmp_path / name for name in ("dataset", "corpus", "dev", "report")]
    manifest = _manifest(paths)
    manifest["first_evaluation_run"] = True

    with pytest.raises(RuntimeError, match="completed first evaluation"):
        validate_freeze_manifest(manifest, *paths)


def test_fixed_rules_are_evaluated_without_recalibration():
    rows = [
        {
            "id": "supported", "expected_supported": True,
            "top_score": 0.7, "lexical_coverage": 0.4,
            "top_title": "A", "candidate_title": "A",
        },
        {
            "id": "gap", "expected_supported": False,
            "top_score": 0.55, "lexical_coverage": 0.2,
            "top_title": "A", "candidate_title": "A",
        },
    ]
    rules = {
        "original_score_gate": {"threshold": 0.4},
        "calibrated_score_gate": {"threshold": 0.6},
        "score_and_coverage_gate": {
            "threshold": 0.5, "coverage_threshold": 0.3, "operator": "and",
        },
    }

    reports = evaluate_rules(rows, rules)

    assert reports["original_score_gate"]["metrics"]["correct_rejection_rate"] == 0.0
    assert reports["calibrated_score_gate"]["metrics"]["accuracy"] == 1.0
    assert reports["score_and_coverage_gate"]["metrics"]["accuracy"] == 1.0


def test_finalize_manifest_records_report_hash(tmp_path):
    manifest_path = tmp_path / "freeze.json"
    report_path = tmp_path / "first-run.json"
    report_path.write_text("report", encoding="utf-8")
    manifest = {"first_evaluation_run": False}

    completed = finalize_first_run_manifest(
        manifest_path, manifest, report_path, "2026-10-26T00:00:00+00:00"
    )

    assert completed["first_evaluation_run"] is True
    assert completed["first_evaluation_report_sha256"] == hashlib.sha256(b"report").hexdigest()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == completed


def test_repository_evidence_gate_freeze_manifest_matches_files():
    backend = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (backend / "evaluation/data/evidence_gate_test_v1.freeze.json").read_text(encoding="utf-8")
    )
    paths = [
        backend.parent / manifest["dataset_path"],
        backend.parent / manifest["corpus_path"],
        backend.parent / manifest["development_dataset_path"],
        backend.parent / manifest["development_report_path"],
    ]

    validate_freeze_manifest(manifest, *paths, allow_completed=True)

    assert manifest["total_cases"] == 42
    assert manifest["supported_cases"] == 24
    assert manifest["knowledge_gap_cases"] == 18
    assert manifest["first_evaluation_run"] is True
    first_report = backend.parent / manifest["first_evaluation_report"]
    assert hashlib.sha256(first_report.read_bytes()).hexdigest() == manifest["first_evaluation_report_sha256"]
    with pytest.raises(RuntimeError, match="completed first evaluation"):
        validate_freeze_manifest(manifest, *paths)
