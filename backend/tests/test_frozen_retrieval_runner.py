import hashlib
import json
import pathlib

import pytest

from evaluation.retrieval_comparison import load_test_dataset
from evaluation.run_frozen_retrieval_test import validate_freeze_manifest


def _write(path, content):
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_validation_checks_all_frozen_hashes(tmp_path):
    dataset = tmp_path / "dataset.json"
    corpus = tmp_path / "corpus.json"
    calibration = tmp_path / "calibration.json"
    manifest = {
        "first_evaluation_run": False,
        "dataset_sha256": _write(dataset, "dataset"),
        "corpus_sha256": _write(corpus, "corpus"),
        "calibration_report_sha256": _write(calibration, "calibration"),
        "top_k": 6,
        "models": {
            "baseline_minilm": {"embedding_model": "mini", "threshold": 0.1},
            "candidate_bge": {"embedding_model": "bge", "threshold": 0.2},
        },
    }

    validate_freeze_manifest(manifest, dataset, corpus, calibration)


def test_manifest_validation_refuses_completed_first_run(tmp_path):
    paths = [tmp_path / name for name in ("dataset", "corpus", "calibration")]
    hashes = [_write(path, path.name) for path in paths]
    manifest = {
        "first_evaluation_run": True,
        "dataset_sha256": hashes[0],
        "corpus_sha256": hashes[1],
        "calibration_report_sha256": hashes[2],
        "top_k": 6,
        "models": {
            "baseline_minilm": {"embedding_model": "mini", "threshold": 0.1},
            "candidate_bge": {"embedding_model": "bge", "threshold": 0.2},
        },
    }

    with pytest.raises(RuntimeError, match="completed first evaluation"):
        validate_freeze_manifest(manifest, *paths)


def test_repository_retrieval_freeze_manifest_matches_files():
    backend = pathlib.Path(__file__).resolve().parents[1]
    dataset = backend / "evaluation/data/retrieval_test_v1.json"
    corpus = backend / "evaluation/data/retrieval_corpus_v1.json"
    calibration = backend / "evaluation/reports/retrieval_threshold_calibration_v1_20261026.json"
    manifest = json.loads(
        (backend / "evaluation/data/retrieval_test_v1.freeze.json").read_text(encoding="utf-8")
    )

    validate_freeze_manifest(
        manifest,
        dataset,
        corpus,
        calibration,
        allow_completed=True,
    )
    metadata, cases = load_test_dataset(dataset)

    assert metadata["baseline_commit"] == manifest["evaluation_code_commit"]
    assert len(cases) == manifest["total_cases"] == 60
    assert sum(case.expected_title is not None for case in cases) == 36
    assert manifest["first_evaluation_run"] is True
    first_report = backend.parent / manifest["first_evaluation_report"]
    assert hashlib.sha256(first_report.read_bytes()).hexdigest() == manifest["first_evaluation_report_sha256"]
