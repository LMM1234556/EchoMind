import hashlib
import json

import pytest

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
