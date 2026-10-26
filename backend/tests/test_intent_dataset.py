import hashlib
import json
import pathlib

import pytest

from evaluation.intent_dataset import (
    DatasetValidationError,
    audit_dataset,
    load_dataset,
    normalize_message,
    scoreable_intent_cases,
)


def _write_dataset(tmp_path, cases):
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps({"version": "test", "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _case(**overrides):
    case = {
        "id": "case-1",
        "message": "一个新的测试问题",
        "expected_intent": "query",
        "split": "candidate",
        "difficulty": "medium",
        "source": "human_written",
        "label_status": "draft",
    }
    case.update(overrides)
    return case


def test_candidate_dataset_is_audited_but_not_scoreable(tmp_path):
    path = _write_dataset(tmp_path, [_case()])

    report = audit_dataset(path)

    assert report["total"] == 1
    assert report["status_counts"] == {"draft": 1}
    assert report["scoreable_total"] == 0
    with pytest.raises(DatasetValidationError, match="no human-verified"):
        scoreable_intent_cases(path)


def test_verified_test_case_requires_reviewer(tmp_path):
    path = _write_dataset(tmp_path, [
        _case(label_status="verified", split="test"),
    ])

    with pytest.raises(DatasetValidationError, match="require a reviewer"):
        load_dataset(path)


def test_verified_test_case_can_be_converted_for_evaluation(tmp_path):
    path = _write_dataset(tmp_path, [
        _case(label_status="verified", split="test", reviewer="human-1"),
    ])

    cases = scoreable_intent_cases(path)

    assert len(cases) == 1
    assert cases[0].message == "一个新的测试问题"
    assert cases[0].expected_intent == "query"


def test_normalization_detects_punctuation_only_variants(tmp_path):
    path = _write_dataset(tmp_path, [
        _case(id="a", message="退款，多久到账？", expected_intent="refund"),
        _case(id="b", message="退款多久到账", expected_intent="refund"),
    ])

    assert normalize_message("退款，多久到账？") == "退款多久到账"
    with pytest.raises(DatasetValidationError, match="duplicate message"):
        load_dataset(path)


def test_audit_reports_cross_dataset_overlap(tmp_path):
    current = _write_dataset(tmp_path, [
        _case(id="current", message="退款，多久到账？", expected_intent="refund"),
    ])
    comparison_dir = tmp_path / "comparison"
    comparison_dir.mkdir()
    comparison = comparison_dir / "dev.json"
    comparison.write_text(
        json.dumps({
            "name": "reviewed-dev",
            "version": "test",
            "cases": [
                _case(id="dev-1", message="退款多久到账", expected_intent="refund"),
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = audit_dataset(current, compare_paths=[comparison])

    assert report["cross_dataset_overlaps"] == [
        {"id": "current", "matches": ["reviewed-dev:dev-1"]}
    ]
    assert any("comparison datasets" in warning for warning in report["warnings"])


def test_audit_reports_lexical_near_duplicate_without_calling_it_exact(tmp_path):
    current = _write_dataset(tmp_path, [
        _case(id="current", message="退款大概多久能到账", expected_intent="refund"),
    ])
    comparison_dir = tmp_path / "comparison-near"
    comparison_dir.mkdir()
    comparison = comparison_dir / "dev.json"
    comparison.write_text(
        json.dumps({
            "name": "reviewed-dev",
            "version": "test",
            "cases": [
                _case(id="dev-1", message="退款多久到账", expected_intent="refund"),
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    report = audit_dataset(
        current,
        compare_paths=[comparison],
        near_duplicate_threshold=0.8,
    )

    assert report["cross_dataset_overlaps"] == []
    assert report["lexical_near_overlaps"] == [{
        "id": "current",
        "matches": [{"source": "reviewed-dev:dev-1", "similarity": 0.8}],
    }]


def test_frozen_dataset_rejects_non_test_cases(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(
        json.dumps({
            "name": "frozen-test",
            "version": "1.0",
            "frozen": True,
            "baseline_commit": "abc123",
            "cases": [
                _case(
                    label_status="verified",
                    reviewer="human-1",
                    split="candidate",
                ),
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="verified/test"):
        load_dataset(path)


def test_frozen_intent_v1_matches_freeze_manifest():
    backend_root = pathlib.Path(__file__).resolve().parents[1]
    dataset_path = backend_root / "evaluation" / "data" / "intent_test_v1.json"
    manifest_path = (
        backend_root / "evaluation" / "data" / "intent_test_v1.freeze.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    assert dataset_hash == manifest["dataset_sha256"]
    metadata, cases = load_dataset(dataset_path)
    assert metadata["frozen"] is True
    assert metadata["baseline_commit"] == manifest["baseline_commit"]
    assert len(cases) == manifest["total_cases"] == 95
    if manifest.get("first_evaluation_run"):
        report_path = backend_root.parent / manifest["first_evaluation_report"]
        report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
        assert report_hash == manifest["first_evaluation_report_sha256"]
