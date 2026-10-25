import json

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
