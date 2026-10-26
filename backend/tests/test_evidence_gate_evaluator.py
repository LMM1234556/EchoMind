import json

import pytest

from evaluation.evidence_gate_evaluator import (
    add_lexical_coverage,
    annotate_rankings,
    calibrate_score_coverage_gate,
    evidence_gate_metrics,
    load_evidence_gate_dataset,
)


def _write_dataset(path, *, reviewed=2, scoreable=2):
    path.write_text(json.dumps({
        "metadata": {
            "total_cases": 2,
            "supported_cases": 1,
            "knowledge_gap_cases": 1,
            "reviewed_cases": reviewed,
            "scoreable_total": scoreable,
        },
        "cases": [
            {
                "id": "supported",
                "query": "answerable",
                "candidate_title": "A",
                "expected_supported": True,
                "label_basis": "document says so",
                "review": {"status": "verified", "reviewer": "owner"},
            },
            {
                "id": "gap",
                "query": "unsupported detail",
                "candidate_title": "A",
                "expected_supported": False,
                "label_basis": "document omits detail",
                "review": {"status": "verified", "reviewer": "owner"},
            },
        ],
    }), encoding="utf-8")


def test_load_evidence_gate_dataset_requires_reviewed_scoreable_cases(tmp_path):
    path = tmp_path / "gate.json"
    _write_dataset(path)

    metadata, cases = load_evidence_gate_dataset(path)

    assert metadata["scoreable_total"] == 2
    assert cases[0].as_retrieval_case().expected_title == "A"
    assert cases[1].as_retrieval_case().expected_title is None


@pytest.mark.parametrize("reviewed,scoreable", [(1, 2), (2, 1)])
def test_load_evidence_gate_dataset_rejects_incomplete_dataset(tmp_path, reviewed, scoreable):
    path = tmp_path / "gate.json"
    _write_dataset(path, reviewed=reviewed, scoreable=scoreable)

    with pytest.raises(ValueError):
        load_evidence_gate_dataset(path)


def test_annotate_rankings_adds_candidate_and_margin(tmp_path):
    path = tmp_path / "gate.json"
    _write_dataset(path)
    _, cases = load_evidence_gate_dataset(path)
    rows = [
        {
            "id": "supported", "top_titles": ["A", "B"],
            "top_scores": [0.9, 0.5], "top_score": 0.9,
        },
        {
            "id": "gap", "top_titles": ["B", "A"],
            "top_scores": [0.8, 0.7], "top_score": 0.8,
        },
    ]

    annotated = annotate_rankings(rows, cases)

    assert annotated[0]["score_margin"] == 0.4
    assert annotated[1]["candidate_rank"] == 2
    assert annotated[1]["candidate_score"] == 0.7


def test_evidence_gate_metrics_reports_false_accepts_and_strict_title_errors():
    rows = [
        {
            "expected_supported": True, "top_score": 0.9,
            "top_title": "wrong", "candidate_title": "A",
        },
        {
            "expected_supported": False, "top_score": 0.8,
            "top_title": "A", "candidate_title": "A",
        },
        {
            "expected_supported": False, "top_score": 0.2,
            "top_title": "B", "candidate_title": "A",
        },
    ]

    metrics = evidence_gate_metrics(rows, threshold=0.5)

    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 0, 1)
    assert metrics["accuracy"] == 0.6667
    assert metrics["strict_decision_accuracy"] == 0.3333
    assert metrics["correct_rejection_rate"] == 0.5


def test_add_lexical_coverage_measures_query_units_found_in_top_document():
    rows = [{"query": "退款到账时间", "top_title": "退款政策"}]
    documents = [{"title": "退款政策", "content": "退款将在三个工作日内到账。"}]

    enriched = add_lexical_coverage(rows, documents)

    assert enriched[0]["query_unit_count"] > 0
    assert enriched[0]["matched_query_unit_count"] > 0
    assert 0 < enriched[0]["lexical_coverage"] <= 1


def test_calibrate_score_coverage_gate_uses_both_features():
    rows = [
        {
            "expected_supported": True, "top_score": 0.9, "lexical_coverage": 0.8,
            "top_title": "A", "candidate_title": "A",
        },
        {
            "expected_supported": True, "top_score": 0.7, "lexical_coverage": 0.6,
            "top_title": "A", "candidate_title": "A",
        },
        {
            "expected_supported": False, "top_score": 0.8, "lexical_coverage": 0.2,
            "top_title": "A", "candidate_title": "A",
        },
        {
            "expected_supported": False, "top_score": 0.3, "lexical_coverage": 0.7,
            "top_title": "B", "candidate_title": "A",
        },
    ]

    calibrated = calibrate_score_coverage_gate(rows)

    assert calibrated["macro_f1"] == 1.0
    assert calibrated["correct_rejection_rate"] == 1.0
    assert calibrated["rule"].startswith("top_score >=")
