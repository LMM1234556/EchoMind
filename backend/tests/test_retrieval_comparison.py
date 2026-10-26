import json

import numpy as np
import pytest

from evaluation.retrieval_comparison import (
    LabelledRetrievalCase,
    LocalCosineRetriever,
    answerability_metrics,
    calibrate_threshold,
    collect_rankings,
    load_corpus,
    load_test_dataset,
    summarize_retrieval,
)


def test_local_cosine_retriever_ranks_by_normalized_similarity():
    vectors = {
        "doc a": [1.0, 0.0],
        "doc b": [0.0, 2.0],
        "query": [0.8, 0.2],
    }

    def embed(texts):
        return [np.asarray(vectors[text], dtype=np.float32) for text in texts]

    retriever = LocalCosineRetriever([
        {"title": "A", "content": "doc a"},
        {"title": "B", "content": "doc b"},
    ], embed)

    results = retriever.search("query", top_k=2)

    assert [item["title"] for item in results] == ["A", "B"]
    assert results[0]["score"] > results[1]["score"]


def _rows():
    cases = [
        LabelledRetrievalCase("p1", "positive one", "A", "positive"),
        LabelledRetrievalCase("p2", "positive two", "B", "positive"),
        LabelledRetrievalCase("n1", "negative hard", None, "hard_negative"),
        LabelledRetrievalCase("n2", "negative easy", None, "out_of_domain"),
    ]
    rankings = {
        "positive one": [{"title": "A", "score": 0.9}],
        "positive two": [{"title": "B", "score": 0.7}],
        "negative hard": [{"title": "A", "score": 0.6}],
        "negative easy": [{"title": "B", "score": 0.2}],
    }
    return collect_rankings(lambda query, top_k: rankings[query], cases, top_k=3)


def test_calibration_uses_only_scores_and_prefers_separating_threshold():
    calibrated = calibrate_threshold(_rows())

    assert calibrated["threshold"] == 0.65
    assert calibrated["macro_f1"] == 1.0
    assert calibrated["correct_rejection_rate"] == 1.0


def test_answerability_metrics_count_accept_and_reject_decisions():
    metrics = answerability_metrics(_rows(), threshold=0.65)

    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 0, 0, 2)
    assert metrics["answerable_f1"] == 1.0


def test_summary_requires_correct_top_title_for_decision_accuracy():
    rows = _rows()
    rows[1]["top_titles"] = ["A"]
    rows[1]["rank"] = None
    summary = summarize_retrieval(rows, threshold=0.65)

    assert summary["hit_at_1"] == 0.5
    assert summary["decision_accuracy"] == 0.75
    assert summary["correct_rejection_rate"] == 1.0
    assert summary["hard_negative_rejection_rate"] == 1.0


def test_load_corpus_validates_document_count(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({
        "metadata": {"document_count": 2},
        "documents": [{"title": "A", "content": "body"}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="document_count"):
        load_corpus(path)


def test_load_test_dataset_requires_frozen_reviewed_scoreable_cases(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps({
        "metadata": {
            "frozen": True,
            "total_cases": 2,
            "reviewed_cases": 2,
            "scoreable_total": 2,
        },
        "cases": [
            {
                "id": "p1",
                "query": "answerable",
                "expected_title": "A",
                "case_type": "positive",
                "review": {"status": "verified", "reviewer": "owner"},
            },
            {
                "id": "n1",
                "query": "not answerable",
                "expected_title": None,
                "case_type": "hard_negative",
                "review": {"status": "verified", "reviewer": "owner"},
            },
        ],
    }), encoding="utf-8")

    metadata, cases = load_test_dataset(path)

    assert metadata["frozen"] is True
    assert [case.id for case in cases] == ["p1", "n1"]


def test_load_test_dataset_refuses_unfrozen_data(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps({
        "metadata": {
            "frozen": False,
            "total_cases": 0,
            "reviewed_cases": 0,
            "scoreable_total": 0,
        },
        "cases": [],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="not frozen"):
        load_test_dataset(path)
