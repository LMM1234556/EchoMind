import pytest

from evaluation.run_frozen_intent_test import compute_metrics


def test_compute_metrics_reports_accuracy_and_macro_f1():
    report = compute_metrics([
        {"expected": "query", "predicted": "query", "failed": False},
        {"expected": "query", "predicted": "other", "failed": False},
        {"expected": "other", "predicted": "other", "failed": False},
    ])

    assert report["total"] == 3
    assert report["correct"] == 2
    assert report["accuracy"] == 0.6667
    assert report["per_class"]["query"]["recall"] == 0.5


def test_compute_metrics_refuses_provider_failures():
    with pytest.raises(ValueError, match="must not be scored"):
        compute_metrics([
            {"expected": "query", "predicted": "other", "failed": True},
        ])
