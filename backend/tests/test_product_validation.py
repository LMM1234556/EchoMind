import pytest

from evaluation.product_validation import analyse_product_validation


def test_product_validation_computes_primary_metrics_and_guardrails():
    payload = {
        "metadata": {"minimum_participants": 2},
        "scenarios": [
            {"id": "refund", "name": "退款政策查询"},
            {"id": "handoff", "name": "人工交接"},
        ],
        "sessions": [
            {
                "participant_id": "p01",
                "tasks": [
                    {
                        "scenario_id": "refund",
                        "outcome": "success",
                        "time_seconds": 50,
                        "trust_rating": 4,
                        "evidence_clarity_rating": 5,
                        "ease_rating": 4,
                        "assistance_count": 0,
                        "critical_error": False,
                        "timed_out": False,
                    },
                    {
                        "scenario_id": "handoff",
                        "outcome": "partial",
                        "time_seconds": 100,
                        "trust_rating": 3,
                        "evidence_clarity_rating": 3,
                        "ease_rating": 2,
                        "assistance_count": 1,
                        "critical_error": False,
                        "timed_out": False,
                    },
                ],
            },
            {
                "participant_id": "p02",
                "tasks": [
                    {
                        "scenario_id": "refund",
                        "outcome": "success",
                        "time_seconds": 70,
                        "trust_rating": 5,
                        "evidence_clarity_rating": 4,
                        "ease_rating": 4,
                        "assistance_count": 0,
                        "critical_error": False,
                        "timed_out": False,
                    },
                    {
                        "scenario_id": "handoff",
                        "outcome": "failed",
                        "time_seconds": 120,
                        "trust_rating": 2,
                        "evidence_clarity_rating": 2,
                        "ease_rating": 2,
                        "assistance_count": 1,
                        "critical_error": True,
                        "timed_out": True,
                    },
                ],
            },
        ],
    }

    report = analyse_product_validation(payload)

    assert report["status"] == "formative_round_complete"
    assert report["sample"]["participants"] == 2
    assert report["primary_kpis"]["strict_task_completion_rate"] == 0.5
    assert report["primary_kpis"]["median_successful_task_time_seconds"] == 60.0
    assert report["primary_kpis"]["mean_trust_rating_1_to_5"] == 3.5
    assert report["guardrails"]["critical_error_rate"] == 0.25
    assert report["guardrails"]["timeout_rate"] == 0.25


def test_product_validation_rejects_unknown_outcome():
    with pytest.raises(ValueError, match="invalid outcome"):
        analyse_product_validation(
            {
                "sessions": [
                    {
                        "participant_id": "p01",
                        "tasks": [{"scenario_id": "refund", "outcome": "almost"}],
                    }
                ]
            }
        )


def test_product_validation_rejects_invalid_rating():
    with pytest.raises(ValueError, match="invalid trust_rating"):
        analyse_product_validation(
            {
                "sessions": [
                    {
                        "participant_id": "p01",
                        "tasks": [
                            {
                                "scenario_id": "refund",
                                "outcome": "success",
                                "trust_rating": 6,
                            }
                        ],
                    }
                ]
            }
        )
