import pytest

from evaluation.judge_calibration import _paired_rows, analyse_calibration


def test_adjudicated_reference_keeps_independent_reviewer_scores():
    case = _case("resolved", "calibration", [3, 4, 4, 4], [0.5, 0.75, 0.75, 0.75], [5, 4, 4, 4])
    case["human_reference_overrides"] = {"relevance": {
        "score": 3, "status": "confirmed", "confirmed_by": "project_owner",
        "reason": "Core response relevant with substantial redundant requests.",
    }}
    rows = _paired_rows({"cases": [case]})
    assert rows[0]["human"]["relevance"] == 0.5
    assert rows[0]["human"]["accuracy"] == 0.75
    assert [r["relevance"] for r in rows[0]["human_ratings"]] == [3, 5]
    report = analyse_calibration({"cases": [case]})
    assert report["inter_reviewer"]["mean_absolute_difference"] == 0.125


@pytest.mark.parametrize("override", [
    {"score": 6, "status": "confirmed", "confirmed_by": "project_owner", "reason": "outside scale"},
    {"score": 3, "status": "draft", "confirmed_by": "project_owner", "reason": "pending"},
    {"score": True, "status": "confirmed", "confirmed_by": "project_owner", "reason": "not numeric"},
])
def test_invalid_adjudicated_reference_is_rejected(override):
    case = _case("invalid", "calibration", [3, 4, 4, 4], [0.5, 0.75, 0.75, 0.75])
    case["human_reference_overrides"] = {"relevance": override}
    with pytest.raises(ValueError):
        _paired_rows({"cases": [case]})


def _case(case_id, split, human_values, judge_values, second_reviewer=None):
    ratings = [
        {
            "reviewer_id": "r1",
            "relevance": human_values[0],
            "accuracy": human_values[1],
            "completeness": human_values[2],
            "helpfulness": human_values[3],
        }
    ]
    if second_reviewer:
        ratings.append(
            {
                "reviewer_id": "r2",
                "relevance": second_reviewer[0],
                "accuracy": second_reviewer[1],
                "completeness": second_reviewer[2],
                "helpfulness": second_reviewer[3],
            }
        )
    return {
        "case_id": case_id,
        "split": split,
        "human_ratings": ratings,
        "judge_scores": {
            "relevance": judge_values[0],
            "accuracy": judge_values[1],
            "completeness": judge_values[2],
            "helpfulness": judge_values[3],
        },
    }


def test_calibration_reports_separate_validation_metrics():
    payload = {
        "metadata": {
            "human_score_scale": {"min": 1, "max": 5},
            "pass_threshold": 0.7,
            "minimum_completed_cases": 4,
            "minimum_reviewers_per_case": 1,
        },
        "cases": [
            _case("c1", "calibration", [5, 5, 4, 4], [0.8, 0.8, 0.7, 0.7], [5, 4, 4, 4]),
            _case("c2", "calibration", [2, 2, 2, 2], [0.4, 0.4, 0.4, 0.4]),
            _case("c3", "calibration", [4, 4, 4, 4], [0.7, 0.7, 0.7, 0.7]),
            _case("c4", "calibration", [3, 3, 3, 3], [0.6, 0.6, 0.6, 0.6]),
            _case("v1", "validation", [5, 5, 5, 5], [0.85, 0.85, 0.85, 0.85]),
            _case("v2", "validation", [2, 2, 2, 2], [0.35, 0.35, 0.35, 0.35]),
        ],
    }

    report = analyse_calibration(payload)

    assert report["status"] == "calibration_evaluated"
    assert report["sample"]["completed_paired_cases"] == 6
    assert report["sample"]["validation_cases"] == 2
    assert set(report["calibration_fit"]["parameters"]) == {
        "relevance",
        "accuracy",
        "completeness",
        "helpfulness",
    }
    assert report["validation"]["raw_dimensions"]["accuracy"]["n"] == 2
    assert report["inter_reviewer"]["comparable_cases"] == 1


def test_calibration_waits_for_paired_scores():
    payload = {
        "metadata": {"human_score_scale": {"min": 1, "max": 5}},
        "cases": [
            {
                "case_id": "pending",
                "split": "calibration",
                "human_ratings": [{"reviewer_id": "r1", "relevance": None}],
                "judge_scores": None,
            }
        ],
    }

    report = analyse_calibration(payload)

    assert report["status"] == "awaiting_human_and_judge_scores"
    assert report["sample"]["completed_paired_cases"] == 0


def test_calibration_rejects_out_of_range_judge_score():
    payload = {
        "metadata": {
            "human_score_scale": {"min": 1, "max": 5},
            "minimum_reviewers_per_case": 1,
        },
        "cases": [
            _case("bad", "calibration", [4, 4, 4, 4], [1.2, 0.8, 0.8, 0.8]),
        ],
    }

    with pytest.raises(ValueError, match="outside"):
        analyse_calibration(payload)
