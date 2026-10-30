"""LLM-as-Judge 与人工评分的一致性和校准分析。"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DIMENSIONS: Tuple[str, ...] = (
    "relevance",
    "accuracy",
    "completeness",
    "helpfulness",
)


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sum((x - x_mean) ** 2 for x in xs)
    y_den = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_den * y_den)
    return numerator / denominator if denominator else None


def _ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + end + 2) / 2.0
        for position in range(cursor, end + 1):
            ranks[indexed[position][0]] = average_rank
        cursor = end + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _cohen_kappa(actual: Sequence[bool], predicted: Sequence[bool]) -> Optional[float]:
    if len(actual) != len(predicted) or not actual:
        return None
    observed = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    actual_positive = sum(actual) / len(actual)
    predicted_positive = sum(predicted) / len(predicted)
    expected = (
        actual_positive * predicted_positive
        + (1.0 - actual_positive) * (1.0 - predicted_positive)
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else None


def _normalise_human_score(value: float, scale_min: float, scale_max: float) -> float:
    if scale_max <= scale_min:
        raise ValueError("human score scale_max must be greater than scale_min")
    if value < scale_min or value > scale_max:
        raise ValueError(f"human score {value} outside [{scale_min}, {scale_max}]")
    return (value - scale_min) / (scale_max - scale_min)


def _complete_scores(value: Any) -> bool:
    return isinstance(value, dict) and all(_is_number(value.get(dimension)) for dimension in DIMENSIONS)


def _paired_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = payload.get("metadata") or {}
    scale = metadata.get("human_score_scale") or {"min": 1, "max": 5}
    scale_min = float(scale.get("min", 1))
    scale_max = float(scale.get("max", 5))
    rows: List[Dict[str, Any]] = []

    for case in payload.get("cases") or []:
        judge_scores = case.get("judge_scores")
        if not _complete_scores(judge_scores) or bool(judge_scores.get("judge_failed")):
            continue
        complete_ratings = [
            rating for rating in (case.get("human_ratings") or []) if _complete_scores(rating)
        ]
        if not complete_ratings:
            continue
        for dimension in DIMENSIONS:
            score = float(judge_scores[dimension])
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    f"judge score {score} outside [0, 1] for {case.get('case_id')}:{dimension}"
                )

        human_scores = {
            dimension: statistics.mean(
                _normalise_human_score(float(rating[dimension]), scale_min, scale_max)
                for rating in complete_ratings
            )
            for dimension in DIMENSIONS
        }
        rows.append(
            {
                "case_id": case.get("case_id"),
                "split": case.get("split", "calibration"),
                "human": human_scores,
                "judge": {dimension: float(judge_scores[dimension]) for dimension in DIMENSIONS},
                "reviewer_count": len(complete_ratings),
                "human_ratings": complete_ratings,
            }
        )
    return rows


def _dimension_metrics(
    rows: Sequence[Dict[str, Any]],
    transforms: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    output: Dict[str, Dict[str, Optional[float]]] = {}
    for dimension in DIMENSIONS:
        human = [float(row["human"][dimension]) for row in rows]
        judge = [float(row["judge"][dimension]) for row in rows]
        if transforms and dimension in transforms:
            params = transforms[dimension]
            judge = [
                _clip(params["intercept"] + params["slope"] * score)
                for score in judge
            ]
        errors = [predicted - actual for actual, predicted in zip(human, judge)]
        output[dimension] = {
            "n": len(rows),
            "human_mean": _round(_mean(human)),
            "judge_mean": _round(_mean(judge)),
            "mean_bias": _round(_mean(errors)),
            "mae": _round(_mean([abs(error) for error in errors])),
            "rmse": _round(math.sqrt(statistics.mean([error ** 2 for error in errors]))) if errors else None,
            "pearson_r": _round(_pearson(human, judge)),
            "spearman_rho": _round(_spearman(human, judge)),
            "within_0_20_rate": _round(
                sum(abs(error) <= 0.20 for error in errors) / len(errors) if errors else None
            ),
        }
    return output


def _fit_affine(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    transforms: Dict[str, Dict[str, float]] = {}
    for dimension in DIMENSIONS:
        xs = [float(row["judge"][dimension]) for row in rows]
        ys = [float(row["human"][dimension]) for row in rows]
        if not xs:
            continue
        x_mean = statistics.mean(xs)
        y_mean = statistics.mean(ys)
        variance = sum((value - x_mean) ** 2 for value in xs)
        if variance:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / variance
        else:
            slope = 1.0
        intercept = y_mean - slope * x_mean
        transforms[dimension] = {
            "intercept": round(intercept, 6),
            "slope": round(slope, 6),
        }
    return transforms


def _decision_metrics(
    rows: Sequence[Dict[str, Any]],
    threshold: float,
    transforms: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Optional[float]]:
    human_pass: List[bool] = []
    judge_pass: List[bool] = []
    for row in rows:
        human_overall = statistics.mean(float(row["human"][d]) for d in DIMENSIONS)
        judge_values = []
        for dimension in DIMENSIONS:
            score = float(row["judge"][dimension])
            if transforms and dimension in transforms:
                params = transforms[dimension]
                score = _clip(params["intercept"] + params["slope"] * score)
            judge_values.append(score)
        human_pass.append(human_overall >= threshold)
        judge_pass.append(statistics.mean(judge_values) >= threshold)
    return {
        "n": len(rows),
        "threshold": threshold,
        "agreement_rate": _round(
            sum(a == p for a, p in zip(human_pass, judge_pass)) / len(rows) if rows else None
        ),
        "cohen_kappa": _round(_cohen_kappa(human_pass, judge_pass)),
        "human_pass_rate": _round(sum(human_pass) / len(rows) if rows else None),
        "judge_pass_rate": _round(sum(judge_pass) / len(rows) if rows else None),
    }


def _inter_reviewer_metrics(rows: Sequence[Dict[str, Any]], scale_min: float, scale_max: float) -> Dict[str, Any]:
    differences: List[float] = []
    comparable_cases = 0
    for row in rows:
        ratings = row["human_ratings"]
        if len(ratings) < 2:
            continue
        comparable_cases += 1
        for left_index in range(len(ratings) - 1):
            for right_index in range(left_index + 1, len(ratings)):
                for dimension in DIMENSIONS:
                    left = _normalise_human_score(float(ratings[left_index][dimension]), scale_min, scale_max)
                    right = _normalise_human_score(float(ratings[right_index][dimension]), scale_min, scale_max)
                    differences.append(abs(left - right))
    return {
        "comparable_cases": comparable_cases,
        "pairwise_dimension_comparisons": len(differences),
        "mean_absolute_difference": _round(_mean(differences)),
        "within_0_25_rate": _round(
            sum(value <= 0.25 for value in differences) / len(differences) if differences else None
        ),
    }


def analyse_calibration(payload: Dict[str, Any]) -> Dict[str, Any]:
    """计算原始一致性、仿射校准参数和独立验证集表现。"""

    metadata = payload.get("metadata") or {}
    paired_rows = _paired_rows(payload)
    minimum_reviewers = int(metadata.get("minimum_reviewers_per_case", 2))
    rows = [row for row in paired_rows if row["reviewer_count"] >= minimum_reviewers]
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    threshold = float(metadata.get("pass_threshold", 0.70))
    scale = metadata.get("human_score_scale") or {"min": 1, "max": 5}
    scale_min = float(scale.get("min", 1))
    scale_max = float(scale.get("max", 5))
    transforms = _fit_affine(calibration_rows)
    minimum_cases = int(metadata.get("minimum_completed_cases", 20))

    if not paired_rows:
        status = "awaiting_human_and_judge_scores"
    elif not rows:
        status = "awaiting_required_reviewer_coverage"
    elif len(rows) < minimum_cases:
        status = "pilot_sample_only"
    elif not validation_rows:
        status = "descriptive_agreement_only"
    else:
        status = "calibration_evaluated"

    result: Dict[str, Any] = {
        "name": "llm-judge-calibration-report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "sample": {
            "total_cases": len(payload.get("cases") or []),
            "completed_with_one_or_more_reviewers": len(paired_rows),
            "completed_paired_cases": len(rows),
            "calibration_cases": len(calibration_rows),
            "validation_cases": len(validation_rows),
            "minimum_completed_cases": minimum_cases,
            "minimum_reviewers_per_case": minimum_reviewers,
        },
        "raw_all_cases": {
            "dimensions": _dimension_metrics(rows),
            "decision": _decision_metrics(rows, threshold),
        },
        "inter_reviewer": _inter_reviewer_metrics(rows, scale_min, scale_max),
        "calibration_fit": {
            "method": "per-dimension affine transform fitted on calibration split",
            "parameters": transforms,
        },
        "validation": {
            "raw_dimensions": _dimension_metrics(validation_rows),
            "calibrated_dimensions": _dimension_metrics(validation_rows, transforms),
            "raw_decision": _decision_metrics(validation_rows, threshold),
            "calibrated_decision": _decision_metrics(validation_rows, threshold, transforms),
        },
        "interpretation": (
            "Calibration and validation splits are reported separately. "
            "Validation metrics are the decision basis; all-case metrics are descriptive."
        ),
    }
    return result


def render_markdown(report: Dict[str, Any]) -> str:
    sample = report["sample"]
    lines = [
        "# LLM-as-Judge 校准报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- 完成配对评分：{sample['completed_paired_cases']} / {sample['total_cases']}",
        f"- 校准集：{sample['calibration_cases']}；独立验证集：{sample['validation_cases']}",
        "",
        "## 独立验证集结果",
        "",
        "| 维度 | 原始 MAE | 校准后 MAE | 原始 Spearman | 校准后 Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    raw = report["validation"]["raw_dimensions"]
    calibrated = report["validation"]["calibrated_dimensions"]
    for dimension in DIMENSIONS:
        lines.append(
            f"| {dimension} | {raw[dimension]['mae']} | {calibrated[dimension]['mae']} | "
            f"{raw[dimension]['spearman_rho']} | {calibrated[dimension]['spearman_rho']} |"
        )
    lines.extend(
        [
            "",
            "## 通过判断一致性",
            "",
            f"- 原始一致率：{report['validation']['raw_decision']['agreement_rate']}",
            f"- 校准后一致率：{report['validation']['calibrated_decision']['agreement_rate']}",
            f"- 校准后 Cohen's kappa：{report['validation']['calibrated_decision']['cohen_kappa']}",
            "",
            "> 独立验证集结果用于判断校准效果；样本不足时报告保持试运行状态。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse human/LLM judge calibration scores")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyse_calibration(payload)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
