"""Utilities for evaluating whether retrieved documents contain enough evidence."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Sequence

from evaluation.retrieval_comparison import LabelledRetrievalCase


@dataclass(frozen=True)
class EvidenceGateCase:
    id: str
    query: str
    candidate_title: str
    expected_supported: bool
    label_basis: str

    def as_retrieval_case(self) -> LabelledRetrievalCase:
        return LabelledRetrievalCase(
            id=self.id,
            query=self.query,
            expected_title=self.candidate_title if self.expected_supported else None,
            case_type="evidence_supported" if self.expected_supported else "knowledge_gap",
        )


def load_evidence_gate_dataset(
    path: pathlib.Path,
) -> tuple[Dict[str, Any], List[EvidenceGateCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    raw_cases = list(payload.get("cases") or [])
    if metadata.get("total_cases") != len(raw_cases):
        raise ValueError("evidence gate total_cases does not match cases")
    if metadata.get("reviewed_cases") != len(raw_cases):
        raise ValueError("all evidence gate cases must be reviewed")
    if metadata.get("scoreable_total") != len(raw_cases):
        raise ValueError("all evidence gate cases must be scoreable")

    cases: List[EvidenceGateCase] = []
    seen_ids = set()
    seen_queries = set()
    for item in raw_cases:
        case_id = str(item.get("id", "")).strip()
        query = str(item.get("query", "")).strip()
        candidate_title = str(item.get("candidate_title", "")).strip()
        expected_supported = item.get("expected_supported")
        label_basis = str(item.get("label_basis", "")).strip()
        review = item.get("review") or {}
        if not case_id or not query or not candidate_title or not label_basis:
            raise ValueError("evidence gate case has invalid required fields")
        if not isinstance(expected_supported, bool):
            raise ValueError("expected_supported must be boolean")
        if case_id in seen_ids or query in seen_queries:
            raise ValueError("evidence gate ids and queries must be unique")
        if review.get("status") != "verified" or not review.get("reviewer"):
            raise ValueError("evidence gate case is not verified")
        seen_ids.add(case_id)
        seen_queries.add(query)
        cases.append(EvidenceGateCase(
            id=case_id,
            query=query,
            candidate_title=candidate_title,
            expected_supported=expected_supported,
            label_basis=label_basis,
        ))

    supported = sum(case.expected_supported for case in cases)
    gaps = len(cases) - supported
    if metadata.get("supported_cases") != supported:
        raise ValueError("supported_cases does not match cases")
    if metadata.get("knowledge_gap_cases") != gaps:
        raise ValueError("knowledge_gap_cases does not match cases")
    return metadata, cases


def annotate_rankings(
    rows: Sequence[Dict[str, Any]],
    cases: Sequence[EvidenceGateCase],
) -> List[Dict[str, Any]]:
    by_id = {case.id: case for case in cases}
    if len(rows) != len(cases) or set(by_id) != {str(row.get("id")) for row in rows}:
        raise ValueError("ranking rows do not match evidence gate cases")

    annotated: List[Dict[str, Any]] = []
    for row in rows:
        case = by_id[str(row["id"])]
        titles = list(row.get("top_titles") or [])
        scores = [float(value) for value in row.get("top_scores") or []]
        candidate_rank = titles.index(case.candidate_title) + 1 if case.candidate_title in titles else None
        candidate_score = scores[candidate_rank - 1] if candidate_rank is not None else None
        annotated.append({
            **row,
            "candidate_title": case.candidate_title,
            "expected_supported": case.expected_supported,
            "label_basis": case.label_basis,
            "top_title": titles[0] if titles else None,
            "second_score": scores[1] if len(scores) > 1 else None,
            "score_margin": round(scores[0] - scores[1], 8) if len(scores) > 1 else None,
            "candidate_rank": candidate_rank,
            "candidate_score": candidate_score,
        })
    return annotated


def _lexical_units(text: str) -> set[str]:
    normalized = str(text or "").lower()
    units = set(re.findall(r"[a-z0-9]+", normalized))
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(segment) == 1:
            units.add(segment)
        else:
            units.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return units


def add_lexical_coverage(
    rows: Sequence[Dict[str, Any]],
    documents: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    contents = {str(item.get("title", "")): str(item.get("content", "")) for item in documents}
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        top_title = str(row.get("top_title") or "")
        if top_title not in contents:
            raise ValueError(f"top title missing from documents: {top_title!r}")
        query_units = _lexical_units(str(row.get("query", "")))
        document_units = _lexical_units(contents[top_title])
        matched = query_units & document_units
        coverage = len(matched) / len(query_units) if query_units else 0.0
        enriched.append({
            **row,
            "query_unit_count": len(query_units),
            "matched_query_unit_count": len(matched),
            "lexical_coverage": round(coverage, 8),
        })
    return enriched


def evidence_gate_metrics(
    rows: Sequence[Dict[str, Any]],
    threshold: float,
    coverage_threshold: float | None = None,
) -> Dict[str, Any]:
    tp = fp = fn = tn = strict_correct = 0
    for row in rows:
        expected = bool(row.get("expected_supported"))
        score = row.get("top_score")
        predicted = score is not None and float(score) >= threshold
        if coverage_threshold is not None:
            predicted = predicted and float(row.get("lexical_coverage", 0.0)) >= coverage_threshold
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
        if expected:
            strict_correct += int(predicted and row.get("top_title") == row.get("candidate_title"))
        else:
            strict_correct += int(not predicted)

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    positive_precision = safe_div(tp, tp + fp)
    positive_recall = safe_div(tp, tp + fn)
    positive_f1 = safe_div(2 * positive_precision * positive_recall, positive_precision + positive_recall)
    negative_precision = safe_div(tn, tn + fn)
    negative_recall = safe_div(tn, tn + fp)
    negative_f1 = safe_div(2 * negative_precision * negative_recall, negative_precision + negative_recall)
    metrics = {
        "threshold": round(float(threshold), 8),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(safe_div(tp + tn, len(rows)), 4),
        "strict_decision_accuracy": round(safe_div(strict_correct, len(rows)), 4),
        "supported_precision": round(positive_precision, 4),
        "supported_recall": round(positive_recall, 4),
        "supported_f1": round(positive_f1, 4),
        "correct_rejection_rate": round(negative_recall, 4),
        "macro_f1": round((positive_f1 + negative_f1) / 2, 4),
    }
    if coverage_threshold is not None:
        metrics["coverage_threshold"] = round(float(coverage_threshold), 8)
    return metrics


def _threshold_candidates(values: Sequence[float]) -> List[float]:
    ordered = sorted({float(value) for value in values})
    if not ordered:
        raise ValueError("cannot calibrate without feature values")
    epsilon = max(1.0, max(abs(value) for value in ordered)) * 1e-9
    return [
        ordered[0] - epsilon,
        *((left + right) / 2 for left, right in zip(ordered, ordered[1:])),
        ordered[-1] + epsilon,
    ]


def calibrate_score_coverage_gate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Calibrate an interpretable AND gate on development data only."""
    score_thresholds = _threshold_candidates([
        float(row["top_score"]) for row in rows if row.get("top_score") is not None
    ])
    coverage_thresholds = _threshold_candidates([
        float(row["lexical_coverage"]) for row in rows
    ])
    scored = [
        evidence_gate_metrics(rows, score_threshold, coverage_threshold)
        for score_threshold in score_thresholds
        for coverage_threshold in coverage_thresholds
    ]
    best = max(
        scored,
        key=lambda item: (
            item["macro_f1"],
            item["strict_decision_accuracy"],
            item["correct_rejection_rate"],
            item["supported_recall"],
            item["coverage_threshold"],
            item["threshold"],
        ),
    )
    return {
        **best,
        "candidate_count": len(scored),
        "selection_rule": (
            "max_macro_f1_then_strict_accuracy_then_rejection_then_recall_"
            "then_higher_thresholds"
        ),
        "rule": "top_score >= threshold AND lexical_coverage >= coverage_threshold",
    }
