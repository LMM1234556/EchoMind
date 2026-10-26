"""Shared metrics and threshold calibration for paired retrieval evaluation."""
from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class LabelledRetrievalCase:
    id: str
    query: str
    expected_title: Optional[str]
    case_type: str


def load_corpus(path: pathlib.Path) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    documents = list(payload.get("documents") or [])
    if metadata.get("document_count") != len(documents):
        raise ValueError("corpus document_count does not match documents")
    titles = [str(item.get("title", "")) for item in documents]
    if len(titles) != len(set(titles)) or not all(titles):
        raise ValueError("corpus titles must be non-empty and unique")
    if any(not str(item.get("content", "")).strip() for item in documents):
        raise ValueError("corpus documents must have non-empty content")
    return metadata, documents


def normalize_documents(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Compare corpus content without relying on collection insertion order."""
    return sorted(
        [
            {
                "title": str(item.get("title", "")),
                "content": str(item.get("content", "")),
            }
            for item in documents
        ],
        key=lambda item: (item["title"], item["content"]),
    )


class LocalCosineRetriever:
    """Evaluate embedding quality without depending on ChromaDB server state."""

    def __init__(
        self,
        documents: Sequence[Dict[str, str]],
        embedding_function: Callable[[Sequence[str]], Sequence[Any]],
    ) -> None:
        self._documents = [dict(item) for item in documents]
        self._embedding_function = embedding_function
        self._document_vectors = self._normalize(
            embedding_function([item["content"] for item in self._documents])
        )
        if self._document_vectors.shape[0] != len(self._documents):
            raise ValueError("embedding function returned the wrong document count")

    @staticmethod
    def _normalize(vectors: Sequence[Any]) -> np.ndarray:
        matrix = np.asarray(list(vectors), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embedding function must return a non-empty 2D matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embedding function returned a zero vector")
        return matrix / norms

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_matrix = self._normalize(self._embedding_function([query]))
        if query_matrix.shape[1] != self._document_vectors.shape[1]:
            raise ValueError("query and document embedding dimensions differ")
        similarities = self._document_vectors @ query_matrix[0]
        order = sorted(
            range(len(self._documents)),
            key=lambda index: (-float(similarities[index]), index),
        )
        return [
            {
                **self._documents[index],
                "score": round(float(similarities[index]), 8),
            }
            for index in order[:max(1, top_k)]
        ]


def collect_rankings(
    search: Callable[[str, int], List[Dict[str, Any]]],
    cases: Sequence[LabelledRetrievalCase],
    top_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        started = time.monotonic()
        results = search(case.query, top_k)
        latency_ms = (time.monotonic() - started) * 1000
        titles = [str(item.get("title", "")) for item in results]
        scores = [float(item["score"]) for item in results if item.get("score") is not None]
        rank = None
        if case.expected_title is not None and case.expected_title in titles:
            rank = titles.index(case.expected_title) + 1
        rows.append({
            "id": case.id,
            "query": case.query,
            "expected_title": case.expected_title,
            "case_type": case.case_type,
            "rank": rank,
            "top_titles": titles,
            "top_scores": scores,
            "top_score": scores[0] if scores else None,
            "latency_ms": round(latency_ms, 1),
        })
    return rows


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def answerability_metrics(rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        expected_answerable = row.get("expected_title") is not None
        score = row.get("top_score")
        predicted_answerable = score is not None and float(score) >= threshold
        if expected_answerable and predicted_answerable:
            tp += 1
        elif not expected_answerable and predicted_answerable:
            fp += 1
        elif expected_answerable and not predicted_answerable:
            fn += 1
        else:
            tn += 1

    positive_precision = _safe_div(tp, tp + fp)
    positive_recall = _safe_div(tp, tp + fn)
    positive_f1 = _safe_div(
        2 * positive_precision * positive_recall,
        positive_precision + positive_recall,
    )
    negative_precision = _safe_div(tn, tn + fn)
    negative_recall = _safe_div(tn, tn + fp)
    negative_f1 = _safe_div(
        2 * negative_precision * negative_recall,
        negative_precision + negative_recall,
    )
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "answerable_precision": round(positive_precision, 4),
        "answerable_recall": round(positive_recall, 4),
        "answerable_f1": round(positive_f1, 4),
        "correct_rejection_rate": round(negative_recall, 4),
        "macro_f1": round((positive_f1 + negative_f1) / 2, 4),
    }


def threshold_candidates(rows: Sequence[Dict[str, Any]]) -> List[float]:
    values = sorted({float(row["top_score"]) for row in rows if row.get("top_score") is not None})
    if not values:
        raise ValueError("cannot calibrate threshold without top scores")
    scale = max(1.0, max(abs(value) for value in values))
    epsilon = scale * 1e-9
    candidates = [values[0] - epsilon]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + epsilon)
    return candidates


def calibrate_threshold(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Maximize binary macro-F1; ties prefer rejection rate then threshold."""
    scored = [answerability_metrics(rows, threshold) for threshold in threshold_candidates(rows)]
    best = max(
        scored,
        key=lambda item: (
            item["macro_f1"],
            item["correct_rejection_rate"],
            item["threshold"],
        ),
    )
    return {
        **best,
        "threshold": round(float(best["threshold"]), 8),
        "candidate_count": len(scored),
        "selection_rule": "max_macro_f1_then_correct_rejection_then_higher_threshold",
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_retrieval(
    rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    positives = [row for row in rows if row.get("expected_title") is not None]
    negatives = [row for row in rows if row.get("expected_title") is None]
    reciprocal_rank = sum(1.0 / row["rank"] for row in positives if row.get("rank"))
    decision_correct = 0
    hard_negative_total = hard_negative_rejected = 0
    for row in rows:
        score = row.get("top_score")
        accepted = score is not None and float(score) >= threshold
        if row.get("expected_title") is None:
            decision_correct += int(not accepted)
            if row.get("case_type") == "hard_negative":
                hard_negative_total += 1
                hard_negative_rejected += int(not accepted)
        else:
            decision_correct += int(accepted and row.get("rank") == 1)

    latency = [float(row.get("latency_ms", 0.0)) for row in rows]
    answerability = answerability_metrics(rows, threshold)
    return {
        "total": len(rows),
        "positive_total": len(positives),
        "negative_total": len(negatives),
        "threshold": round(float(threshold), 8),
        "hit_at_1": round(_safe_div(sum(row.get("rank") == 1 for row in positives), len(positives)), 4),
        "hit_at_3": round(_safe_div(sum(row.get("rank") is not None and row["rank"] <= 3 for row in positives), len(positives)), 4),
        "mrr_at_k": round(_safe_div(reciprocal_rank, len(positives)), 4),
        "decision_accuracy": round(_safe_div(decision_correct, len(rows)), 4),
        "answerable_precision": answerability["answerable_precision"],
        "answerable_recall": answerability["answerable_recall"],
        "answerable_f1": answerability["answerable_f1"],
        "correct_rejection_rate": answerability["correct_rejection_rate"],
        "hard_negative_rejection_rate": (
            round(_safe_div(hard_negative_rejected, hard_negative_total), 4)
            if hard_negative_total else None
        ),
        "latency_ms": {
            "mean": round(_safe_div(sum(latency), len(latency)), 1),
            "p50": round(_percentile(latency, 0.50), 1),
            "p95": round(_percentile(latency, 0.95), 1),
            "max": round(max(latency), 1) if latency else 0.0,
        },
    }
