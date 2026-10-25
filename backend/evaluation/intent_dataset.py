"""Auditable intent-evaluation dataset loading and validation.

The dataset deliberately distinguishes AI-drafted candidates from
human-verified labels. Only verified cases may be converted into scoreable
``IntentTestCase`` objects. This prevents draft labels from being reported as
measured model accuracy.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from core.intent_recognizer import IntentCategory, _TEMPLATES


ALLOWED_LABEL_STATUSES = {"draft", "verified", "rejected"}
ALLOWED_SPLITS = {"candidate", "train", "dev", "test"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


class DatasetValidationError(ValueError):
    """Raised when an intent dataset cannot be audited safely."""


@dataclass(frozen=True)
class IntentDatasetCase:
    id: str
    message: str
    expected_intent: str
    split: str
    difficulty: str
    source: str
    label_status: str
    reviewer: str = ""
    notes: str = ""


def normalize_message(message: str) -> str:
    """Normalize text for exact-duplicate and template-leakage checks."""
    return re.sub(r"[\W_]+", "", str(message or "").lower(), flags=re.UNICODE)


def _case_from_dict(raw: Dict[str, Any], index: int) -> IntentDatasetCase:
    required = {
        "id", "message", "expected_intent", "split", "difficulty",
        "source", "label_status",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise DatasetValidationError(
            f"case[{index}] missing required fields: {', '.join(missing)}"
        )
    return IntentDatasetCase(
        id=str(raw["id"]).strip(),
        message=str(raw["message"]).strip(),
        expected_intent=str(raw["expected_intent"]).strip(),
        split=str(raw["split"]).strip(),
        difficulty=str(raw["difficulty"]).strip(),
        source=str(raw["source"]).strip(),
        label_status=str(raw["label_status"]).strip(),
        reviewer=str(raw.get("reviewer", "")).strip(),
        notes=str(raw.get("notes", "")).strip(),
    )


def load_dataset(path: str | pathlib.Path) -> tuple[Dict[str, Any], List[IntentDatasetCase]]:
    """Load a JSON dataset and validate its schema-level constraints."""
    dataset_path = pathlib.Path(path)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise DatasetValidationError("dataset must be an object containing a cases list")

    cases = [_case_from_dict(raw, index) for index, raw in enumerate(data["cases"])]
    errors: List[str] = []
    allowed_intents = {intent.value for intent in IntentCategory}
    seen_ids: set[str] = set()
    seen_messages: Dict[str, str] = {}

    for case in cases:
        if not case.id:
            errors.append("case id must not be empty")
        elif case.id in seen_ids:
            errors.append(f"duplicate case id: {case.id}")
        seen_ids.add(case.id)

        normalized = normalize_message(case.message)
        if not normalized:
            errors.append(f"{case.id}: message must not be empty")
        elif normalized in seen_messages:
            errors.append(
                f"duplicate message: {case.id} and {seen_messages[normalized]}"
            )
        else:
            seen_messages[normalized] = case.id

        if case.expected_intent not in allowed_intents:
            errors.append(f"{case.id}: unknown intent {case.expected_intent!r}")
        if case.split not in ALLOWED_SPLITS:
            errors.append(f"{case.id}: unknown split {case.split!r}")
        if case.difficulty not in ALLOWED_DIFFICULTIES:
            errors.append(f"{case.id}: unknown difficulty {case.difficulty!r}")
        if case.label_status not in ALLOWED_LABEL_STATUSES:
            errors.append(f"{case.id}: unknown label_status {case.label_status!r}")
        if case.label_status == "verified" and not case.reviewer:
            errors.append(f"{case.id}: verified labels require a reviewer")
        if not case.source:
            errors.append(f"{case.id}: source must not be empty")

    if data.get("frozen") is True:
        if not str(data.get("baseline_commit", "")).strip():
            errors.append("frozen dataset requires baseline_commit")
        for case in cases:
            if case.label_status != "verified" or case.split != "test":
                errors.append(
                    f"{case.id}: frozen dataset requires verified/test cases"
                )

    if errors:
        raise DatasetValidationError("; ".join(errors))
    return data, cases


def audit_dataset(
    path: str | pathlib.Path,
    compare_paths: Optional[Iterable[str | pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Return distribution and leakage warnings without scoring the model."""
    metadata, cases = load_dataset(path)
    template_texts = {
        normalize_message(template)
        for templates in _TEMPLATES.values()
        for template in templates
    }
    template_overlaps = [
        case.id for case in cases if normalize_message(case.message) in template_texts
    ]
    comparison_messages: Dict[str, List[str]] = {}
    for compare_path in compare_paths or []:
        compare_metadata, compare_cases = load_dataset(compare_path)
        compare_name = str(
            compare_metadata.get("name") or pathlib.Path(compare_path).name
        )
        for compare_case in compare_cases:
            normalized = normalize_message(compare_case.message)
            comparison_messages.setdefault(normalized, []).append(
                f"{compare_name}:{compare_case.id}"
            )
    cross_dataset_overlaps = [
        {
            "id": case.id,
            "matches": comparison_messages[normalize_message(case.message)],
        }
        for case in cases
        if normalize_message(case.message) in comparison_messages
    ]
    verified = [case for case in cases if case.label_status == "verified"]
    scoreable = [case for case in verified if case.split in {"dev", "test"}]

    warnings: List[str] = []
    if template_overlaps:
        warnings.append(
            "exact overlap with recognizer templates: " + ", ".join(template_overlaps)
        )
    if cross_dataset_overlaps:
        warnings.append(
            "exact overlap with comparison datasets: "
            + ", ".join(item["id"] for item in cross_dataset_overlaps)
        )
    if not scoreable:
        warnings.append("no human-verified dev/test cases are available for scoring")

    return {
        "name": str(metadata.get("name", "")),
        "version": str(metadata.get("version", "")),
        "frozen": metadata.get("frozen") is True,
        "baseline_commit": str(metadata.get("baseline_commit", "")),
        "total": len(cases),
        "status_counts": dict(sorted(Counter(c.label_status for c in cases).items())),
        "split_counts": dict(sorted(Counter(c.split for c in cases).items())),
        "difficulty_counts": dict(sorted(Counter(c.difficulty for c in cases).items())),
        "intent_counts": dict(sorted(Counter(c.expected_intent for c in cases).items())),
        "template_overlap_ids": template_overlaps,
        "cross_dataset_overlaps": cross_dataset_overlaps,
        "scoreable_total": len(scoreable),
        "warnings": warnings,
    }


def scoreable_intent_cases(path: str | pathlib.Path):
    """Convert only human-verified dev/test records into evaluator cases."""
    from evaluation.evaluator import IntentTestCase

    _, cases = load_dataset(path)
    scoreable = [
        case for case in cases
        if case.label_status == "verified" and case.split in {"dev", "test"}
    ]
    if not scoreable:
        raise DatasetValidationError(
            "dataset has no human-verified dev/test cases; audit and review labels first"
        )
    return [
        IntentTestCase(case.message, case.expected_intent)
        for case in scoreable
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to the intent dataset JSON file")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="another dataset whose messages must not overlap (repeatable)",
    )
    args = parser.parse_args()
    print(json.dumps(
        audit_dataset(args.path, compare_paths=args.compare),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
