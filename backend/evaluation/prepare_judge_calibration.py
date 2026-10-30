"""从冻结 Grounded Chat 报告生成盲评校准工作表。"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import date
from typing import Any, Dict, Iterable, Optional


def build_workbook(source: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    cases = []
    for index, item in enumerate(source.get("results") or []):
        expected = item.get("expected") or {}
        actual = item.get("actual") or {}
        case_id = str(item.get("case_id") or f"case-{index + 1:03d}")
        cases.append(
            {
                "case_id": case_id,
                "split": "validation" if (index + 1) % 3 == 0 else "calibration",
                "question": item.get("query", ""),
                "response": actual.get("response", ""),
                "reference_context": {
                    "expected_supported": expected.get("supported"),
                    "expected_title": expected.get("title"),
                    "evidence_basis": expected.get("evidence_basis"),
                    "expected_answer_facts": (item.get("answer_review") or {}).get("expected_answer_facts", []),
                    "forbidden_claims": (item.get("answer_review") or {}).get("forbidden_claims", []),
                },
                "human_ratings": [
                    {
                        "reviewer_id": "reviewer_1",
                        "relevance": None,
                        "accuracy": None,
                        "completeness": None,
                        "helpfulness": None,
                        "notes": "",
                    },
                    {
                        "reviewer_id": "reviewer_2",
                        "relevance": None,
                        "accuracy": None,
                        "completeness": None,
                        "helpfulness": None,
                        "notes": "",
                    },
                ],
                "judge_scores": None,
            }
        )
    return {
        "metadata": {
            "name": "llm-judge-calibration-v1",
            "created_on": date.today().isoformat(),
            "status": "awaiting_blind_human_ratings",
            "source_report": source_path,
            "human_score_scale": {"min": 1, "max": 5},
            "pass_threshold": 0.70,
            "minimum_completed_cases": 20,
            "minimum_reviewers_per_case": 2,
            "dimensions": ["relevance", "accuracy", "completeness", "helpfulness"],
            "review_protocol": "Reviewers score independently before judge scores are generated.",
        },
        "cases": cases,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare blind human-rating workbook")
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {args.output}; pass --force to replace")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    workbook = build_workbook(source, args.source.as_posix())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workbook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {len(workbook['cases'])} cases at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
