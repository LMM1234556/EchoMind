"""Run the frozen grounded-chat v2 benchmark against the production /chat API.

The default invocation is a zero-cost dry run.  Real requests require the
explicit ``--execute`` flag.  Results are saved after every case and may only
resume from a checkpoint with matching dataset, freeze manifest and base URL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import httpx

from evaluation.e2e_benchmark import distribution


DATA_DIR = pathlib.Path(__file__).with_name("data")
REPORT_DIR = pathlib.Path(__file__).with_name("reports")
DEFAULT_DATASET = DATA_DIR / "grounded_chat_test_v2.json"
DEFAULT_FREEZE = DATA_DIR / "grounded_chat_test_v2.freeze.json"
DEFAULT_CORPUS = DATA_DIR / "retrieval_corpus_v1.json"
DEFAULT_REPORT = REPORT_DIR / "grounded_chat_test_v2_first_run.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_frozen_dataset(
    dataset_path: pathlib.Path,
    freeze_path: pathlib.Path,
    corpus_path: pathlib.Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset_path = dataset_path.resolve()
    freeze_path = freeze_path.resolve()
    corpus_path = corpus_path.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    if sha256(dataset_path) != freeze.get("dataset_sha256"):
        raise RuntimeError("frozen dataset SHA-256 mismatch")
    if sha256(corpus_path) != freeze.get("corpus_sha256"):
        raise RuntimeError("frozen corpus SHA-256 mismatch")
    metadata = dataset.get("metadata") or {}
    cases = dataset.get("cases") or []
    if metadata.get("frozen") is not True or metadata.get("evaluation_status") != "frozen_not_run":
        raise RuntimeError("dataset is not in frozen_not_run state")
    if freeze.get("first_evaluation_run") is not False:
        raise RuntimeError("freeze manifest does not permit a first run")
    if len(cases) != int(freeze.get("total_cases", -1)):
        raise RuntimeError("frozen case count mismatch")
    if len({str(case.get("id")) for case in cases}) != len(cases):
        raise RuntimeError("case ids must be unique")
    if not all((case.get("review") or {}).get("status") == "verified" for case in cases):
        raise RuntimeError("all frozen cases must be human verified")
    supported = sum(bool(case.get("expected_supported")) for case in cases)
    if supported != int(freeze.get("supported_cases", -1)):
        raise RuntimeError("supported case count mismatch")
    return dataset, freeze


def knowledge_calls(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item
        for item in (trace.get("tool_calls") or [])
        if item.get("tool_name") == "search_knowledge_base"
    ]


def _contains_diagnostics(answer: str, terms: Iterable[str]) -> Dict[str, bool]:
    return {str(term): str(term).lower() in answer.lower() for term in terms}


def evaluate_case(
    case: Dict[str, Any],
    response: Dict[str, Any],
    trace: Dict[str, Any],
    external_latency_ms: float,
) -> Dict[str, Any]:
    calls = knowledge_calls(trace)
    selector_calls = [item for item in calls if item.get("evidence_selector_used") is True]
    evidence_found = any(item.get("evidence_found") is True for item in calls)
    selected_titles = [
        str(item.get("evidence_selected_title"))
        for item in selector_calls
        if str(item.get("evidence_selected_title") or "").strip()
    ]
    selector_success = bool(selector_calls) and all(
        item.get("evidence_selector_success") is True for item in selector_calls
    )
    expected_supported = bool(case["expected_supported"])
    expected_title = case.get("expected_title")
    evidence_classification_correct = evidence_found == expected_supported
    if expected_supported:
        strict_selection_correct = (
            selector_success and evidence_found and str(expected_title) in selected_titles
        )
    else:
        strict_selection_correct = (
            selector_success and not evidence_found and not selected_titles
        )

    answer = str(response.get("response", "") or "")
    required_diagnostics = _contains_diagnostics(
        answer, case.get("expected_answer_facts") or []
    )
    forbidden_diagnostics = _contains_diagnostics(
        answer, case.get("forbidden_claims") or []
    )
    checks = {
        "intent_correct": response.get("intent") == case.get("intended_intent"),
        "selector_exercised": bool(selector_calls),
        "selector_success": selector_success,
        "evidence_classification_correct": evidence_classification_correct,
        "strict_selection_correct": strict_selection_correct,
        "no_runtime_failure": not bool(response.get("failure_kind")),
        "non_empty_response": bool(answer.strip()),
    }
    automatic_pipeline_pass = all(checks.values())
    return {
        "case_id": case["id"],
        "query": case["query"],
        "expected": {
            "intent": case["intended_intent"],
            "supported": expected_supported,
            "title": expected_title,
            "evidence_basis": case["evidence_basis"],
        },
        "actual": response,
        "trace": trace,
        "observed": {
            "evidence_found": evidence_found,
            "selected_titles": selected_titles,
            "evidence_gates": [str(item.get("evidence_gate") or "") for item in calls],
        },
        "checks": checks,
        "automatic_pipeline_pass": automatic_pipeline_pass,
        "answer_review": {
            "status": "pending_human_review",
            "expected_answer_facts": case.get("expected_answer_facts") or [],
            "forbidden_claims": case.get("forbidden_claims") or [],
            "required_string_diagnostics": required_diagnostics,
            "forbidden_string_diagnostics": forbidden_diagnostics,
            "diagnostic_warning": "String matches are not semantic-faithfulness scores.",
        },
        "external_latency_ms": round(float(external_latency_ms), 1),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    supported = [item for item in results if item["expected"]["supported"]]
    gaps = [item for item in results if not item["expected"]["supported"]]
    ratio = lambda value, denominator=total: round(value / denominator, 4) if denominator else 0.0
    count_check = lambda name, rows=results: sum(bool(item["checks"].get(name)) for item in rows)
    false_acceptances = sum(item["observed"]["evidence_found"] for item in gaps)
    correct_rejections = sum(not item["observed"]["evidence_found"] for item in gaps)
    supported_strict = count_check("strict_selection_correct", supported)
    llm_components: Counter[str] = Counter()
    llm_total = 0
    for item in results:
        actual = item["actual"]
        count = actual.get("llm_call_count")
        if isinstance(count, int):
            llm_total += count
        llm_components.update(actual.get("llm_calls_by_component") or {})
    return {
        "completed_cases": total,
        "automatic_metrics": {
            "intent_accuracy": ratio(count_check("intent_correct")),
            "selector_coverage": ratio(count_check("selector_exercised")),
            "selector_success_rate": ratio(count_check("selector_success")),
            "evidence_classification_accuracy": ratio(
                count_check("evidence_classification_correct")
            ),
            "strict_evidence_selection_accuracy": ratio(
                count_check("strict_selection_correct")
            ),
            "supported_selection_accuracy": ratio(supported_strict, len(supported)),
            "correct_rejection_rate": ratio(correct_rejections, len(gaps)),
            "false_acceptance_rate": ratio(false_acceptances, len(gaps)),
            "automatic_pipeline_pass_rate": ratio(
                sum(item["automatic_pipeline_pass"] for item in results)
            ),
        },
        "latency_ms": {
            "external": distribution(item["external_latency_ms"] for item in results),
            "api_total": distribution(
                item["actual"].get("stage_timings", {}).get("api_total_ms", 0.0)
                for item in results
                if "api_total_ms" in item["actual"].get("stage_timings", {})
            ),
        },
        "llm_calls": {
            "foreground_total": llm_total,
            "per_case_mean": round(llm_total / total, 2) if total else 0.0,
            "by_component": dict(sorted(llm_components.items())),
        },
        "answer_faithfulness": {
            "status": "pending_human_review",
            "reviewed_cases": 0,
            "pass_rate": None,
            "claim_boundary": (
                "No semantic answer-faithfulness score is claimed until human review "
                "against the frozen facts and forbidden claims is complete."
            ),
        },
    }


def dry_run_plan(dataset: Dict[str, Any], max_llm_calls: int) -> Dict[str, Any]:
    cases = dataset["cases"]
    supported = sum(bool(case["expected_supported"]) for case in cases)
    gaps = len(cases) - supported
    estimated_calls = supported * 3 + gaps * 2
    return {
        "mode": "dry_run_no_api_calls",
        "cases": len(cases),
        "supported_cases": supported,
        "knowledge_gap_cases": gaps,
        "estimated_foreground_llm_calls": estimated_calls,
        "max_observed_foreground_llm_calls": max_llm_calls,
        "estimate_basis": "supported≈intent+selector+answer; gap≈intent+selector",
        "warning": "The call estimate is not a provider billing guarantee.",
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset, freeze = load_frozen_dataset(args.dataset, args.freeze, args.corpus)
    plan = dry_run_plan(dataset, args.max_llm_calls)
    if not args.execute:
        return plan

    dataset_hash = sha256(args.dataset.resolve())
    freeze_hash = sha256(args.freeze.resolve())
    checkpoint: Dict[str, Any] = {
        "dataset_sha256": dataset_hash,
        "freeze_sha256": freeze_hash,
        "base_url": args.base_url,
        "max_llm_calls": args.max_llm_calls,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "observed_foreground_llm_calls": 0,
    }
    if args.checkpoint.exists():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        expected = (dataset_hash, freeze_hash, args.base_url, args.max_llm_calls)
        actual = (
            checkpoint.get("dataset_sha256"),
            checkpoint.get("freeze_sha256"),
            checkpoint.get("base_url"),
            checkpoint.get("max_llm_calls"),
        )
        if actual != expected:
            raise RuntimeError("checkpoint configuration mismatch")
    elif args.report.exists():
        raise RuntimeError("report already exists without a resumable checkpoint")

    results = list(checkpoint.get("results") or [])
    completed_ids = {item["case_id"] for item in results}

    def save(status: str) -> Dict[str, Any]:
        checkpoint["results"] = results
        checkpoint["observed_foreground_llm_calls"] = sum(
            item["actual"].get("llm_call_count", 0)
            for item in results
            if isinstance(item["actual"].get("llm_call_count"), int)
        )
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.checkpoint, checkpoint)
        report = {
            "name": "grounded-chat-test-v2-first-run",
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": dataset_hash,
            "freeze_manifest": str(args.freeze.resolve()),
            "freeze_sha256": freeze_hash,
            "system_under_test_commit": freeze["system_under_test_commit"],
            "base_url": args.base_url,
            "budget": {
                "max_observed_foreground_llm_calls": args.max_llm_calls,
                "observed_foreground_llm_calls": checkpoint[
                    "observed_foreground_llm_calls"
                ],
            },
            "summary": summarize(results),
            "results": results,
        }
        write_json(args.report, report)
        return report

    with httpx.Client(
        base_url=args.base_url,
        timeout=args.timeout,
        trust_env=False,
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        for case in dataset["cases"]:
            if case["id"] in completed_ids:
                continue
            observed = int(checkpoint.get("observed_foreground_llm_calls", 0))
            if observed + 2 > args.max_llm_calls:
                return save("stopped_before_budget_limit")
            unique = uuid.uuid4().hex[:8]
            started = time.perf_counter()
            try:
                http_response = client.post(
                    "/chat",
                    json={
                        "message": case["query"],
                        "user_id": f"ground-v2-{case['id']}-{unique}",
                        "conv_id": f"ground-v2-{case['id']}-{unique}",
                    },
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                http_response.raise_for_status()
                response = http_response.json()
                trace_response = client.get(f"/trace/tool/{response['request_id']}")
                trace_response.raise_for_status()
                trace = trace_response.json().get("trace") or {}
                evaluated = evaluate_case(case, response, trace, elapsed_ms)
            except Exception as ex:
                elapsed_ms = (time.perf_counter() - started) * 1000
                evaluated = {
                    "case_id": case["id"],
                    "query": case["query"],
                    "expected": {
                        "intent": case["intended_intent"],
                        "supported": case["expected_supported"],
                        "title": case.get("expected_title"),
                        "evidence_basis": case["evidence_basis"],
                    },
                    "actual": {
                        "response": "",
                        "failure_kind": "runner_exception",
                        "llm_call_count": None,
                        "llm_calls_by_component": {},
                        "stage_timings": {},
                    },
                    "trace": {},
                    "observed": {
                        "evidence_found": False,
                        "selected_titles": [],
                        "evidence_gates": [],
                    },
                    "checks": {
                        "intent_correct": False,
                        "selector_exercised": False,
                        "selector_success": False,
                        "evidence_classification_correct": False,
                        "strict_selection_correct": False,
                        "no_runtime_failure": False,
                        "non_empty_response": False,
                    },
                    "automatic_pipeline_pass": False,
                    "answer_review": {
                        "status": "not_reviewable_runner_failure",
                        "expected_answer_facts": case.get("expected_answer_facts") or [],
                        "forbidden_claims": case.get("forbidden_claims") or [],
                    },
                    "external_latency_ms": round(elapsed_ms, 1),
                    "error": f"{type(ex).__name__}: {ex}",
                }
                results.append(evaluated)
                return save("stopped_after_runner_failure")
            results.append(evaluated)
            checkpoint["observed_foreground_llm_calls"] = sum(
                item["actual"].get("llm_call_count", 0)
                for item in results
                if isinstance(item["actual"].get("llm_call_count"), int)
            )
            report = save("running")
            print(
                f"[{len(results):02d}/{len(dataset['cases'])}] {case['id']} "
                f"strict={evaluated['checks']['strict_selection_correct']} "
                f"llm_calls={response.get('llm_call_count')} "
                f"observed_total={checkpoint['observed_foreground_llm_calls']} "
                f"latency_ms={elapsed_ms:.1f}",
                flush=True,
            )
            if checkpoint["observed_foreground_llm_calls"] > args.max_llm_calls:
                return save("stopped_after_budget_limit_exceeded")

    return save("completed_automatic_pending_human_review")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET)
    parser.add_argument("--freeze", type=pathlib.Path, default=DEFAULT_FREEZE)
    parser.add_argument("--corpus", type=pathlib.Path, default=DEFAULT_CORPUS)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=DEFAULT_REPORT.with_suffix(".checkpoint.json"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-llm-calls", type=int, default=80)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call /chat. Without this flag the runner only prints a dry-run plan.",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.max_llm_calls < 2:
        parser.error("--max-llm-calls must be at least 2")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.execute:
        print(f"report={args.report.resolve()}")
        print(f"checkpoint={args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
