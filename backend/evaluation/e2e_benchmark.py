"""Run the fixed EchoMind end-to-end pilot benchmark against a live API.

This runner deliberately separates three claims:

* intent/route accuracy: exact comparison against human-labelled targets;
* response-contract pass rate: deterministic checks for required facts and
  safety behaviour;
* semantic answer quality: not claimed by this script and still requires
  human review or a separately calibrated judge.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List

import httpx


DEFAULT_CASES = pathlib.Path(__file__).with_name("e2e_cases.json")


def percentile(values: Iterable[float], quantile: float) -> float:
    """Linear percentile with deterministic behaviour for small pilot sets."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> Dict[str, float]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(items),
        "mean": round(statistics.mean(items), 1),
        "p50": round(percentile(items, 0.50), 1),
        "p95": round(percentile(items, 0.95), 1),
        "max": round(max(items), 1),
    }


def _knowledge_trace(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item
        for item in trace.get("tool_calls", [])
        if item.get("tool_name") == "search_knowledge_base"
    ]


def evaluate_turn(
    expected: Dict[str, Any],
    response: Dict[str, Any],
    trace: Dict[str, Any],
    external_latency_ms: float,
) -> Dict[str, Any]:
    answer = str(response.get("response", ""))
    supporting = set(response.get("supporting_agents", []))
    expected_supporting = set(expected.get("expected_supporting_agents", []))
    knowledge_calls = _knowledge_trace(trace)

    checks: Dict[str, bool] = {
        "intent": response.get("intent") == expected["expected_intent"],
        "primary_agent": response.get("primary_agent") == expected["expected_primary_agent"],
        "supporting_agents": expected_supporting.issubset(supporting),
        "escalated": response.get("escalated") == expected["expected_escalated"],
        "no_runtime_failure": not bool(response.get("failure_kind")),
        "non_empty_response": bool(answer.strip()),
    }

    if "expected_knowledge_used" in expected:
        checks["knowledge_used"] = (
            response.get("knowledge_used") == expected["expected_knowledge_used"]
        )

    if "expected_evidence_found" in expected:
        evidence_values = [bool(item.get("evidence_found")) for item in knowledge_calls]
        expected_evidence = bool(expected["expected_evidence_found"])
        checks["evidence_found"] = bool(evidence_values) and (
            any(evidence_values) if expected_evidence else not any(evidence_values)
        )

    group_results = []
    for group in expected.get("must_contain_groups", []):
        group_results.append(any(str(term) in answer for term in group))
    checks["required_answer_terms"] = all(group_results)

    route_checks = ["intent", "primary_agent", "supporting_agents"]
    response_checks = [name for name in checks if name not in route_checks]
    route_pass = all(checks[name] for name in route_checks)
    response_contract_pass = all(checks[name] for name in response_checks)

    return {
        "message": expected["message"],
        "expected": {
            key: value
            for key, value in expected.items()
            if key != "message"
        },
        "actual": response,
        "trace": trace,
        "checks": checks,
        "route_pass": route_pass,
        "response_contract_pass": response_contract_pass,
        "e2e_pass": route_pass and response_contract_pass,
        "external_latency_ms": round(external_latency_ms, 1),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    count_true = lambda key: sum(bool(item[key]) for item in results)
    llm_components: Counter[str] = Counter()
    retrieval_modes: Counter[str] = Counter()
    evidence = Counter()
    stage_values: Dict[str, List[float]] = {}

    for item in results:
        actual = item["actual"]
        llm_components.update(actual.get("llm_calls_by_component", {}))
        for name, value in actual.get("stage_timings", {}).items():
            stage_values.setdefault(name, []).append(float(value))
        for call in _knowledge_trace(item["trace"]):
            retrieval_modes.update([str(call.get("retrieval_mode") or "unspecified")])
            if call.get("evidence_found") is True:
                evidence.update(["accepted"])
            elif call.get("evidence_found") is False:
                evidence.update(["rejected"])

    ratio = lambda numerator: round(numerator / total, 4) if total else 0.0
    intent_correct = sum(item["checks"]["intent"] for item in results)
    primary_correct = sum(item["checks"]["primary_agent"] for item in results)
    route_correct = count_true("route_pass")
    contract_correct = count_true("response_contract_pass")
    e2e_correct = count_true("e2e_pass")
    observed_llm_calls = [
        item["actual"].get("llm_call_count")
        for item in results
        if isinstance(item["actual"].get("llm_call_count"), int)
    ]
    llm_total = sum(observed_llm_calls)
    unknown_llm_requests = total - len(observed_llm_calls)

    return {
        "sample_size": total,
        "intent_accuracy": ratio(intent_correct),
        "primary_agent_accuracy": ratio(primary_correct),
        "route_accuracy": ratio(route_correct),
        "response_contract_pass_rate": ratio(contract_correct),
        "e2e_pass_rate": ratio(e2e_correct),
        "counts": {
            "intent_correct": intent_correct,
            "primary_agent_correct": primary_correct,
            "route_correct": route_correct,
            "response_contract_passed": contract_correct,
            "e2e_passed": e2e_correct,
        },
        "latency_ms": {
            "external": distribution(item["external_latency_ms"] for item in results),
            "api_total": distribution(
                item["actual"].get("stage_timings", {}).get("api_total_ms", 0.0)
                for item in results
                if "api_total_ms" in item["actual"].get("stage_timings", {})
            ),
            "by_stage": {
                name: distribution(values)
                for name, values in sorted(stage_values.items())
            },
        },
        "llm_calls": {
            "foreground_total": llm_total,
            "per_completed_request_mean": (
                round(llm_total / len(observed_llm_calls), 2)
                if observed_llm_calls else 0.0
            ),
            "requests_with_unknown_call_count": unknown_llm_requests,
            "by_component": dict(sorted(llm_components.items())),
            "scope_note": (
                "Counts application-level messages.create invocations completed before /chat returned. "
                "Background profile updates and provider SDK internal retries are excluded."
            ),
        },
        "retrieval": {
            "modes": dict(sorted(retrieval_modes.items())),
            "evidence": dict(sorted(evidence.items())),
        },
        "claim_boundary": (
            "Intent and route metrics use fixed human labels. Response metrics are deterministic contract checks, "
            "not a calibrated semantic-quality or production-accuracy estimate."
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    selected_cases = dataset["cases"]
    if args.case:
        requested = set(args.case)
        selected_cases = [case for case in selected_cases if case["id"] in requested]
        missing = requested - {case["id"] for case in selected_cases}
        if missing:
            raise ValueError(f"Unknown case id(s): {', '.join(sorted(missing))}")
    run_id = f"e2e-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results: List[Dict[str, Any]] = []

    def save_report(status: str) -> Dict[str, Any]:
        report = {
            "run_id": run_id,
            "generated_at": datetime.now().astimezone().isoformat(),
            "status": status,
            "base_url": args.base_url,
            "dataset": {
                "version": dataset.get("version"),
                "description": dataset.get("description"),
                "case_count": len(selected_cases),
                "request_count": len(results),
            },
            "summary": summarize(results),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    # The benchmark targets a loopback service.  Inheriting HTTP(S)_PROXY can
    # route 127.0.0.1 through a corporate proxy and produce a misleading 502.
    with httpx.Client(
        base_url=args.base_url,
        timeout=args.timeout,
        trust_env=False,
    ) as client:
        health = client.get("/health")
        health.raise_for_status()

        for case in selected_cases:
            conv_id = f"{run_id}-{case['id']}-{uuid.uuid4().hex[:6]}"
            user_id = f"benchmark-{run_id}"
            for turn_index, expected in enumerate(case["turns"]):
                started = time.perf_counter()
                try:
                    response = client.post(
                        "/chat",
                        json={
                            "message": expected["message"],
                            "user_id": user_id,
                            "conv_id": conv_id,
                        },
                    )
                except httpx.TimeoutException as ex:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    timeout_result = {
                        "case_id": case["id"],
                        "turn_index": turn_index,
                        "conv_id": conv_id,
                        "message": expected["message"],
                        "expected": {
                            key: value for key, value in expected.items() if key != "message"
                        },
                        "actual": {
                            "response": "",
                            "intent": None,
                            "primary_agent": None,
                            "supporting_agents": [],
                            "failure_kind": "client_timeout",
                            "stage_timings": {},
                            "llm_call_count": None,
                            "llm_calls_by_component": {},
                        },
                        "trace": {},
                        "checks": {
                            "intent": False,
                            "primary_agent": False,
                            "supporting_agents": False,
                            "escalated": False,
                            "no_runtime_failure": False,
                            "non_empty_response": False,
                            "required_answer_terms": False,
                        },
                        "route_pass": False,
                        "response_contract_pass": False,
                        "e2e_pass": False,
                        "external_latency_ms": round(elapsed_ms, 1),
                        "error": f"{type(ex).__name__}: exceeded {args.timeout:.1f}s",
                    }
                    results.append(timeout_result)
                    report = save_report("stopped_after_timeout")
                    print(
                        f"[{len(results):02d}] {case['id']}#{turn_index}: "
                        f"TIMEOUT after {elapsed_ms:.1f}ms; partial report saved",
                        flush=True,
                    )
                    return report
                elapsed_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()
                trace_response = client.get(f"/trace/tool/{payload['request_id']}")
                trace_response.raise_for_status()
                trace = trace_response.json().get("trace", {})
                evaluated = evaluate_turn(expected, payload, trace, elapsed_ms)
                evaluated.update({
                    "case_id": case["id"],
                    "turn_index": turn_index,
                    "conv_id": conv_id,
                })
                results.append(evaluated)
                save_report("running")
                print(
                    f"[{len(results):02d}] {case['id']}#{turn_index}: "
                    f"intent={payload.get('intent')} primary={payload.get('primary_agent')} "
                    f"pass={evaluated['e2e_pass']} latency={elapsed_ms:.1f}ms "
                    f"llm_calls={payload.get('llm_call_count', 0)}",
                    flush=True,
                )

    return save_report("completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("reports") / "e2e_latest.json",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--case",
        action="append",
        help="Run only the named case id; repeat this flag to select multiple cases.",
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
