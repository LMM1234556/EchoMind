"""汇总小规模产品可用性测试的任务级指标。"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence


VALID_OUTCOMES = {"success", "partial", "failed", "not_attempted"}


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _valid_rating(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 1 <= float(value) <= 5


def analyse_product_validation(payload: Dict[str, Any]) -> Dict[str, Any]:
    sessions = payload.get("sessions") or []
    tasks: List[Dict[str, Any]] = []
    for session in sessions:
        participant_id = session.get("participant_id")
        for task in session.get("tasks") or []:
            outcome = task.get("outcome")
            if outcome not in VALID_OUTCOMES:
                raise ValueError(f"invalid outcome {outcome!r} for {task.get('scenario_id')}")
            if outcome == "not_attempted":
                continue
            duration = task.get("time_seconds")
            if duration is not None and (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or float(duration) < 0
            ):
                raise ValueError(f"invalid time_seconds {duration!r} for {task.get('scenario_id')}")
            for field in ("trust_rating", "evidence_clarity_rating", "ease_rating"):
                if task.get(field) is not None and not _valid_rating(task.get(field)):
                    raise ValueError(f"invalid {field} {task.get(field)!r} for {task.get('scenario_id')}")
            assistance_count = task.get("assistance_count", 0)
            if (
                not isinstance(assistance_count, int)
                or isinstance(assistance_count, bool)
                or assistance_count < 0
            ):
                raise ValueError(
                    f"invalid assistance_count {assistance_count!r} for {task.get('scenario_id')}"
                )
            item = dict(task)
            item["participant_id"] = participant_id
            tasks.append(item)

    successful = [task for task in tasks if task["outcome"] == "success"]
    partial = [task for task in tasks if task["outcome"] == "partial"]
    failed = [task for task in tasks if task["outcome"] == "failed"]
    durations = [float(task["time_seconds"]) for task in tasks if isinstance(task.get("time_seconds"), (int, float))]
    successful_durations = [
        float(task["time_seconds"])
        for task in successful
        if isinstance(task.get("time_seconds"), (int, float))
    ]
    trust = [float(task["trust_rating"]) for task in tasks if _valid_rating(task.get("trust_rating"))]
    evidence_clarity = [
        float(task["evidence_clarity_rating"])
        for task in tasks
        if _valid_rating(task.get("evidence_clarity_rating"))
    ]
    ease = [float(task["ease_rating"]) for task in tasks if _valid_rating(task.get("ease_rating"))]
    critical_errors = sum(bool(task.get("critical_error")) for task in tasks)
    assisted = sum(int(task.get("assistance_count", 0)) > 0 for task in tasks)
    timed_out = sum(bool(task.get("timed_out")) for task in tasks)
    participant_ids = {str(session.get("participant_id")) for session in sessions if session.get("participant_id")}

    scenario_rows = []
    scenarios = {str(item.get("id")): item for item in payload.get("scenarios") or []}
    for scenario_id in sorted({str(task.get("scenario_id")) for task in tasks}):
        subset = [task for task in tasks if str(task.get("scenario_id")) == scenario_id]
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": (scenarios.get(scenario_id) or {}).get("name"),
                "attempted": len(subset),
                "completed": sum(task["outcome"] == "success" for task in subset),
                "completion_rate": _round(
                    sum(task["outcome"] == "success" for task in subset) / len(subset)
                ),
                "median_time_seconds": _round(
                    _median([
                        float(task["time_seconds"])
                        for task in subset
                        if isinstance(task.get("time_seconds"), (int, float))
                    ])
                ),
            }
        )

    minimum_participants = int((payload.get("metadata") or {}).get("minimum_participants", 5))
    status = "awaiting_sessions" if not tasks else (
        "formative_round_complete" if len(participant_ids) >= minimum_participants else "collecting_formative_sessions"
    )
    return {
        "name": "product-validation-summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "sample": {
            "participants": len(participant_ids),
            "sessions": len(sessions),
            "attempted_tasks": len(tasks),
            "minimum_participants": minimum_participants,
        },
        "primary_kpis": {
            "strict_task_completion_rate": _round(len(successful) / len(tasks) if tasks else None),
            "median_successful_task_time_seconds": _round(_median(successful_durations)),
            "mean_trust_rating_1_to_5": _round(_mean(trust)),
        },
        "diagnostics": {
            "partial_completion_rate": _round(len(partial) / len(tasks) if tasks else None),
            "failed_task_rate": _round(len(failed) / len(tasks) if tasks else None),
            "all_task_time_p50_seconds": _round(_median(durations)),
            "all_task_time_p95_seconds": _round(_percentile(durations, 0.95)),
            "mean_evidence_clarity_rating_1_to_5": _round(_mean(evidence_clarity)),
            "mean_ease_rating_1_to_5": _round(_mean(ease)),
            "assisted_task_rate": _round(assisted / len(tasks) if tasks else None),
        },
        "guardrails": {
            "critical_error_rate": _round(critical_errors / len(tasks) if tasks else None),
            "timeout_rate": _round(timed_out / len(tasks) if tasks else None),
        },
        "by_scenario": scenario_rows,
        "interpretation": (
            "These are formative usability metrics for the recorded participants and tasks. "
            "They are not production adoption or customer satisfaction metrics."
        ),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    sample = report["sample"]
    kpis = report["primary_kpis"]
    lines = [
        "# EchoMind 产品验证汇总",
        "",
        f"- 状态：`{report['status']}`",
        f"- 参与者：{sample['participants']}；任务尝试：{sample['attempted_tasks']}",
        "",
        "## 核心指标",
        "",
        f"- 严格任务完成率：{kpis['strict_task_completion_rate']}",
        f"- 成功任务用时中位数：{kpis['median_successful_task_time_seconds']} 秒",
        f"- 平均可信度评分：{kpis['mean_trust_rating_1_to_5']} / 5",
        "",
        "## 分场景",
        "",
        "| 场景 | 完成数 / 尝试数 | 完成率 | 用时中位数（秒） |",
        "|---|---:|---:|---:|",
    ]
    for row in report["by_scenario"]:
        lines.append(
            f"| {row['scenario_name'] or row['scenario_id']} | {row['completed']} / {row['attempted']} | "
            f"{row['completion_rate']} | {row['median_time_seconds']} |"
        )
    lines.extend(
        [
            "",
            "> 本报告描述当前参与者与任务范围内的形成性可用性结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise product-validation sessions")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = analyse_product_validation(payload)
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
