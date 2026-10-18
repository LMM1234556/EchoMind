"""Consolidate a base E2E run with newer per-case retest reports."""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Tuple

from evaluation.e2e_benchmark import evaluate_turn, summarize


def _expectations(dataset: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for case in dataset["cases"]:
        for turn_index, turn in enumerate(case["turns"]):
            result[(case["id"], turn_index)] = turn
    return result


def _refresh_result(
    item: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    if item.get("actual", {}).get("failure_kind") == "client_timeout":
        return item
    refreshed = evaluate_turn(
        expected,
        item["actual"],
        item.get("trace", {}),
        item["external_latency_ms"],
    )
    refreshed.update({
        "case_id": item["case_id"],
        "turn_index": item["turn_index"],
        "conv_id": item.get("conv_id", ""),
    })
    return refreshed


def consolidate(
    dataset_path: pathlib.Path,
    base_path: pathlib.Path,
    override_paths: List[pathlib.Path],
) -> Dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    expectations = _expectations(dataset)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    indexed = {
        (item["case_id"], int(item["turn_index"])): item
        for item in base["results"]
    }

    for override_path in override_paths:
        override = json.loads(override_path.read_text(encoding="utf-8"))
        for item in override["results"]:
            indexed[(item["case_id"], int(item["turn_index"]))] = item

    missing = sorted(set(expectations) - set(indexed))
    if missing:
        raise ValueError(f"Missing result(s): {missing}")

    results = [
        _refresh_result(indexed[key], expectations[key])
        for key in expectations
    ]
    return {
        "run_id": f"consolidated-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "consolidated_completed_runs",
        "dataset": {
            "version": dataset.get("version"),
            "description": dataset.get("description"),
            "case_count": len(dataset["cases"]),
            "request_count": len(results),
        },
        "sources": [str(base_path), *[str(path) for path in override_paths]],
        "summary": summarize(results),
        "results": results,
    }


def markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    latency = summary["latency_ms"]["external"]
    calls = summary["llm_calls"]
    lines = [
        "# EchoMind 端到端试运行报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 固定场景：{report['dataset']['case_count']} 个，真实请求：{report['dataset']['request_count']} 次",
        f"- 意图准确率：{summary['intent_accuracy']:.1%}",
        f"- 主 Agent 准确率：{summary['primary_agent_accuracy']:.1%}",
        f"- 完整路由准确率：{summary['route_accuracy']:.1%}",
        f"- 回答契约通过率：{summary['response_contract_pass_rate']:.1%}",
        f"- 端到端通过率：{summary['e2e_pass_rate']:.1%}",
        f"- 外部延迟：P50 {latency['p50']:.1f} ms，P95 {latency['p95']:.1f} ms，最大 {latency['max']:.1f} ms",
        f"- 已观测前台应用层 LLM SDK 调用：{calls['foreground_total']} 次，完成请求平均 {calls['per_completed_request_mean']:.2f} 次",
        "",
        "## 场景明细",
        "",
        "| 场景 | 轮次 | 意图 | 主 Agent | 结果 | 延迟(ms) | LLM 调用 |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for item in report["results"]:
        actual = item["actual"]
        calls_value = actual.get("llm_call_count")
        calls_text = str(calls_value) if isinstance(calls_value, int) else "未知"
        lines.append(
            f"| {item['case_id']} | {item['turn_index']} | "
            f"{actual.get('intent') or '-'} | {actual.get('primary_agent') or '-'} | "
            f"{'通过' if item['e2e_pass'] else '失败'} | "
            f"{item['external_latency_ms']:.1f} | {calls_text} |"
        )
    lines.extend([
        "",
        "## 结论边界",
        "",
        "- 意图和路由指标来自固定人工标注。",
        "- 回答指标是确定性契约检查，不等于经过校准的语义质量或生产准确率。",
        "- 样本量只有 8 次请求；P50/P95 仅用于本轮试运行定位瓶颈，不能作为稳定 SLA。",
        "- LLM 调用数只统计 `/chat` 返回前完成的应用层 `messages.create` 调用，不包含异步用户画像更新和 SDK 内部自动重试。",
        "- 报告由首次完整运行与修复后的定向复测合并，来源文件保留在 JSON 的 `sources` 字段。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--base", type=pathlib.Path, required=True)
    parser.add_argument("--override", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    report = consolidate(args.dataset, args.base, args.override)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"json={args.output.resolve()}")
    print(f"markdown={args.output.with_suffix('.md').resolve()}")


if __name__ == "__main__":
    main()
