from evaluation.e2e_benchmark import distribution, evaluate_turn, percentile, summarize


def test_percentile_and_distribution_are_stable_for_small_samples():
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20], 0.95) == 19.5
    assert distribution([10, 20])["p95"] == 19.5


def test_evaluate_turn_separates_route_and_response_contracts():
    expected = {
        "message": "退款多久到账？",
        "expected_intent": "refund",
        "expected_primary_agent": "billing",
        "expected_knowledge_used": True,
        "expected_evidence_found": True,
        "expected_escalated": False,
        "must_contain_groups": [["5-7"]],
    }
    response = {
        "response": "审核通过后 5-7 个工作日到账。",
        "intent": "refund",
        "primary_agent": "billing",
        "supporting_agents": [],
        "knowledge_used": True,
        "escalated": False,
        "failure_kind": "",
        "stage_timings": {"api_total_ms": 100},
        "llm_call_count": 2,
        "llm_calls_by_component": {"intent_recognizer": 1, "agent.billing": 1},
    }
    trace = {
        "tool_calls": [
            {
                "tool_name": "search_knowledge_base",
                "evidence_found": True,
                "retrieval_mode": "fast_keyword",
            }
        ]
    }
    result = evaluate_turn(expected, response, trace, 110)
    assert result["route_pass"] is True
    assert result["response_contract_pass"] is True
    assert result["e2e_pass"] is True

    summary = summarize([result])
    assert summary["intent_accuracy"] == 1.0
    assert summary["llm_calls"]["foreground_total"] == 2
    assert summary["retrieval"]["modes"] == {"fast_keyword": 1}
