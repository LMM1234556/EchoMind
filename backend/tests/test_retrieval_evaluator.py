from evaluation.retrieval_evaluator import (
    CHALLENGE_RETRIEVAL_CASES,
    LexicalRetriever,
    NEGATIVE_RETRIEVAL_CASES,
    ReciprocalRankFusionRetriever,
    RetrievalCase,
    evaluate_retriever,
    lexical_units,
)


def test_retrieval_metrics_use_one_based_rank_and_top_three():
    cases = [
        RetrievalCase("q1", "A"),
        RetrievalCase("q2", "B"),
        RetrievalCase("q3", "C"),
        RetrievalCase("q4", "D"),
    ]
    rankings = {
        "q1": ["A", "B", "C", "D"],
        "q2": ["A", "B", "C", "D"],
        "q3": ["A", "B", "C", "D"],
        "q4": ["A", "B", "C"],
    }

    def fake_search(query, top_k):
        return [{"title": title} for title in rankings[query][:top_k]]

    report = evaluate_retriever(fake_search, cases, retrieval_k=4)

    assert report["total"] == 4
    assert report["hit_at_1"] == 0.25
    assert report["hit_at_3"] == 0.75
    assert report["mrr_at_k"] == 0.4583
    assert [detail["rank"] for detail in report["details"]] == [1, 2, 3, None]


def test_empty_retrieval_dataset_has_zero_metrics():
    report = evaluate_retriever(lambda query, top_k: [], [], retrieval_k=6)

    assert report["total"] == 0
    assert report["hit_at_1"] == 0.0
    assert report["hit_at_3"] == 0.0
    assert report["mrr_at_k"] == 0.0
    assert report["false_positive_rate"] is None
    assert report["avg_latency_ms"] == 0.0


def test_lexical_units_extract_chinese_bigrams_and_ascii_tokens():
    assert lexical_units("登录提示 401 怎么办") == {
        "登录", "录提", "提示", "怎么", "么办", "401"
    }


def test_lexical_retriever_prefers_title_and_content_overlap():
    retriever = LexicalRetriever([
        {"title": "退款政策", "content": "退款申请提交后进行审核"},
        {"title": "账户安全", "content": "忘记密码后可以重置"},
        {"title": "配送说明", "content": "标准配送需要三天"},
    ])

    results = retriever.search("退款审核需要多久", top_k=2)

    assert results[0]["title"] == "退款政策"
    assert results[0]["keyword_score"] > 0


def test_lexical_retriever_does_not_return_zero_overlap_documents():
    retriever = LexicalRetriever([
        {"title": "退款政策", "content": "退款申请提交后进行审核"},
        {"title": "账户安全", "content": "忘记密码后可以重置"},
    ])

    assert retriever.search("完全无关的句子", top_k=2) == []


def test_challenge_set_is_separate_and_covers_all_demo_titles():
    assert len(CHALLENGE_RETRIEVAL_CASES) == 12
    assert {case.expected_title for case in CHALLENGE_RETRIEVAL_CASES} == {
        "退款政策", "订单查询", "账户安全", "技术故障排查", "会员与积分", "配送说明"
    }


def test_rrf_fuses_ranks_without_mixing_raw_scores():
    documents = {
        "A": {"title": "A", "content": "a", "chunk": 0, "score": -0.9},
        "B": {"title": "B", "content": "b", "chunk": 0, "score": 100.0},
        "C": {"title": "C", "content": "c", "chunk": 0, "score": 0.1},
    }

    def vector_search(query, top_k):
        return [documents[name] for name in ["A", "B", "C"][:top_k]]

    def keyword_search(query, top_k):
        return [documents[name] for name in ["B", "C", "A"][:top_k]]

    retriever = ReciprocalRankFusionRetriever(vector_search, keyword_search, candidate_k=3)
    results = retriever.search("query", top_k=3)

    assert [item["title"] for item in results] == ["B", "A", "C"]
    assert results[0]["vector_rank"] == 2
    assert results[0]["keyword_rank"] == 1


def test_rrf_falls_back_to_vector_order_when_keyword_has_no_match():
    vector_results = [
        {"title": "A", "content": "a", "chunk": 0},
        {"title": "B", "content": "b", "chunk": 0},
    ]
    retriever = ReciprocalRankFusionRetriever(
        lambda query, top_k: vector_results[:top_k],
        lambda query, top_k: [],
        candidate_k=2,
    )

    assert [item["title"] for item in retriever.search("query", 2)] == ["A", "B"]


def test_negative_cases_measure_false_positive_and_rejection_rates():
    cases = [
        RetrievalCase("unrelated accepted", None),
        RetrievalCase("unrelated rejected", None),
    ]

    def fake_search(query, top_k):
        return [{"title": "A"}] if query.endswith("accepted") else []

    report = evaluate_retriever(fake_search, cases)

    assert report["positive_total"] == 0
    assert report["negative_total"] == 2
    assert report["false_positive_rate"] == 0.5
    assert report["correct_rejection_rate"] == 0.5


def test_negative_set_contains_only_no_answer_cases():
    assert len(NEGATIVE_RETRIEVAL_CASES) == 8
    assert all(case.expected_title is None for case in NEGATIVE_RETRIEVAL_CASES)
