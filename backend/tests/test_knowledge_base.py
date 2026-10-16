from mcp.knowledge_base import KnowledgeBase


def make_knowledge_base(documents):
    knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
    knowledge_base.list_documents = lambda: documents
    return knowledge_base


def test_keyword_search_ranks_refund_document_first():
    knowledge_base = make_knowledge_base([
        {"title": "技术故障排查", "content": "登录失败或页面报错", "chunk": 0},
        {"title": "退款政策", "content": "退款申请在 1-3 个工作日审核，通过后 5-7 个工作日到账。", "chunk": 0},
        {"title": "配送说明", "content": "标准配送需要 3-5 个工作日。", "chunk": 0},
    ])

    results = knowledge_base.keyword_search("退款审核时效 退款到账时间", top_k=3)

    assert results[0]["title"] == "退款政策"
    assert results[0]["retrieval"] == "keyword"
    assert results[0]["score"] > 0
    assert len(results) == 1


def test_keyword_search_rejects_query_without_lexical_overlap():
    knowledge_base = make_knowledge_base([
        {"title": "退款政策", "content": "审核通过后原路退回。", "chunk": 0},
    ])

    assert knowledge_base.keyword_search("买错东西后想退掉", top_k=3) == []
