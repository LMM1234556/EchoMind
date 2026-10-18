from pathlib import Path

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from mcp.knowledge_base import (
    BGE_ZH_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    FastEmbedEmbeddingFunction,
    KnowledgeBase,
)


def make_knowledge_base(documents):
    knowledge_base = KnowledgeBase.__new__(KnowledgeBase)
    knowledge_base.list_documents = lambda: documents
    return knowledge_base


def test_embedding_cache_can_be_redirected_outside_system_drive(monkeypatch, tmp_path):
    cache_path = tmp_path / "chroma-model"
    original_path = ONNXMiniLM_L6_V2.DOWNLOAD_PATH
    monkeypatch.setattr(ONNXMiniLM_L6_V2, "DOWNLOAD_PATH", original_path)
    monkeypatch.setenv("CHROMA_EMBEDDING_CACHE", str(cache_path))

    embedding_function = KnowledgeBase._build_embedding_function()

    assert isinstance(embedding_function, ONNXMiniLM_L6_V2)
    assert ONNXMiniLM_L6_V2.DOWNLOAD_PATH == Path(cache_path)


def test_embedding_models_use_separate_default_collections():
    assert KnowledgeBase._default_collection_name(DEFAULT_EMBEDDING_MODEL) == "knowledge_base"
    assert KnowledgeBase._default_collection_name(BGE_ZH_EMBEDDING_MODEL) == "knowledge_base_bge_v1"


def test_fastembed_adapter_uses_configured_e_drive_cache(monkeypatch, tmp_path):
    captured = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def embed(self, values):
            return ([float(index), 1.0] for index, _ in enumerate(values))

    import sys
    import types
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    cache_path = tmp_path / "fastembed"

    function = FastEmbedEmbeddingFunction(BGE_ZH_EMBEDDING_MODEL, str(cache_path))

    assert captured["model_name"] == BGE_ZH_EMBEDDING_MODEL
    assert captured["cache_dir"] == str(cache_path)
    assert function(["查询", "文档"]) == [[0.0, 1.0], [1.0, 1.0]]


def test_unknown_embedding_model_is_rejected():
    try:
        KnowledgeBase._build_embedding_function("unknown/model")
    except ValueError as error:
        assert "不支持" in str(error)
    else:
        raise AssertionError("unknown model should be rejected")


def test_loopback_chroma_bypasses_system_proxy(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    KnowledgeBase._bypass_proxy_for_loopback("127.0.0.1")

    assert set(__import__("os").environ["NO_PROXY"].split(",")) == {
        "127.0.0.1", "localhost", "::1"
    }


def test_remote_chroma_does_not_change_proxy_bypass(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "internal.example")

    KnowledgeBase._bypass_proxy_for_loopback("chroma.internal")

    assert __import__("os").environ["NO_PROXY"] == "internal.example"


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
