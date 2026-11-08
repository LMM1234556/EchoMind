"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BGE_ZH_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTIONS = {
    DEFAULT_EMBEDDING_MODEL: "knowledge_base",
    BGE_ZH_EMBEDDING_MODEL: "knowledge_base_bge_v1",
}


class FastEmbedEmbeddingFunction:
    """把 FastEmbed 模型适配为 ChromaDB 0.5.x 的 EmbeddingFunction。"""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as ex:
            raise RuntimeError(
                "使用 BGE Embedding 需要安装 fastembed；请执行 "
                "python -m pip install -r requirements.txt"
            ) from ex

        options: Dict[str, Any] = {}
        if os.getenv("FASTEMBED_LOCAL_FILES_ONLY", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            options["local_files_only"] = True
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir or None,
            **options,
        )

    def __call__(self, input):
        # BGE v1.5 在 FastEmbed 0.9.0 中的 embed/query_embed/passage_embed
        # 输出一致；统一走 embed 可兼容 ChromaDB 0.5.x 的单入口协议。
        return list(self._model.embed(input))


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    默认使用 ChromaDB 内置的 all-MiniLM-L6-v2，也可以通过环境变量
    切换到 BAAI/bge-small-zh-v1.5。不同维度的模型使用不同 collection，
    避免覆盖旧向量并保留回滚能力。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    @staticmethod
    def _bypass_proxy_for_loopback(host: str) -> None:
        """避免 Windows 系统代理接管本机 ChromaDB 请求。"""
        if str(host).strip().lower() not in {"127.0.0.1", "localhost", "::1"}:
            return
        existing = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
        entries = [item.strip() for item in existing.split(",") if item.strip()]
        for item in ("127.0.0.1", "localhost", "::1"):
            if item not in entries:
                entries.append(item)
        value = ",".join(entries)
        os.environ["NO_PROXY"] = value
        os.environ["no_proxy"] = value

    @staticmethod
    def _build_embedding_function(model_name: Optional[str] = None):
        """按配置创建本地 Embedding 函数。

        ``chromadb-client`` 不会自动附加 Embedding 函数，因此 Windows
        本地连接 Docker ChromaDB 时也必须显式提供。
        """
        selected = (model_name or os.getenv("CHROMA_EMBEDDING_MODEL") or
                    DEFAULT_EMBEDDING_MODEL).strip()
        if selected == DEFAULT_EMBEDDING_MODEL:
            cache_path = os.getenv("CHROMA_EMBEDDING_CACHE", "").strip()
            if cache_path:
                ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(cache_path)
            return ONNXMiniLM_L6_V2()
        if selected == BGE_ZH_EMBEDDING_MODEL:
            cache_path = os.getenv("FASTEMBED_CACHE_PATH", "").strip()
            return FastEmbedEmbeddingFunction(selected, cache_dir=cache_path)
        raise ValueError(
            f"不支持的 CHROMA_EMBEDDING_MODEL={selected!r}；"
            f"当前支持: {', '.join(DEFAULT_COLLECTIONS)}"
        )

    @staticmethod
    def _default_collection_name(model_name: str) -> str:
        try:
            return DEFAULT_COLLECTIONS[model_name]
        except KeyError as ex:
            raise ValueError(f"没有为 Embedding 模型 {model_name!r} 配置 collection") from ex

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.embedding_model = (
            embedding_model or os.getenv("CHROMA_EMBEDDING_MODEL") or
            DEFAULT_EMBEDDING_MODEL
        ).strip()
        self.collection_name = (
            collection_name or os.getenv("CHROMA_COLLECTION_NAME") or
            self._default_collection_name(self.embedding_model)
        ).strip()
        if not self.collection_name:
            raise ValueError("CHROMA_COLLECTION_NAME 不能为空")

        # 优先连接独立 ChromaDB 服务。HttpClient 仍在本地计算 query embedding。
        self._use_server = False
        self._bypass_proxy_for_loopback(chroma_host)
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception as ex:
            if getattr(chromadb, "is_client", False):
                raise RuntimeError(
                    f"无法连接 ChromaDB 服务 {chroma_host}:{chroma_port}；"
                    "当前安装的是 chromadb-client，仅支持 HTTP 服务模式。"
                ) from ex
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # HttpClient 仍在客户端计算 query embedding；Windows 使用轻量
        # chromadb-client 时尤其需要显式绑定 Embedding 函数。
        self._embedding_function = self._build_embedding_function(self.embedding_model)
        collection_metadata: Dict[str, Any] = {
            "description": "EchoMind RAG 知识库",
            "embedding_model": self.embedding_model,
        }
        if self.embedding_model == BGE_ZH_EMBEDDING_MODEL:
            collection_metadata["hnsw:space"] = "cosine"
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=collection_metadata,
            embedding_function=self._embedding_function,
        )

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    @staticmethod
    def _lexical_units(text: str) -> set[str]:
        """提取英文/数字词和中文字符二元组，用于零模型调用的精确召回。"""
        normalized = (text or "").lower()
        units = set(re.findall(r"[a-z0-9]+", normalized))
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
        for run in chinese_runs:
            if len(run) == 1:
                units.add(run)
            else:
                units.update(run[index:index + 2] for index in range(len(run) - 1))
        return units

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """本地关键词检索；仅返回有实际词面交集的文档。"""
        query_units = self._lexical_units(query)
        if not query_units:
            return []

        ranked: List[tuple[int, int, Dict[str, Any]]] = []
        for index, item in enumerate(self.list_documents()):
            title_units = self._lexical_units(str(item.get("title", "")))
            content_units = self._lexical_units(str(item.get("content", "")))
            lexical_score = 3 * len(query_units & title_units) + len(query_units & content_units)
            if lexical_score <= 0:
                continue
            result = dict(item)
            result["score"] = float(lexical_score)
            result["retrieval"] = "keyword"
            ranked.append((lexical_score, -index, result))

        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        if not ranked:
            return []

        # 单个偶然二元词交集不足以支撑快速路径；同时剔除与首名
        # 差距过大的弱相关文档，不把“最不坏”的结果硬交给生成模型。
        confidence_floor = max(2.0, ranked[0][0] * 0.4)
        confident = [row[2] for row in ranked if row[0] >= confidence_floor]
        return confident[:max(1, top_k)]

    async def keyword_search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步关键词检索，避免 ChromaDB get 阻塞事件循环。"""
        return await asyncio.to_thread(self.keyword_search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    def list_documents(self) -> List[Dict[str, Any]]:
        """返回知识库中的全部文档片段，供离线评测和诊断使用。"""
        result = self._collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        items: List[Dict[str, Any]] = []
        for document, metadata in zip(documents, metadatas):
            metadata = metadata or {}
            items.append({
                "title": metadata.get("title", ""),
                "content": document or "",
                "chunk": metadata.get("chunk_index", 0),
            })
        return items

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        if params.get("mode") == "keyword":
            return await self.keyword_search_async(query, top_k=top_k)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
