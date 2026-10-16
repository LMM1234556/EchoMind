"""Deterministic retrieval evaluation for the built-in knowledge base.

The default cases are a small seed/sanity set derived from the six demo
documents. They are useful for regression checks, but they are not a
representative production benchmark.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from mcp.knowledge_base import KnowledgeBase


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_title: Optional[str]


SEED_RETRIEVAL_CASES = [
    RetrievalCase("退款申请提交后多久审核", "退款政策"),
    RetrievalCase("审核通过后几天退回原支付账户", "退款政策"),
    RetrievalCase("订单已经发货为什么还没收到", "订单查询"),
    RetrievalCase("物流信息发货后多久更新", "订单查询"),
    RetrievalCase("忘记密码怎么重置", "账户安全"),
    RetrievalCase("怎么开启两步验证", "账户安全"),
    RetrievalCase("登录提示 401 怎么办", "技术故障排查"),
    RetrievalCase("应用总是崩溃如何处理", "技术故障排查"),
    RetrievalCase("一百积分能抵多少钱", "会员与积分"),
    RetrievalCase("金卡会员有什么折扣", "会员与积分"),
    RetrievalCase("标准配送需要几天", "配送说明"),
    RetrievalCase("加急配送运费是多少", "配送说明"),
]

# Created after the lexical scoring rule was frozen. These colloquial
# paraphrases intentionally reuse fewer exact phrases from the demo documents.
CHALLENGE_RETRIEVAL_CASES = [
    RetrievalCase("钱什么时候能原路退回来", "退款政策"),
    RetrievalCase("买错东西后想退掉", "退款政策"),
    RetrievalCase("包裹状态在哪里查", "订单查询"),
    RetrievalCase("订单显示寄出了但一直没收到", "订单查询"),
    RetrievalCase("账号好像被别人登过", "账户安全"),
    RetrievalCase("密码想不起来了", "账户安全"),
    RetrievalCase("打开软件就闪退", "技术故障排查"),
    RetrievalCase("网页报 500 应该怎么处理", "技术故障排查"),
    RetrievalCase("我的积分什么时候作废", "会员与积分"),
    RetrievalCase("银卡能便宜多少", "会员与积分"),
    RetrievalCase("偏远地区会晚几天", "配送说明"),
    RetrievalCase("想当天收到货要选什么方式", "配送说明"),
]

NEGATIVE_RETRIEVAL_CASES = [
    RetrievalCase("明天北京天气怎么样", None),
    RetrievalCase("帮我写一首关于秋天的诗", None),
    RetrievalCase("比特币现在多少钱", None),
    RetrievalCase("推荐一部科幻电影", None),
    RetrievalCase("红烧肉应该怎么做", None),
    RetrievalCase("法国的首都是哪里", None),
    RetrievalCase("计算 123 乘以 456", None),
    RetrievalCase("今天有什么新闻", None),
]

DEFAULT_RETRIEVAL_CASES = SEED_RETRIEVAL_CASES


def lexical_units(text: str) -> set[str]:
    """Extract deterministic ASCII tokens and Chinese character bigrams."""
    normalized = str(text or "").lower()
    units = set(re.findall(r"[a-z0-9]+", normalized))
    for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(segment) == 1:
            units.add(segment)
        else:
            units.update(segment[index:index + 2] for index in range(len(segment) - 1))
    return units


class LexicalRetriever:
    """Small deterministic keyword baseline; no model or external package."""

    def __init__(self, documents: List[Dict[str, Any]], title_weight: float = 3.0):
        self._documents = [dict(document) for document in documents]
        self._title_weight = title_weight
        self._features = [
            (lexical_units(document.get("title", "")), lexical_units(document.get("content", "")))
            for document in self._documents
        ]

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_units = lexical_units(query)
        scored = []
        for index, (document, (title_units, content_units)) in enumerate(
            zip(self._documents, self._features)
        ):
            score = (
                self._title_weight * len(query_units & title_units)
                + len(query_units & content_units)
            )
            scored.append((score, index, document))

        # A zero-overlap document is not a lexical match. Returning it would
        # turn collection insertion order into a fake relevance signal.
        ranked = [
            row for row in sorted(scored, key=lambda row: (-row[0], row[1]))
            if row[0] > 0
        ]
        return [
            {**document, "keyword_score": score}
            for score, _, document in ranked[:top_k]
        ]


class ReciprocalRankFusionRetriever:
    """Fuse vector and lexical rankings without mixing raw score scales."""

    def __init__(
        self,
        vector_search: Callable[[str, int], List[Dict[str, Any]]],
        lexical_search: Callable[[str, int], List[Dict[str, Any]]],
        candidate_k: int = 6,
        rrf_k: int = 60,
    ):
        self._vector_search = vector_search
        self._lexical_search = lexical_search
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    @staticmethod
    def _identity(item: Dict[str, Any]) -> str:
        identity = {
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "chunk": item.get("chunk", 0),
        }
        return json.dumps(identity, ensure_ascii=False, sort_keys=True)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        fetch_k = max(top_k, self._candidate_k)
        rankings = {
            "vector": self._vector_search(query, fetch_k),
            "keyword": self._lexical_search(query, fetch_k),
        }
        candidates: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}
        ranks: Dict[str, Dict[str, int]] = {}

        for source, items in rankings.items():
            for rank, item in enumerate(items, start=1):
                identity = self._identity(item)
                candidates.setdefault(identity, dict(item))
                scores[identity] = scores.get(identity, 0.0) + 1.0 / (self._rrf_k + rank)
                ranks.setdefault(identity, {})[source] = rank

        missing_rank = fetch_k + 1
        ordered = sorted(
            candidates,
            key=lambda identity: (
                -scores[identity],
                ranks[identity].get("vector", missing_rank)
                + ranks[identity].get("keyword", missing_rank),
                identity,
            ),
        )
        return [
            {
                **candidates[identity],
                "rrf_score": round(scores[identity], 8),
                "vector_rank": ranks[identity].get("vector"),
                "keyword_rank": ranks[identity].get("keyword"),
            }
            for identity in ordered[:top_k]
        ]


def evaluate_retriever(
    search: Callable[[str, int], List[Dict[str, Any]]],
    cases: List[RetrievalCase],
    retrieval_k: int = 6,
    dataset: str = "custom",
    dataset_scope: str = "caller-provided labelled cases",
) -> Dict[str, Any]:
    """Evaluate title retrieval with Hit@1, Hit@3 and MRR@K."""
    details: List[Dict[str, Any]] = []
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_total = 0.0
    total_latency_ms = 0.0
    positive_total = 0
    negative_total = 0
    false_positives = 0

    for case in cases:
        started_at = time.monotonic()
        results = search(case.query, retrieval_k)
        latency_ms = (time.monotonic() - started_at) * 1000
        total_latency_ms += latency_ms
        titles = [str(item.get("title", "")) for item in results]
        rank: Optional[int] = None
        if case.expected_title is None:
            negative_total += 1
            false_positives += int(bool(results))
        else:
            positive_total += 1
        if case.expected_title is not None and case.expected_title in titles:
            rank = titles.index(case.expected_title) + 1
            reciprocal_rank_total += 1.0 / rank
            hit_at_1 += int(rank == 1)
            hit_at_3 += int(rank <= 3)

        details.append({
            **asdict(case),
            "rank": rank,
            "top_titles": titles[:3],
            "accepted": bool(results),
            "latency_ms": round(latency_ms, 1),
        })

    total = len(cases)
    return {
        "dataset": dataset,
        "dataset_scope": dataset_scope,
        "total": total,
        "positive_total": positive_total,
        "negative_total": negative_total,
        "retrieval_k": retrieval_k,
        "hit_at_1": round(hit_at_1 / positive_total, 4) if positive_total else 0.0,
        "hit_at_3": round(hit_at_3 / positive_total, 4) if positive_total else 0.0,
        "mrr_at_k": round(reciprocal_rank_total / positive_total, 4) if positive_total else 0.0,
        "false_positive_rate": (
            round(false_positives / negative_total, 4) if negative_total else None
        ),
        "correct_rejection_rate": (
            round(1.0 - false_positives / negative_total, 4) if negative_total else None
        ),
        "avg_latency_ms": round(total_latency_ms / total, 1) if total else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--retrieval-k", type=int, default=6)
    args = parser.parse_args()

    knowledge_base = KnowledgeBase(chroma_host=args.host, chroma_port=args.port)
    documents = knowledge_base.list_documents()
    lexical_retriever = LexicalRetriever(documents)
    hybrid_retriever = ReciprocalRankFusionRetriever(
        knowledge_base.search,
        lexical_retriever.search,
        candidate_k=max(args.retrieval_k, len(documents)),
    )
    # Warm up the local embedding model so first-request initialization does not
    # dominate the vector latency comparison.
    knowledge_base.search("warmup", top_k=1)
    datasets = {
        "seed": (
            SEED_RETRIEVAL_CASES,
            "12 sanity cases derived from 6 demo documents",
        ),
        "challenge": (
            CHALLENGE_RETRIEVAL_CASES,
            "12 colloquial paraphrases created after freezing lexical scoring",
        ),
        "negative": (
            NEGATIVE_RETRIEVAL_CASES,
            "8 out-of-domain questions with no answer in the demo knowledge base",
        ),
    }
    reports = {}
    for dataset_name, (cases, scope) in datasets.items():
        reports[dataset_name] = {
            "vector": evaluate_retriever(
                knowledge_base.search,
                cases,
                retrieval_k=args.retrieval_k,
                dataset=dataset_name,
                dataset_scope=scope,
            ),
            "keyword": evaluate_retriever(
                lexical_retriever.search,
                cases,
                retrieval_k=args.retrieval_k,
                dataset=dataset_name,
                dataset_scope=scope,
            ),
            "hybrid_rrf": evaluate_retriever(
                hybrid_retriever.search,
                cases,
                retrieval_k=args.retrieval_k,
                dataset=dataset_name,
                dataset_scope=scope,
            ),
        }
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
