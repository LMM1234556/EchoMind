# Chinese Embedding Candidate Benchmark

Evaluation date: 2026-10-18  
Status: controlled runtime migration completed with rollback collection retained

## Motivation

The existing ChromaDB setup uses `all-MiniLM-L6-v2`, which performs poorly on some colloquial Chinese support queries. This benchmark compares a Chinese retrieval model without changing the running collection or calling a generation API.

## Setup

- Candidate: `BAAI/bge-small-zh-v1.5`
- Runtime: FastEmbed 0.9.0 with ONNX Runtime on CPU
- Embedding size: 512 dimensions
- Model cache size: approximately 90.8 MB
- Corpus: 6 built-in demo documents
- Positive cases: 12 seed queries and 12 separately authored colloquial challenge queries
- Negative cases: 8 out-of-domain queries

The candidate was evaluated with both content-only input and title-plus-content input. The two variants produced the same retrieval metrics, so the observed improvement was not caused by adding document titles.

## Results

| Model/input | Dataset | Hit@1 | Hit@3 | MRR |
|---|---|---:|---:|---:|
| all-MiniLM-L6-v2 | Seed | 0.3333 | 0.8333 | Not recorded |
| all-MiniLM-L6-v2 | Challenge | 0.4167 | 0.6667 | Not recorded |
| BGE-small-zh, content only | Seed | 0.9167 | 1.0000 | 0.9583 |
| BGE-small-zh, content only | Challenge | 1.0000 | 1.0000 | 1.0000 |
| BGE-small-zh, title + content | Seed | 0.9167 | 1.0000 | 0.9583 |
| BGE-small-zh, title + content | Challenge | 1.0000 | 1.0000 | 1.0000 |

For content-only BGE embeddings, the highest top-document similarity among the eight negative cases was 0.4247, while the lowest correct-document similarity among the 24 positive cases was 0.5092. This apparent separation is only descriptive for the small demo set and is not sufficient evidence for a production threshold.

The only seed query not ranked first was the two-factor-authentication query: `账户安全` ranked second behind `技术故障排查`. All twelve challenge queries ranked their expected document first.

## Decision boundary

- The candidate was migrated to the isolated `knowledge_base_bge_v1` collection.
- The original 384-dimensional `knowledge_base` collection remains intact for rollback; it must not be mixed with 512-dimensional BGE vectors.
- A fixed similarity threshold is deferred until the benchmark includes hard negatives, close-domain unanswered questions, and more confusable documents.
- These numbers describe a small regression dataset, not production accuracy.

## Runtime verification

- The BGE collection contains the 6 built-in demo documents.
- A refund timing query ranked `退款政策` first.
- In-container vector recall for four rewritten queries took approximately 80.7 ms in one trace.
- The full `/search` request took approximately 29.8 seconds because query rewriting and reranking each called the configured remote LLM. This single trace is diagnostic evidence, not a stable latency benchmark.
- The guarded chat retrieval path now tries one local BGE recall after a keyword miss when intent confidence is at least 0.8. A container check for the colloquial refund query completed in 25.7 ms, returned `fast_vector`, and made no rewrite or rerank call.
- Each retrieval branch is capped at five candidates. A fast-vector result is accepted only when its document title matches the audited intent-to-document mapping; otherwise the full RAG path remains available.
- Pure vector retrieval still returns a nearest document for every out-of-domain query in the negative set. Intent routing and the evidence gate therefore remain enabled.

## References

- [BAAI/bge-small-zh-v1.5 model card](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
