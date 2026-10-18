# Embedding 模型配置与回滚

EchoMind 支持两套本地知识库 Embedding 配置：

| 模型 | 维度 | 默认 collection |
|---|---:|---|
| `all-MiniLM-L6-v2` | 384 | `knowledge_base` |
| `BAAI/bge-small-zh-v1.5` | 512 | `knowledge_base_bge_v1` |

不同维度的向量不能写入同一个 collection。切换模型时必须同步切换 collection 名称。

## 使用中文 BGE

在 `.env` 中配置：

```dotenv
CHROMA_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CHROMA_COLLECTION_NAME=knowledge_base_bge_v1
FASTEMBED_CACHE_PATH=/app/data/model-cache/fastembed
```

Docker Compose 将宿主机的 `backend/data/model-cache` 挂载到容器中的 `/app/data/model-cache`。该目录被 Git 和 Docker 构建上下文忽略，避免提交模型文件或将其重复复制进镜像。

## 回滚到原模型

将 `.env` 改为：

```dotenv
CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_COLLECTION_NAME=knowledge_base
```

然后重新创建应用容器：

```powershell
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --force-recreate echomind
```

原 `knowledge_base` collection 在迁移期间不会删除或覆盖，因此回滚不需要重新生成旧向量。

## 验证边界

本地小型回归集表明 BGE 更适合当前中文演示语料，但不代表生产准确率。向量检索始终会返回最近文档，因此不能仅靠 Top-K 结果判断“知识库中确实存在答案”；意图路由、词面召回和证据门控仍需保留。
