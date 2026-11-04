# 仓库交付验证记录

验证日期：2026-11-04

## 版本与范围

后端业务代码基于提交 `29cbc11`；仓库整理提交 `131a768` 的全新检出再次通过全量回归，不包含本地尚未完成的产品验证脚本改动。此次整理集中于文档状态、评测入口、配置说明和应用容器健康检查，Agent、RAG 与 `MCPToolManager` 的调用逻辑保持原有实现。

| 检查 | 结果 | 范围 |
|---|---|---|
| 后端全量自动化回归 | 163/163 通过，最终检出耗时 10.63 秒 | Windows，Python 3.12.10，使用现有依赖环境和独立检出源码 |
| 前端生产构建 | 通过 | Node.js 24.19.0，Vite 7.3.5，使用现有 node_modules，离线构建 |
| 干净示例配置解析 | 通过 | 在独立副本将 `.env.example` 复制为 `.env`，执行 `docker compose config --quiet` |
| Compose 配置语义 | 通过 | Qwen 模型配置、容器内 ChromaDB 地址、Python 健康检查与示例空密钥 |
| 全新依赖安装 | 待验证 | 本轮复用已安装依赖；未重新下载环境和模型 |
| 新容器构建及启动 | 待验证 | 验证时 Docker 引擎未运行，配置解析未启动容器 |
| 本轮外部模型调用 | 0 | 本轮整理与回归复用既有评测记录 |

## 冻结文件换行一致性

首次在 Windows 隔离检出回归时，161 项通过、2 项失败；失败源于 Git 自动换行转换，使部分冻结数据从 LF 转为 CRLF，导致字节 SHA-256 不匹配。

对冻结清单中的文件逐项核对后，在 `.gitattributes` 中明确记录对应换行格式：数据集保持 LF，清单中原始字节为 CRLF 的报告保持 CRLF。再次创建独立检出副本后，163 项全部通过。冻结问题、标签、历史报告内容及冻结清单中的哈希保持不变。

Judge 校准输入 `judge_calibration_v1_ready.json` 同样固定为 LF，与已保存评分中的输入 SHA-256 一致，保持后续检查与断点识别可追溯。

## 启动配置说明

- 普通 Docker Compose 读取 `backend/.env`；
- 主机直接运行 Python 评测时使用 `backend/.env.local`，并配置宿主机 Redis/ChromaDB 地址；
- 示例配置中的 API Key 保持为空，运行者在本地填写自己的密钥；
- 应用容器健康检查使用 Python 标准库，与 Dockerfile 一致，避免依赖镜像中未安装的 curl；
- 本轮确认配置可解析；完整新容器启动状态保留为待验证。

## 评测与展示入口

- [正式检索对照](../backend/evaluation/reports/retrieval_test_v1_first_run_20261026.md)：60 条冻结查询，Hit@1/Hit@3 统计其中 36 条有答案查询；
- [当前完整链路基线](../backend/evaluation/reports/e2e_current_top3_20261026.md)：7 个场景、8 次真实请求，包含多轮会话；
- [Judge 校准结果](../backend/evaluation/reports/judge_calibration_v1_findings_20261104.md)：24 条配对评分及 16/8 切分结果；
- [早期 Embedding 探索](EMBEDDING_BENCHMARK.md)：作为开发记录保留；
- [演示视频](assets/portfolio/echomind-demo.mp4)：保留既有入口；
- [产品验证方案](PRODUCT_VALIDATION_PLAN.md)：实际参与者数据等待采集。

163 项回归结果描述该版本的自动化检查表现；真实模型效果、产品可用性和新环境启动分别由对应记录说明。
