# Intent Test v1 首次冻结评测

## 评测身份

- 日期：2026-10-26
- 待测基线提交：`3aeca2011262594b1a6b8f774795909f3df9dbdf`
- 冻结数据：`backend/evaluation/data/intent_test_v1.json`
- 数据 SHA-256：`13a83f914caac6a2faae530b9116eaa8bfe9df794edb7fc46b2d41dfec19803b`
- 模型：`qwen3.7-plus`
- 样本来源：AI 起草、项目负责人逐条人工复核的合成测试集
- 样本规模：95 条，19 个意图，每类 5 条
- API 请求：95/95 成功，供应商失败 0，未使用降级结果计分

## 首次盲测结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 94.74%（90/95） |
| Macro-F1 | 94.95% |
| Accuracy Wilson 95% CI | 88.27%–97.73% |
| 平均延迟 | 5413.5 ms |
| P50 延迟 | 4625 ms |
| P95 延迟 | 10234 ms |
| 最大延迟 | 15281 ms |

## 错分记录

| ID | 消息 | 期望 | 预测 |
|---|---|---|---|
| `test-technical-004` | 调用接口偶尔超时但服务没有崩溃 | `technical` | `technical_crash` |
| `test-billing-002` | 套餐费用是按月还是按年计算 | `billing` | `query` |
| `test-billing-005` | 试用期结束后的计费标准是什么 | `billing` | `query` |
| `test-logistics-004` | 发货后还能改收货地址吗 | `logistics` | `query` |
| `test-other-004` | 一百美元大概能换多少人民币 | `other` | `query` |

## 主要混淆

- `query`：Recall 1.00，但 Precision 0.56，说明通用查询类别吸收了多个细粒度问题。
- `billing`：Recall 0.60，两个计费规则问题被识别为通用 `query`。
- `technical`、`logistics`、`other`：Recall 均为 0.80，各有一条流向相邻或通用类别。
- 其余 13 个类别在本次每类 5 条样本上均全部正确；类别级结果用于识别本批样本的混淆方向，真实业务表现通过实际采样评估。

## 解释边界

本结果为 95 条受控合成样本的首次冻结评测，每类 5 条，类别级指标用于发现混淆方向。跨实验比较需对齐来源、标签体系和切分；置信区间用于描述估计精度，线上客服效果按实际流量独立评估。

首次结果揭示的 5 条错例作为固定回归记录保留。后续提示词、Few-shot、Pattern 和阈值优化使用新开发样本，独立效果由首次预测前冻结的测试集 v2 衡量。
