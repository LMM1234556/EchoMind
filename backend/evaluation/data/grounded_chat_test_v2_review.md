# Grounded Chat Test v2 人工审核单

状态：24 条已完成人工审核；正式数据集已冻结为 `grounded_chat_test_v2.json`，尚未运行模型评测。

审核时只依据 `retrieval_corpus_v1.json` 中的 6 篇文档，不参考模型预测。每条需要确认：

1. 问题表达自然且含义唯一；
2. `expected_supported` 是否正确；
3. 有证据时，`expected_title` 是否是能够完整回答问题的一篇文档；
4. `expected_answer_facts` 是否只包含文档明确事实或一步确定性计算；
5. `forbidden_claims` 是否确实不应出现在回答中；
6. 是否与开发集或 v1 测试问题过于相似。

## 分批确认范围

- 第 1 批：1–8，退款与订单（已确认）
- 第 2 批：9–16，账户与技术（已确认）
- 第 3 批：17–24，会员与配送（已确认）

只有 24 条全部由项目负责人确认后，才能把 `review.status` 改为 `verified`、计算 SHA-256 并生成冻结清单。冻结后不得根据首次结果修改提示词、知识库、意图规则或阈值；任何改进都必须使用新的开发样本，并另建 v3 测试集验证。

冻结结果：24/24 已确认，12 条有证据、12 条知识缺口。正式数据集 SHA-256 为 `b90847bd1ac6dff852a930ebc9c1ab6eb9d2a7a6ef4a331587db3415fc7b6c13`，冻结清单见 `grounded_chat_test_v2.freeze.json`。
