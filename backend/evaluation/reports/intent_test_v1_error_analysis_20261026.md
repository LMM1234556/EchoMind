# Intent Test v1 错误归因

## 结论

首次冻结测试的 5 条错误可分为两种机制：4 条 LLM 类别边界错误，1 条 Pattern 否定语义错误。不能把它们笼统归因于“模型不够强”。

## 1. 通用 `query` 过度吸收

`billing` 两条、`logistics` 一条和 `other` 一条被预测为 `query`。这四条的 LLM 置信度均为 0.90 或 0.95，表明主要错误来自 LLM 判断，而不是低置信度融合。

当前业务定义已经写明 `query` 不应覆盖订单、物流等细粒度类别，但通用类 Few-shot 存在结构性冲突：

- `query` 模板包含订单状态、重置密码和物流问题；
- `billing` 模板包含重复扣款、退款和发票；
- `technical` 模板包含崩溃、登录和 500 错误。

这些示例分别应归入 `order_status` / `account_security` / `logistics`、`payment_issue` / `refund` / `invoice`、`technical_crash` / `technical_login`。定义与示例冲突会让 LLM 学到“通用类可以吞掉细分类”的错误边界。

## 2. Pattern 不理解否定语义

`test-technical-004` 的 LLM reasoning 明确判断为一般技术问题，并说明用户排除了服务崩溃；但关键词分支看到“崩溃”后给出 `technical_crash`，融合器又用细粒度 Pattern 覆盖了通用技术判断。该错误来自否定词作用域缺失，而不是 LLM 判断错误。

## 3. 合规的优化路径

1. 用新的开发样本确认通用类与细粒度类边界，不复制冻结测试文本。
2. 在开发集上替换互相矛盾的通用类 Few-shot。
3. 为“没有、并未、不存在、不会”等否定表达补充 Pattern 保护和单元测试。
4. 只在开发集上迭代。冻结测试 v1 已经揭示，后续仅作回归参考；新的独立指标必须来自测试集 v2。
