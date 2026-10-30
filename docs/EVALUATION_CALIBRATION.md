# LLM-as-Judge 人工校准方案

## 当前结论

EchoMind 已具备四维 LLM-as-Judge 评分框架，也完成了 24 条 Grounded Chat 回答的人工事实忠实度复核。二者衡量的对象不同：现有人工复核给出“事实是否忠于冻结证据”的二元结论；Judge 输出相关性、准确性、完整性和有用性四个连续分数。

本轮新增同一批回答的四维人工盲评、Judge 评分和独立验证集分析。在配对评分完成前，项目继续把 LLM-as-Judge 定位为评测工具框架，不报告其校准准确率。

## 样本与切分

- 样本来源：`grounded_chat_test_v2_first_run.json` 中的 24 条真实模型回答；
- 16 条用于拟合每个维度的仿射校准参数；
- 8 条作为独立验证集，用于比较原始 Judge 与校准后 Judge；
- 由两位评分者独立完成 1–5 分评分，评分完成后再生成 Judge 分数；
- 校准表不展示既有事实复核 verdict，降低锚定效应。

24 条样本适合形成第一轮校准基线，独立验证集只有 8 条，因此结果用于发现系统性偏高、偏低和维度错位。后续扩大到至少 60 条并覆盖更多回答类型后，再评估更稳定的泛化表现。

## 人工评分量表

所有维度使用 1–5 分，并在分析时线性转换到 0–1。

| 维度 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| 相关性 | 与问题无关 | 回答主要问题但有明显偏离 | 直接回应用户核心诉求 |
| 准确性 | 存在关键事实错误或无依据承诺 | 核心事实基本正确，仍有需核验内容 | 所有关键事实均与冻结依据一致 |
| 完整性 | 核心诉求基本未处理 | 处理主要诉求，遗漏部分必要信息 | 覆盖全部核心诉求与必要边界 |
| 有用性 | 用户无法据此行动 | 给出方向但下一步不够明确 | 用户可以据此完成下一步操作 |

评分者在 `notes` 中记录关键依据。涉及知识缺口时，准确说明“当前依据不足”可以获得高准确性分；流畅但扩展证据外事实的回答应降低准确性分。

## 主要指标

1. **独立验证集 MAE**：Judge 与人工平均分的绝对误差，按四个维度分别报告；
2. **Spearman 相关系数**：判断 Judge 是否能正确排序回答质量；
3. **通过判断一致率与 Cohen's kappa**：将四维平均分达到 0.70 视为通过，衡量决策一致性；
4. **评分者间差异**：先确认人工评分本身是否稳定，再解释 Judge 偏差；
5. **原始与校准后对照**：仿射参数仅在 16 条校准集拟合，效果只在 8 条验证集判断。

第一轮以形成基线为目标。验证集 MAE、kappa 和评分者差异会决定后续动作，而不是预先选择一个有利阈值。

## 执行流程

在 `backend` 目录执行。

### 1. 生成盲评工作表

```powershell
python -m evaluation.prepare_judge_calibration `
  evaluation/reports/grounded_chat_test_v2_first_run.json `
  evaluation/data/judge_calibration_v1.json
```

生成结果已经预留两位评分者。每位评分者独立填写四个维度的 1–5 分和 `notes`。

### 2. 生成 Judge 分数

该步骤会产生最多 24 次真实模型调用。确认 API 额度后执行：

```powershell
python -m evaluation.run_judge_calibration `
  evaluation/data/judge_calibration_v1.json `
  evaluation/data/judge_calibration_v1_scored.json `
  --max-cases 24 --execute
```

### 3. 输出校准报告

```powershell
python -m evaluation.judge_calibration `
  evaluation/data/judge_calibration_v1_scored.json `
  --json-output evaluation/reports/judge_calibration_v1.json `
  --markdown-output evaluation/reports/judge_calibration_v1.md
```

## 结果使用方式

- 验证集 MAE 下降且通过判断一致性提升：采用校准参数，并用新样本持续复核；
- 排序相关较高但绝对分数偏移：保留 Judge 排序能力，同时调整评分映射；
- 人工评分者差异较大：先修订量表与示例，再重新评分；
- Judge 与人工在准确性维度分歧明显：优先强化证据上下文和准确性 rubric。

校准结果描述的是当前模型、当前 rubric 和当前 24 条样本范围内的一致性。模型、Prompt 或评分标准变化后，需要重新执行校准。
