# JDDC 客服对话评估集扩展流程

目标：把现有 50 条评论评估集扩展为“评论 + 多轮客服对话”的 100 条混合评估集。

## 1. 获取 JDDC 数据

JDDC/JDDC 2.0 是中文电商客服对话数据集。公开论文说明该数据集需要通过官方渠道注册或申请授权后使用。

建议把授权下载后的文件放到：

```text
data/jddc/raw/
```

## 2. 抽取售后相关多轮对话

示例命令：

```powershell
python scripts\prepare_jddc_eval_candidates.py --input data/jddc/raw/jddc_train.jsonl --limit 50
```

脚本会自动：

- 读取 JSONL / JSON / CSV 格式
- 抽取多轮对话文本
- 过滤退款、退货、物流、客服、质量、投诉等售后相关对话
- 用当前 Agent 生成候选标签
- 输出待人工审核表

输出文件：

```text
data/eval/jddc_eval_candidates_50.csv
```

## 3. 人工审核

打开候选表，主要审核这几列：

```text
human_expected_categories
human_expected_risk_level
human_expected_priority
human_rule_reason
review_status
```

建议审核标准：

- 高风险：投诉、维权、退款纠纷、安全问题、假货、客服长期不处理
- 中风险：质量问题、核心功能异常、物流明显异常、客服体验差
- 低风险：已解决问题、轻微咨询、普通正向反馈、没有明确售后诉求

## 4. 合并成 100 条评估集

人工审核完成后运行：

```powershell
python scripts\merge_eval_sets.py
```

输出：

```text
data/eval/customer_service_eval_100.csv
```

## 5. 重新评估

```powershell
python scripts\evaluate_golden_set.py --input data/eval/customer_service_eval_100.csv --output data/eval/customer_service_eval_100_results.csv --errors-output data/eval/customer_service_eval_100_errors.csv --summary-output data/eval/customer_service_eval_100_summary.json
```

重点观察：

- 风险等级准确率
- 高风险召回率
- 风险低估率
- 问题分类命中率
- 多轮对话场景下的错误案例
