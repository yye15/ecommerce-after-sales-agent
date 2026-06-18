# JDDC 数据放置说明

JDDC/JDDC 2.0 是京东客服对话数据集，适合用于本项目的多轮售后对话评估。

请在获得官方授权后，把原始数据文件放到本目录，例如：

```text
data/jddc/raw/jddc_train.jsonl
data/jddc/raw/jddc_dialogues.json
data/jddc/raw/jddc_dialogues.csv
```

然后运行：

```powershell
python scripts\prepare_jddc_eval_candidates.py --input data/jddc/raw/jddc_train.jsonl --limit 50
```

脚本会输出：

```text
data/eval/jddc_eval_candidates_50.csv
```

这个文件不是最终金标准，需要人工审核以下列：

```text
human_expected_categories
human_expected_risk_level
human_expected_priority
human_rule_reason
review_status
```
