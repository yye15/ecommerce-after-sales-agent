# 中文真实评论数据准备流程

## 推荐数据源：ChineseNlpCorpus online_shopping_10_cats

MARC 官方入口目前不可稳定访问，所以当前优先使用 `ChineseNlpCorpus` 的 `online_shopping_10_cats`。它包含 10 个类别、6 万多条中文评论，正负向评论各约 3 万条，类别包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店。

## 当前流程

```text
中文真实评论
  -> 清洗空文本、重复文本、过短文本
  -> 按正负标签分桶抽样
  -> 保存为 data/raw/chinese_shopping_sample.csv
  -> 后续用 DeepSeek 预标注风险等级和问题分类
  -> 人工抽查后形成 data/eval/golden_cases.csv
```

## 推荐抽样比例

| 类型 | 标签 | 比例 | 用途 |
|---|---:|---:|---|
| negative | 0 | 60% | 重点测试风险判断 |
| positive | 1 | 40% | 测试低风险判断 |

## 运行命令

```bash
python scripts/prepare_chinese_shopping.py --target-size 300
```

如果只想先试小样本：

```bash
python scripts/prepare_chinese_shopping.py --target-size 30
```

输出文件：

```text
data/raw/chinese_shopping_sample.csv
```

如果自动下载不可用，可以手动下载 `online_shopping_10_cats.csv` 后运行：

```bash
python scripts/prepare_chinese_shopping.py --raw-csv data/raw/online_shopping_10_cats.csv --target-size 300
```

## 后续标注标准

第一版风险标准：

- 安全问题：高风险
- 投诉 / 维权 / 举报：高风险
- 退款退货 + 客服不处理：高风险
- 质量 / 耐用性 / 核心功能负面：至少中风险
- 物流慢但语气温和：低到中风险
- 纯正面评价：低风险
