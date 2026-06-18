# LALUN Chinese Fine-Tuning Plan

The current LALUN integration supports English inference. Chinese e-commerce fine-tuning requires span-level ASTE data.

## Why We Need Extra Data

The 50-row golden set labels business risk and issue categories. LALUN needs a different label format:

```text
aspect span + opinion span + sentiment
```

Example:

```text
差评，才用了一周，屏幕自己裂开
=> 屏幕 / 裂开 / NEG
```

The span positions of `屏幕` and `裂开` must be known.

## Current Data Scaffold

Generate a first small Chinese ASTE dataset:

```bash
python scripts/prepare_lalun_chinese_data.py
```

Output:

```text
external/LALUN/delivery_105/data/aste_data_bert/V2/zh_ecommerce/
  train.json
  dev.json
  test.json
```

This is only a bootstrap dataset. It keeps only examples where the aspect and opinion can be located in the original Chinese text. By default it also skips reviews longer than 80 Chinese characters, because LALUN's table encoder is memory-heavy.

## Serious Fine-Tuning Requirements

Before serious Chinese fine-tuning, prepare:

- A Chinese pretrained model, such as `bert-base-chinese` or a Chinese RoBERTa/MacBERT model.
- More Chinese ASTE triplets, ideally hundreds to thousands of examples.
- Better Chinese tokenization/POS/dependency features.
- A validation set that is manually checked.

## Recommended Route

1. Use DeepSeek to pseudo-label Chinese reviews into `aspect/opinion/sentiment` triplets.
2. Manually audit 100-300 examples.
3. Convert audited labels into LALUN JSON format.
4. Fine-tune LALUN with a Chinese pretrained model.
5. Compare LALUN output against DeepSeek/rule sentiment on the same golden set.

## Current Chinese Pseudo-Label Dataset

The current project has generated:

```text
data/labeled/deepseek_aste_300.csv
```

Summary:

```text
300 real Chinese e-commerce reviews
743 pseudo-labeled aspect/opinion/sentiment triplets
245 examples convertible to LALUN span format
```

The converted LALUN data is written to:

```text
external/LALUN/delivery_105/data/aste_data_bert/V2/zh_ecommerce_deepseek300/
  train.json
  dev.json
  test.json
```

For manual review, use:

```text
data/labeled/deepseek_aste_audit_100.csv
```

Reviewed output:

```text
data/labeled/deepseek_aste_audit_100_reviewed.csv
data/labeled/deepseek_aste_300_mixed_reviewed.csv
```

Audit guidance:

- Keep `triplets_json` if the label is correct.
- Put corrected labels into `reviewed_triplets_json` if it needs changes.
- Set `review_status` to `reviewed` after checking.
- Set `review_status` to `rejected` if the review is unusable.

Current audit result:

```text
100 sampled rows checked
91 rows kept
9 rows rejected
232 reviewed triplets
0 span-location errors
```

## Training Launcher

Download the recommended Chinese pretrained model first:

```bash
python scripts/download_chinese_pretrained_model.py
```

Default output:

```text
models/hfl_chinese_macbert_base
```

Use this project-side launcher instead of calling LALUN directly:

```bash
python scripts/run_lalun_chinese_finetune.py --dataset zh_ecommerce_auto_reviewed300 --model-name-or-path models/hfl_chinese_macbert_base --run
```

The launcher intentionally requires a Chinese pretrained model path. If you only want to test whether the training pipeline can start, use:

```bash
python scripts/run_lalun_chinese_finetune.py --dataset zh_ecommerce_auto_reviewed300 --smoke-with-distillation --run --epoch 1
```

The smoke mode uses the existing English/distilled LALUN checkpoint only to verify the pipeline. It is not a real Chinese fine-tuned model.
