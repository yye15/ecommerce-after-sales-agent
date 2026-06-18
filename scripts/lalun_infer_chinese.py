"""Single-sentence Chinese inference wrapper for the fine-tuned LALUN model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_LALUN_ROOT = Path(os.getenv("LALUN_ROOT", "external/LALUN/delivery_105"))
DEFAULT_MODEL_DIR = (
    DEFAULT_LALUN_ROOT
    / "code"
    / "output"
    / "zh_ecommerce_deepseek1000"
    / "lr_5e-06_bs_1_epo_3"
    / "model"
)
DEFAULT_TOKENIZER_DIR = Path(os.getenv("LALUN_CHINESE_TOKENIZER_DIR", "models/hfl_chinese_macbert_base"))
POLARITY = {1: "NEG", 2: "NEU", 3: "POS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LALUN Chinese ASTE inference.")
    parser.add_argument("--text", required=True, help="Chinese review text to analyze.")
    parser.add_argument("--lalun-root", default=str(DEFAULT_LALUN_ROOT))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--max-seq-length", type=int, default=128)
    return parser.parse_args()


def ensure_lalun_imports(lalun_root: Path) -> Path:
    code_dir = lalun_root / "code"
    if not code_dir.exists():
        raise RuntimeError(f"LALUN code directory not found: {code_dir}")
    sys.path.insert(0, str(code_dir))
    return code_dir


def clean_text(text: str) -> str:
    return "".join((text or "").split()).strip()


def tokenize_chars(text: str) -> list[str]:
    return [char for char in clean_text(text) if char.strip()]


def simple_adj(length: int) -> list[list[int]]:
    matrix = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(length):
        matrix[i][i] = 1
        if i > 0:
            matrix[i][i - 1] = 1
        if i < length - 1:
            matrix[i][i + 1] = 1
    return matrix


def build_example(text: str) -> dict[str, Any]:
    tokens = tokenize_chars(text)
    return {
        "ID": 0,
        "sentence": " ".join(tokens),
        "entities": [],
        "pairs": [],
        "tokens": str(tokens),
        "adj": simple_adj(len(tokens)),
        "postag": ["NN"] * len(tokens),
    }


def tensor_batch_to_cuda(batch: dict[str, Any]) -> dict[str, Any]:
    import torch

    moved = {}
    for key, value in batch.items():
        moved[key] = value.cuda() if isinstance(value, torch.Tensor) else value
    return moved


def span_text(tokens: list[str], start: int, end: int) -> str:
    start = max(0, start)
    end = min(len(tokens), end)
    return "".join(tokens[start:end])


def run_lalun(
    text: str,
    lalun_root: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    max_seq_length: int,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("LALUN inference requires CUDA because the original model code calls .cuda().")

    ensure_lalun_imports(lalun_root)
    from model.PBLUN_model import PBLUNModel
    from utils.aste_datamodule import DataCollatorForASTE, Example

    if not (model_dir / "model.safetensors").exists():
        raise RuntimeError(f"Fine-tuned Chinese LALUN model not found: {model_dir}")
    if not (tokenizer_dir / "vocab.txt").exists():
        raise RuntimeError(f"Chinese tokenizer not found: {tokenizer_dir}")

    config = AutoConfig.from_pretrained(str(model_dir))
    model = PBLUNModel.from_pretrained(str(model_dir), config=config).cuda()
    model.eval()

    example_dict = build_example(text)
    example = Example(example_dict, max_length=max_seq_length)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    collator = DataCollatorForASTE(tokenizer=tokenizer, max_seq_length=max_seq_length)
    batch = tensor_batch_to_cuda(collator([example]))

    with torch.no_grad():
        outputs = model(**batch)

    tokens = example_dict["sentence"].split()
    triplets = []
    seen = set()
    for pred in outputs.get("pairs_preds", []):
        _, a_start, a_end, o_start, o_end, polarity_id = pred
        aspect = span_text(tokens, int(a_start), int(a_end))
        opinion = span_text(tokens, int(o_start), int(o_end))
        sentiment = POLARITY.get(int(polarity_id), "NEU")
        key = (aspect, opinion, sentiment)
        if aspect and opinion and key not in seen:
            seen.add(key)
            triplets.append(
                {
                    "aspect": aspect,
                    "opinion": opinion,
                    "sentiment": sentiment,
                    "evidence": clean_text(text),
                }
            )

    return {
        "enabled": True,
        "engine": "lalun_zh_experimental",
        "model_dir": str(model_dir),
        "input": text,
        "normalized_sentence": example_dict["sentence"],
        "triplets": triplets,
    }


def main() -> int:
    args = parse_args()
    try:
        result = run_lalun(
            text=args.text,
            lalun_root=Path(args.lalun_root),
            model_dir=Path(args.model_dir),
            tokenizer_dir=Path(args.tokenizer_dir),
            max_seq_length=args.max_seq_length,
        )
    except Exception as exc:
        result = {
            "enabled": False,
            "engine": "lalun_zh_experimental",
            "input": args.text,
            "error": str(exc),
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
