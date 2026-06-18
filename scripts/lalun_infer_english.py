"""Single-sentence English inference wrapper for the local LALUN model.

Run this script with the LALUN environment, for example:

    python scripts/lalun_infer_english.py --text "The bread is top notch as well ."

The wrapper intentionally lives outside the original LALUN directory so we do
not mutate the research code. It builds the minimum ASTE example structure that
LALUN expects and returns JSON triplets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LALUN_ROOT = Path(os.getenv("LALUN_ROOT", "external/LALUN/delivery_105"))
DEFAULT_DATASET = "14res"
MODEL_MAP = {
    "14res": "lr_2e-05_bs_4_epo_10",
    "14lap": "lr_4e-05_bs_4_epo_15",
    "15res": "lr_6e-05_bs_4_epo_15",
    "16res": "lr_4e-05_bs_6_epo_20",
}
POLARITY = {1: "NEG", 2: "NEU", 3: "POS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LALUN English ASTE inference.")
    parser.add_argument("--text", required=True, help="English sentence to analyze.")
    parser.add_argument("--lalun-root", default=str(DEFAULT_LALUN_ROOT))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(MODEL_MAP))
    parser.add_argument("--max-seq-length", type=int, default=-1)
    return parser.parse_args()


def ensure_lalun_imports(lalun_root: Path) -> Path:
    code_dir = lalun_root / "code"
    if not code_dir.exists():
        raise RuntimeError(f"LALUN code directory not found: {code_dir}")
    sys.path.insert(0, str(code_dir))
    return code_dir


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", text)
    return tokens or text.split()


def guess_pos(tokens: list[str]) -> list[str]:
    adjective_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "bad",
        "terrible",
        "awful",
        "slow",
        "fast",
        "delicious",
        "fresh",
        "rude",
        "friendly",
        "expensive",
        "cheap",
        "clean",
        "dirty",
        "top",
        "notch",
    }
    pos_tags: list[str] = []
    for token in tokens:
        lower = token.lower()
        if re.fullmatch(r"[^\w\s]", token):
            pos_tags.append(".")
        elif lower in adjective_words or lower.endswith(("y", "ive", "ous", "ful", "less", "able")):
            pos_tags.append("JJ")
        elif lower.endswith("ing"):
            pos_tags.append("VBG")
        elif lower.endswith("ed"):
            pos_tags.append("VBN")
        else:
            pos_tags.append("NN")
    return pos_tags


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
    tokens = tokenize(text)
    sentence = " ".join(tokens)
    return {
        "ID": 0,
        "sentence": sentence,
        "entities": [],
        "pairs": [],
        "tokens": str([token.lower() for token in tokens]),
        "adj": simple_adj(len(tokens)),
        "postag": guess_pos(tokens),
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
    return " ".join(tokens[start:end])


def run_lalun(text: str, lalun_root: Path, dataset: str, max_seq_length: int) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("LALUN inference requires CUDA because the original model code calls .cuda().")

    code_dir = ensure_lalun_imports(lalun_root)
    from model.PBLUN_model import PBLUNModel
    from utils.aste_datamodule import DataCollatorForASTE, Example

    model_dir = code_dir / "output" / dataset / MODEL_MAP[dataset] / "model"
    tokenizer_dir = code_dir / "pretrained_model" / "distillation"
    if not (model_dir / "model.safetensors").exists():
        raise RuntimeError(f"Trained LALUN model not found: {model_dir}")

    config = AutoConfig.from_pretrained(str(model_dir))
    model = PBLUNModel.from_pretrained(str(model_dir), config=config).cuda()
    model.eval()

    example_dict = build_example(text)
    example = Example(example_dict, max_length=max_seq_length)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    collator = DataCollatorForASTE(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length if max_seq_length > 0 else "longest",
    )
    batch = collator([example])
    batch = tensor_batch_to_cuda(batch)

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
                    "evidence": example_dict["sentence"],
                }
            )

    return {
        "enabled": True,
        "engine": "lalun",
        "dataset": dataset,
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
            dataset=args.dataset,
            max_seq_length=args.max_seq_length,
        )
    except Exception as exc:
        result = {
            "enabled": False,
            "engine": "lalun",
            "input": args.text,
            "error": str(exc),
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
