"""Launch LALUN fine-tuning on the prepared Chinese e-commerce ASTE data.

This script is intentionally conservative:

- It checks that the Chinese ASTE JSON files exist.
- It requires a local Chinese pretrained model path for real fine-tuning.
- It offers a smoke mode with LALUN's bundled distillation model, but that mode
  is only for checking whether the training pipeline starts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


LALUN_ROOT = Path(os.getenv("LALUN_ROOT", "external/LALUN/delivery_105"))
LALUN_CODE = LALUN_ROOT / "code"
LALUN_DATA_ROOT = LALUN_ROOT / "data" / "aste_data_bert" / "V2"
LALUN_PYTHON = Path(os.getenv("LALUN_PYTHON", "python"))
DISTILLATION_MODEL = LALUN_CODE / "pretrained_model" / "distillation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LALUN Chinese fine-tuning.")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--dataset", default="zh_ecommerce_auto_reviewed300")
    parser.add_argument("--smoke-with-distillation", action="store_true")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--run", action="store_true", help="Actually start training.")
    return parser.parse_args()


def count_split(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as f:
        return len(json.load(f))


def validate_data(dataset: str) -> tuple[Path, dict[str, int]]:
    data_dir = LALUN_DATA_ROOT / dataset
    counts = {
        "train": count_split(data_dir / "train.json"),
        "dev": count_split(data_dir / "dev.json"),
        "test": count_split(data_dir / "test.json"),
    }
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise FileNotFoundError(
            "Missing or empty LALUN Chinese data splits: "
            + ", ".join(missing)
            + ". Run scripts/prepare_lalun_chinese_data.py first."
        )
    return data_dir, counts


def resolve_model(args: argparse.Namespace) -> tuple[str, str]:
    if args.smoke_with_distillation:
        if not DISTILLATION_MODEL.exists():
            raise FileNotFoundError(f"Missing smoke model: {DISTILLATION_MODEL}")
        return str(DISTILLATION_MODEL), "smoke"

    if not args.model_name_or_path:
        raise ValueError(
            "Real Chinese fine-tuning needs --model-name-or-path pointing to a "
            "local Chinese pretrained model, for example bert-base-chinese or "
            "hfl/chinese-macbert-base downloaded locally."
        )

    model_path = Path(args.model_name_or_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. "
            "Please download/provide a Chinese pretrained model first."
        )
    return str(model_path), "real"


def build_command(args: argparse.Namespace, model_path: str) -> list[str]:
    return [
        str(LALUN_PYTHON),
        "aste_train.py",
        "--model_name_or_path",
        model_path,
        "--prefix",
        "../data/aste_data_bert/V2/",
        "--dataset",
        args.dataset,
        "--epoch",
        str(args.epoch),
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--p",
        str(args.precision),
        "--max_seq_length",
        str(args.max_seq_length),
        "--num_workers",
        "1",
    ]


def main() -> int:
    args = parse_args()
    if not LALUN_PYTHON.exists():
        raise FileNotFoundError(f"Missing LALUN Python env: {LALUN_PYTHON}")
    if not (LALUN_CODE / "aste_train.py").exists():
        raise FileNotFoundError(f"Missing LALUN train script: {LALUN_CODE / 'aste_train.py'}")

    data_dir, counts = validate_data(args.dataset)
    model_path, mode = resolve_model(args)
    command = build_command(args, model_path)

    print("LALUN Chinese fine-tuning launcher")
    print(f"Mode: {mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Data: {data_dir}")
    print(f"Split counts: {counts}")
    print(f"Model: {model_path}")
    print("Command:")
    print(" ".join(command))

    if not args.run:
        print("\nDry run only. Add --run to start training.")
        return 0

    if mode == "smoke":
        print("\nWarning: smoke mode is not a real Chinese fine-tuned model.")

    completed = subprocess.run(command, cwd=str(LALUN_CODE), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
