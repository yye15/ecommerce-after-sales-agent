"""Download the recommended Chinese pretrained model for LALUN fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REPO = "hfl/chinese-macbert-base"
DEFAULT_OUTPUT = Path("models/hfl_chinese_macbert_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Chinese MacBERT for LALUN.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
