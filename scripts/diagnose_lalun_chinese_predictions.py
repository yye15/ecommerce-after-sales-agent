"""Diagnose why a fine-tuned LALUN checkpoint emits empty pair predictions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig


LALUN_CODE = Path(os.getenv("LALUN_CODE", "external/LALUN/delivery_105/code"))
LALUN_DATA_PREFIX = "../data/aste_data_bert/V2/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=os.getenv("LALUN_CHINESE_TOKENIZER_DIR", "models/hfl_chinese_macbert_base"))
    parser.add_argument("--dataset", default="zh_ecommerce_auto_reviewed300")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--span-pruning", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(LALUN_CODE))

    from model.PBLUN_model import PBLUNModel
    from utils.aste_datamodule import ASTEDataModule

    class HParams:
        model_name_or_path = args.tokenizer
        prefix = LALUN_DATA_PREFIX
        dataset = args.dataset
        data_dir = LALUN_DATA_PREFIX + args.dataset
        max_seq_length = args.max_seq_length
        train_batch_size = 1
        eval_batch_size = 1
        num_workers = 1
        cuda_ids = 0
        table_encoder = "resnet"
        num_table_layers = 2
        num_t = 2
        seq = "tensorcontext"
        seq2mat = "tensorcontext"
        num_d = 64
        span_pruning = args.span_pruning

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(LALUN_CODE)
        dm = ASTEDataModule(HParams())
        dm.load_dataset()
        config = AutoConfig.from_pretrained(args.checkpoint)
        config.table_num_labels = dm.table_num_labels
        config.table_encoder = HParams.table_encoder
        config.num_table_layers = HParams.num_table_layers
        config.span_pruning = HParams.span_pruning
        config.seq2mat = HParams.seq
        config.num_d = HParams.num_d

        model = PBLUNModel.from_pretrained(args.checkpoint, config=config).cuda().eval()
        loader = {
            "train": dm.train_dataloader,
            "dev": dm.val_dataloader,
            "test": dm.test_dataloader,
        }[args.split]()

        total_s = total_e = total_pair_preds = total_true = 0
        shown = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= args.limit:
                    break
                cuda_batch = {}
                for key, value in batch.items():
                    cuda_batch[key] = value.cuda() if torch.is_tensor(value) else value
                outputs = model(**cuda_batch)
                s_count = int(outputs["table_predict_S"].sum().item())
                e_count = int(outputs["table_predict_E"].sum().item())
                pair_count = len(outputs["pairs_preds"])
                true_count = sum(len(x) for x in batch["pairs_true"])
                total_s += s_count
                total_e += e_count
                total_pair_preds += pair_count
                total_true += true_count
                if shown < 5:
                    print(
                        f"batch={batch_idx} ids={batch['ids']} "
                        f"S={s_count} E={e_count} pair_preds={pair_count} true={true_count}"
                    )
                    print(
                        "  debug="
                        f"candidate_count={outputs.get('debug_candidate_count')} "
                        f"argmax_counts={outputs.get('debug_pair_argmax_counts')} "
                        f"gold_in_candidates={outputs.get('debug_gold_in_candidates')}"
                    )
                    print(f"  text={batch['text'][0]}")
                    print(f"  true={batch['pairs_true'][0]}")
                    print(f"  pred={outputs['pairs_preds']}")
                    shown += 1

        print("summary")
        print(f"  checked_batches={min(args.limit, len(loader))}")
        print(f"  total_S_candidates={total_s}")
        print(f"  total_E_candidates={total_e}")
        print(f"  total_pair_preds={total_pair_preds}")
        print(f"  total_true_pairs={total_true}")
    finally:
        import os

        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
