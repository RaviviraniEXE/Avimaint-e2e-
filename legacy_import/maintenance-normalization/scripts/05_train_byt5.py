"""Step 5 — fine-tune ByT5 (System C), leak-free.

Training = silver (System B output over all records) with gold_train corrections,
EXCLUDING every record in gold_dev/gold_test. Validation = gold_dev. gold_test is
never seen here (only 04_evaluate.py uses it).

Needs: torch (CUDA build for your driver), transformers, datasets, accelerate,
sentencepiece.  python scripts/05_train_byt5.py --out outputs/models/byt5
"""
import _bootstrap  # noqa: F401
import argparse
import os

import pandas as pd

from src.data.load import load_config


def _load(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False) if os.path.exists(path) else None


def build_pairs(cfg):
    silver_path = os.path.join(cfg["paths"]["outputs_dir"], "normalized", "normalized_B_rules.csv")
    if not os.path.exists(silver_path):
        raise SystemExit("Run System B first: scripts/03_run_normalization.py --systems B")
    silver = pd.read_csv(silver_path, dtype=str, keep_default_na=False)
    pairs = {r.IDENT: (r.raw, r.normalized) for r in silver.itertuples()}

    ev = cfg["evaluation"]
    g_tr, g_dv, g_te = _load(ev["gold_train_file"]), _load(ev["gold_dev_file"]), _load(ev["gold_test_file"])
    if g_tr is None and g_dv is None and g_te is None:
        raise SystemExit("Run scripts/split_gold.py first for a leak-free setup.")
    held = set()
    for g in (g_dv, g_te):
        if g is not None:
            held |= set(g["IDENT"])
    if g_tr is not None:
        for r in g_tr.itertuples():
            pairs[r.IDENT] = (r.RAW, r.GOLD)
        print(f"Applied {len(g_tr)} gold_train corrections.")
    train = {k: v for k, v in pairs.items() if k not in held}
    print(f"Excluded {len(held)} dev/test records from training.")
    tr_src = [v[0] for v in train.values() if v[0].strip()]
    tr_tgt = [v[1] for v in train.values() if v[0].strip()]
    dv_src = [r.RAW for r in g_dv.itertuples() if r.RAW.strip()] if g_dv is not None else []
    dv_tgt = [r.GOLD for r in g_dv.itertuples() if r.RAW.strip()] if g_dv is not None else []
    return tr_src, tr_tgt, dv_src, dv_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/models/byt5")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = cfg["byt5"]
    try:
        import torch
        from datasets import Dataset
        from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
                                  Seq2SeqTrainer, Seq2SeqTrainingArguments,
                                  T5ForConditionalGeneration)
    except ImportError as e:
        raise SystemExit(
            f"\nDependency import failed: {type(e).__name__}: {e}\n"
            "Fix the library named above. Full set needed:\n"
            "  pip install transformers datasets accelerate sentencepiece\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n")

    tr_src, tr_tgt, dv_src, dv_tgt = build_pairs(cfg)
    print(f"train pairs: {len(tr_src)} | dev pairs: {len(dv_src)}")
    tok = AutoTokenizer.from_pretrained(b["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(b["model_name"], use_safetensors=True)
    if b.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    train_ds = Dataset.from_dict({"src": tr_src, "tgt": tr_tgt})
    if dv_src:
        eval_ds = Dataset.from_dict({"src": dv_src, "tgt": dv_tgt})
    else:
        sp = train_ds.train_test_split(test_size=0.1, seed=b["seed"])
        train_ds, eval_ds = sp["train"], sp["test"]

    def tk(batch):
        mi = tok(batch["src"], max_length=b["max_source_length"], truncation=True)
        mi["labels"] = tok(text_target=batch["tgt"], max_length=b["max_target_length"], truncation=True)["input_ids"]
        return mi

    train_ds = train_ds.map(tk, batched=True, remove_columns=["src", "tgt"])
    eval_ds = eval_ds.map(tk, batched=True, remove_columns=["src", "tgt"])

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out, per_device_train_batch_size=b["train_batch_size"],
        per_device_eval_batch_size=b["train_batch_size"],
        gradient_accumulation_steps=b["gradient_accumulation_steps"],
        learning_rate=float(b["learning_rate"]), num_train_epochs=b["num_epochs"],
        fp16=False, bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()), eval_strategy="epoch",
        save_strategy="epoch", logging_steps=50, predict_with_generate=True,
        seed=b["seed"], report_to=[])
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds,
                             eval_dataset=eval_ds, data_collator=DataCollatorForSeq2Seq(tok, model=model),
                             processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
