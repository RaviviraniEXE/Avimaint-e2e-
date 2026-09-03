"""Step 8b (optional) — Domain-Adaptive PreTraining (DAPT) for the Tier-3 encoder.

Continues masked-language-model pretraining of the transformer encoder on the FULL
unlabeled maintenance corpus (~thousands of records), so it learns domain vocabulary
and phrasing BEFORE fine-tuning on the small gold set (Gururangan et al. 2020,
"Don't Stop Pretraining"). Unsupervised — reads raw text only, no labels, no leakage.

  python scripts/08b_domain_pretrain.py
  # then in config/schema.yaml set:  models.transformer.encoder: outputs/dapt/encoder
  # and re-run:  python scripts/09_report.py --tiers 3 --tune --run-id gold1400_dapt

Install: pip install torch transformers
"""
import _bootstrap  # noqa: F401
import argparse
import os
import random

from src.data.corpus import load
from src.schema import load_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="base encoder (default: schema encoder, or distilbert)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--out", default="outputs/dapt/encoder")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                              DataCollatorForLanguageModeling)

    schema = load_schema()
    base = args.base or schema.get("models", {}).get("transformer", {}).get("encoder", "distilbert-base-uncased")
    if os.path.isdir(base):                     # don't DAPT on top of a previous DAPT dir
        base = "distilbert-base-uncased"
    df, _, _, _ = load()
    texts = [t for t in df["text"].tolist() if t and t.strip()]
    print(f"DAPT: MLM-pretraining '{base}' on {len(texts)} unlabeled records "
          f"({args.epochs} epochs, max_len={args.max_len})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForMaskedLM.from_pretrained(base).to(dev)
    ids = tok(texts, truncation=True, max_length=args.max_len)["input_ids"]
    coll = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    order = list(range(len(ids)))
    for ep in range(args.epochs):
        random.Random(ep).shuffle(order)
        model.train(); tot = nb = 0
        for s in range(0, len(order), args.batch_size):
            batch = coll([{"input_ids": ids[i]} for i in order[s:s + args.batch_size]])
            batch = {k: v.to(dev) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(out.loss.detach()); nb += 1
        print(f"  epoch {ep + 1}/{args.epochs}  mlm-loss={tot / max(1, nb):.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nsaved DAPT encoder -> {args.out}")
    print(f"next: set config/schema.yaml  models.transformer.encoder: {args.out}")
    print("      then: python scripts/09_report.py --tiers 1 2 3 --tune --run-id gold1400_dapt")


if __name__ == "__main__":
    main()

