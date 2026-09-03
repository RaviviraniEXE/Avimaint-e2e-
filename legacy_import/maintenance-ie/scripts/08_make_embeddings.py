"""Step 8 (optional, one-off) — train domain FastText embeddings for Tier 2.

The embeddings are trained on the full *unlabelled* aviation corpus.  Labels,
DEV scores and TEST scores are never read here, so this preprocessing step does
not use the frozen evaluation labels.

In the rebuilt project the canonical inputs are:
  ../../data/aviation/raw/Aircraft_Annotation_DataFile.csv
  ../../data/aviation/processed/normalized_corpus.csv

The loader also retains compatibility with the historical legacy-local paths.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data.corpus import load, resolve_corpus_paths
from src.data.preannotate import preannotate
from src.models.embeddings import train_fasttext


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/Aircraft_Annotation_DataFile.csv")
    p.add_argument("--norm", default="data/raw/normalized_corpus.csv")
    p.add_argument("--check-only", action="store_true", help="Resolve and validate inputs without training FastText.")
    return p.parse_args()


def main():
    args = parse_args()
    raw_path, norm_path = resolve_corpus_paths(args.raw, args.norm)

    print("[FASTTEXT] Canonical corpus preflight", flush=True)
    print(f"  raw : {raw_path}", flush=True)
    print(f"  norm: {norm_path}", flush=True)

    df, pool, norm_map, _ = load(raw_path, norm_path)
    nonempty = int(df["text"].astype(str).str.strip().ne("").sum())
    print(
        f"  records={len(df)} | unique_pool={len(pool)} | "
        f"normalized_nonempty={nonempty} | norm_map={len(norm_map)}",
        flush=True,
    )

    if args.check_only:
        print("[FASTTEXT] Input check passed; training was not started.", flush=True)
        return

    texts = [t for t in df["text"].tolist() if t and t.strip()]
    sentences = [preannotate(t)["tokens"] for t in texts]
    sentences = [s for s in sentences if s]
    n_tokens = sum(len(s) for s in sentences)

    print(
        f"[FASTTEXT] Training one-off domain embeddings on {len(sentences)} records "
        f"({n_tokens} tokens). dim=100, epochs=15, seed=42, workers=1",
        flush=True,
    )
    print("[FASTTEXT] This is auxiliary unsupervised preprocessing, not DEV/TEST model selection.", flush=True)

    model = train_fasttext(sentences)
    target = Path("outputs/embeddings/domain_ft.model")
    print(f"[FASTTEXT] COMPLETE | vocab={len(model.wv)} | dim={model.wv.vector_size}", flush=True)
    print(f"[FASTTEXT] Saved -> {target.resolve()}", flush=True)
    print("[FASTTEXT] Tier 2 will automatically reuse this model on later runs.", flush=True)


if __name__ == "__main__":
    main()
