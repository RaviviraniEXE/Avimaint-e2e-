"""Train MaintIE FastText embeddings from the FROZEN TRAIN split only.

This benchmark-specific version intentionally does NOT call the aviation corpus
loader and does NOT read Aircraft_Annotation_DataFile.csv.  It trains the
unsupervised FastText model on tokens from MaintIE TRAIN documents only, keeping
DEV/TEST completely outside embedding fitting for a strict external-benchmark
protocol.

Writes outputs/embeddings/domain_ft.model, consumed automatically by Tier 2.
"""
import _bootstrap  # noqa: F401

from src.data.gold import load_gold
from src.data.split import assign, load_splits
from src.models.embeddings import train_fasttext


def main():
    gold = load_gold("outputs/gold/*.jsonl")
    if not gold:
        raise SystemExit("No MaintIE gold found under outputs/gold/*.jsonl. Run 01_prepare.bat first.")
    if not load_splits():
        raise SystemExit("Frozen MaintIE split missing: outputs/splits.json. Run 01_prepare.bat first.")

    train, dev, test = assign(gold)
    if not train:
        raise SystemExit("MaintIE TRAIN split is empty.")

    sentences = [list(r.get("tokens", [])) for r in train if r.get("tokens")]
    n_tokens = sum(len(s) for s in sentences)
    print("[MAINTIE EMBEDDINGS] benchmark TRAIN-only FastText")
    print(f"frozen split: train={len(train)} dev={len(dev)} test={len(test)}")
    print(f"embedding corpus: {len(sentences)} TRAIN documents / {n_tokens} tokens")
    print("DEV/TEST used for embedding fitting: NO")

    model = train_fasttext(sentences)
    print(f"vocab={len(model.wv)}  dim={model.wv.vector_size}")
    print("Saved -> outputs/embeddings/domain_ft.model")
    print("Tier 2 will use this benchmark-specific TRAIN-only embedding model.")


if __name__ == "__main__":
    main()
