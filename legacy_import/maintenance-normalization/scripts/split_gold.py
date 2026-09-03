"""Split the gold into train/dev/test (leak-free) for Systems C and D.

ByT5 trains on gold_train (+ silver), validates on gold_dev, and is NEVER
trained on gold_test. Deterministic (seed-fixed), disjoint-checked.
"""
import _bootstrap  # noqa: F401
import argparse
import os
import random

import pandas as pd

from src.data.load import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    sp = cfg["evaluation"]["split"]

    gold = pd.read_csv(cfg["evaluation"]["gold_file"], dtype=str, keep_default_na=False)
    idx = list(range(len(gold)))
    random.Random(seed).shuffle(idx)
    n_tr = int(round(len(gold) * sp["train"]))
    n_dv = int(round(len(gold) * sp["dev"]))
    parts = {
        cfg["evaluation"]["gold_train_file"]: gold.iloc[sorted(idx[:n_tr])],
        cfg["evaluation"]["gold_dev_file"]: gold.iloc[sorted(idx[n_tr:n_tr + n_dv])],
        cfg["evaluation"]["gold_test_file"]: gold.iloc[sorted(idx[n_tr + n_dv:])],
    }
    for path, part in parts.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part.to_csv(path, index=False)
    keys = [set(p["IDENT"]) for p in parts.values()]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2]), "OVERLAP"
    print(f"Split {len(gold)} (seed={seed}): "
          f"train={len(parts[cfg['evaluation']['gold_train_file']])}, "
          f"dev={len(parts[cfg['evaluation']['gold_dev_file']])}, "
          f"test={len(parts[cfg['evaluation']['gold_test_file']])}  | disjoint OK")


if __name__ == "__main__":
    main()

