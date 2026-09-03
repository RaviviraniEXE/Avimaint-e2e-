"""Step 3 — normalize the combined records with one or more systems.

  python scripts/03_run_normalization.py --systems A B
  python scripts/03_run_normalization.py --systems A B C D --byt5-dir outputs/models/byt5

Writes outputs/normalized/normalized_<system>.csv (IDENT, raw, normalized,
n_tokens, n_expansions) and alignment_<system>.jsonl for B/D.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from src.data.dictionary import build_lexicon
from src.data.load import load_config
from src.normalization.system_a_raw import RawNormalizer
from src.normalization.system_b_rules import RuleBasedNormalizer
from src.normalization.system_c_byt5 import ByT5Normalizer
from src.normalization.system_d_hybrid import HybridNormalizer


def build_systems(cfg, which, byt5_dir):
    lex, _ = build_lexicon(cfg)
    rules = RuleBasedNormalizer(lex, cfg)
    systems = {}
    if "A" in which:
        systems["A_raw"] = RawNormalizer(cfg["normalizer"].get("lowercase", True))
    if "B" in which:
        systems["B_rules"] = rules
    byt5 = ByT5Normalizer(byt5_dir, cfg.get("byt5", {})) if (byt5_dir and ("C" in which or "D" in which)) else None
    if "C" in which:
        if byt5:
            systems["C_byt5"] = byt5
        else:
            print("[warn] System C requested but --byt5-dir missing; skipping.")
    if "D" in which:
        systems["D_hybrid"] = HybridNormalizer(rules, byt5)
    return systems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["A", "B"], choices=["A", "B", "C", "D"])
    ap.add_argument("--byt5-dir", default=None)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rec_path = os.path.join(cfg["paths"]["interim_dir"], "records_combined.csv")
    if not os.path.exists(rec_path):
        raise SystemExit("Run scripts/01_prepare_data.py first.")
    recs = pd.read_csv(rec_path, dtype=str, keep_default_na=False)
    out_dir = os.path.join(cfg["paths"]["outputs_dir"], "normalized")
    os.makedirs(out_dir, exist_ok=True)

    for name, system in build_systems(cfg, set(args.systems), args.byt5_dir).items():
        rows = []
        write_align = name in ("B_rules", "D_hybrid")
        af = open(os.path.join(out_dir, f"alignment_{name}.jsonl"), "w", encoding="utf-8") if write_align else None
        for _, r in tqdm(recs.iterrows(), total=len(recs), desc=name):
            res = system.normalize(r["RAW"])
            rows.append({"IDENT": r["IDENT"], "raw": res.raw, "normalized": res.normalized,
                         "n_tokens": res.stats.get("n_tokens", 0),
                         "n_expansions": res.stats.get("n_expansions", 0)})
            if af and res.alignment:
                af.write(json.dumps({"IDENT": r["IDENT"],
                                     "alignment": [{"raw": a[0], "span": a[1], "norm": a[2], "rule": a[3]}
                                                   for a in res.alignment]}) + "\n")
        if af:
            af.close()
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"normalized_{name}.csv"), index=False)
        print(f"[{name}] wrote {len(rows)} rows")


if __name__ == "__main__":
    main()

