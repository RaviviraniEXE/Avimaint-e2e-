"""Produce a per-column normalized dataset for the AviMaint-DSS dashboard.

Your normalization pipeline normalizes the COMBINED problem+action text. The
dashboard needs PROBLEM and ACTION as separate columns. This script reuses your
exact System B (Amin's expert abbreviation + misspelling + keep lists) and
applies it to each column independently — so the combined-vs-separate issue
disappears and the output is a drop-in dataset for the dashboard.

RUN THIS FROM THE ROOT OF YOUR NORMALIZATION PROJECT (the folder that has
`src/`, `scripts/`, and `config/config.yaml`):

    .venv\\Scripts\\activate            # your existing venv
    python make_dashboard_dataset.py

Output:
    outputs/normalized/dashboard_dataset_B.csv   (IDENT, PROBLEM, ACTION)

Then in the dashboard's config.yaml set:
    data:  { csv_path: <path to dashboard_dataset_B.csv> }
    extraction: { normalize: false }   # data is already normalized by System B
and delete the dashboard's data/cache/ folder.

To use System D (rules + ByT5) instead of B, pass --system D --byt5-dir outputs/models/byt5
(needs your trained ByT5 checkpoint).
"""
import argparse
import os
import sys

# make `src` importable when run from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src.data.load import load_config, load_logbook
from src.data.dictionary import build_lexicon
from src.normalization.system_b_rules import RuleBasedNormalizer


def build_normalizer(cfg, system, byt5_dir):
    lex, _ = build_lexicon(cfg)
    rules = RuleBasedNormalizer(lex, cfg)
    if system == "B":
        return rules
    if system in ("C", "D"):
        from src.normalization.system_c_byt5 import ByT5Normalizer
        if not os.path.isdir(byt5_dir):
            raise SystemExit(
                f"Trained ByT5 checkpoint not found at '{byt5_dir}'.\n"
                "Train it first:  python scripts/05_train_byt5.py --out outputs/models/byt5\n"
                "or pass --byt5-dir <path to your checkpoint>.")
        byt5 = ByT5Normalizer(byt5_dir, cfg.get("byt5", {}))
        if system == "C":
            return byt5
        from src.normalization.system_d_hybrid import HybridNormalizer
        return HybridNormalizer(rules, byt5)
    raise SystemExit("Use --system B (rules), C (ByT5), or D (rules+ByT5).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--system", default="C", choices=["B", "C", "D"],
                    help="C = ByT5 transformer (best per the chapter); D = rules+ByT5; B = rules")
    ap.add_argument("--byt5-dir", default="outputs/models/byt5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"outputs/normalized/dashboard_dataset_{args.system}.csv"

    cfg = load_config(args.config)
    norm = build_normalizer(cfg, args.system, args.byt5_dir)
    df = load_logbook(cfg)
    lc = cfg["logbook"]

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    def nz_col(series, label):
        return [norm.normalize(t).normalized
                for t in tqdm(series.tolist(), desc=label, unit="rec")]

    n = len(df)
    print(f"Normalizing {n} records per column with System {args.system} "
          f"({2 * n} generations total). ByT5 beam-search is the slow part — "
          f"the progress bars below show it working.\n")
    out = pd.DataFrame({
        "IDENT": df[lc["id_col"]].astype(str),
        "PROBLEM": nz_col(df[lc["problem_col"]], "PROBLEM"),
        "ACTION": nz_col(df[lc["action_col"]], "ACTION"),
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} rows -> {args.out}\n")

    # show a few before/after examples so you can eyeball it
    print("Examples (raw -> normalized, per column):")
    for i in range(min(4, len(df))):
        rp, ra = df[lc["problem_col"]].iloc[i], df[lc["action_col"]].iloc[i]
        print(f"  IDENT {out['IDENT'].iloc[i]}")
        print(f"    PROBLEM: {rp!r}\n          -> {out['PROBLEM'].iloc[i]!r}")
        print(f"    ACTION : {ra!r}\n          -> {out['ACTION'].iloc[i]!r}")


if __name__ == "__main__":
    main()
