"""Compare systems A/B/C/D and save all results.

Intrinsic metrics on every normalized record; extrinsic (WER/CER/ERR/exact) on
the held-out test split when it exists (else full gold). Writes:
    outputs/reports/intrinsic_metrics.csv
    outputs/reports/extrinsic_metrics.csv
    outputs/reports/experiment_report.md
    outputs/reports/results_log.csv   (one row per system per run, appended)
"""
from __future__ import annotations

import os
from typing import Dict

import pandas as pd

from src.data.load import load_config, load_domain_vocab
from src.evaluation.metrics import extrinsic_report, intrinsic_report

LABELS = {"A_raw": "A · Raw", "B_rules": "B · Rule-based",
          "C_byt5": "C · Transformer (ByT5)", "D_hybrid": "D · Hybrid"}


def _load_outputs(norm_dir):
    out = {}
    for fn in sorted(os.listdir(norm_dir)):
        if fn.startswith("normalized_") and fn.endswith(".csv"):
            sysname = fn[len("normalized_"):-len(".csv")]
            out[sysname] = pd.read_csv(os.path.join(norm_dir, fn), dtype=str, keep_default_na=False)
    return out


def run(config_path: str = "config/config.yaml", run_id: str = "run") -> Dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    vocab = load_domain_vocab(cfg)
    norm_dir = os.path.join(cfg["paths"]["outputs_dir"], "normalized")
    report_dir = cfg["evaluation"]["report_dir"]
    os.makedirs(report_dir, exist_ok=True)
    outputs = _load_outputs(norm_dir)
    if not outputs:
        return {"intrinsic": pd.DataFrame(), "extrinsic": pd.DataFrame()}

    # intrinsic
    rows = []
    for name, df in outputs.items():
        stats = [{"n_expansions": int(e or 0), "n_tokens": int(t or 0)}
                 for e, t in zip(df.get("n_expansions", ["0"] * len(df)),
                                 df.get("n_tokens", ["0"] * len(df)))]
        rep = intrinsic_report(df["raw"].tolist(), df["normalized"].tolist(), stats, vocab)
        rep["system"] = name
        rows.append(rep)
    intrinsic = pd.DataFrame(rows)
    intrinsic = intrinsic[["system"] + [c for c in intrinsic.columns if c != "system"]].sort_values("system")
    intrinsic.to_csv(os.path.join(report_dir, "intrinsic_metrics.csv"), index=False)

    # extrinsic (prefer held-out test split)
    extrinsic = pd.DataFrame()
    test = cfg["evaluation"].get("gold_test_file", "")
    gold_path = test if test and os.path.exists(test) else cfg["evaluation"]["gold_file"]
    scored_on = "held-out test split" if gold_path == test else "full gold"
    if os.path.exists(gold_path):
        print(f"[extrinsic] scoring on {scored_on}: {gold_path}")
        gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
        base_map = {}
        if "A_raw" in outputs:
            a = outputs["A_raw"]
            m = gold.merge(a[["IDENT", "normalized"]], on="IDENT", how="inner")
            base_map = {r.IDENT: r.normalized for r in m.itertuples()}
        erows = []
        for name, df in outputs.items():
            m = gold.merge(df[["IDENT", "normalized"]], on="IDENT", how="inner")
            if m.empty:
                continue
            base = [base_map.get(r.IDENT, r.RAW) for r in m.itertuples()] if base_map else None
            rep = extrinsic_report(m["normalized"].tolist(), m["GOLD"].tolist(), base)
            rep["system"] = name
            erows.append(rep)
        if erows:
            extrinsic = pd.DataFrame(erows)
            extrinsic = extrinsic[["system"] + [c for c in extrinsic.columns if c != "system"]].sort_values("system")
            extrinsic.to_csv(os.path.join(report_dir, "extrinsic_metrics.csv"), index=False)

    _write_report(cfg, intrinsic, extrinsic, report_dir, scored_on)
    _append_log(cfg, intrinsic, extrinsic, run_id)
    return {"intrinsic": intrinsic, "extrinsic": extrinsic}


def _md(df):
    h = "| " + " | ".join(df.columns) + " |"
    s = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    r = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([h, s] + r)


def _write_report(cfg, intrinsic, extrinsic, report_dir, scored_on):
    L = ["# Normalization experiment — system comparison", "",
         f"Number convention: **{cfg.get('numbers')}**. Gold: Amin expert-cleaned dataset.",
         f"Extrinsic scored on: **{scored_on}**.", "",
         "| System | Method |", "|---|---|",
         "| A · Raw | no normalization (control) |",
         "| B · Rule-based | Amin abbreviation + misspelling + keep lists |",
         "| C · Transformer | ByT5 char-level seq2seq |",
         "| D · Hybrid | rules then ByT5 |", "",
         "## Intrinsic metrics (all records)", ""]
    if not intrinsic.empty:
        show = intrinsic.copy(); show["system"] = show["system"].map(lambda s: LABELS.get(s, s))
        L.append(_md(show))
    L += ["", "## Extrinsic metrics (vs Amin gold)", ""]
    if extrinsic is not None and not extrinsic.empty:
        show = extrinsic.copy(); show["system"] = show["system"].map(lambda s: LABELS.get(s, s))
        L.append(_md(show))
        L += ["", "`wer`/`cer` lower is better; `exact_match`, `err_word`, `err_char` higher is better. "
              "**ERR** = fraction of raw error removed (MaintNorm's metric; they report 95.8%)."]
    else:
        L.append("_No gold found — run split_gold.py / 01_prepare_data.py first._")
    with open(os.path.join(report_dir, "experiment_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _append_log(cfg, intrinsic, extrinsic, run_id):
    path = cfg["evaluation"]["results_log"]
    ex = {r["system"]: r for r in extrinsic.to_dict("records")} if extrinsic is not None and not extrinsic.empty else {}
    rows = []
    for r in intrinsic.to_dict("records"):
        s = r["system"]
        e = ex.get(s, {})
        rows.append({"run_id": run_id, "numbers": cfg.get("numbers"), "system": s,
                     "oov_reduction_pct": r["oov_reduction_pct"], "expansions": r["expansions"],
                     "wer": e.get("wer", ""), "cer": e.get("cer", ""),
                     "exact_match": e.get("exact_match", ""),
                     "err_word": e.get("err_word", ""), "err_char": e.get("err_char", "")})
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False), new.astype(str)], ignore_index=True)
    new.to_csv(path, index=False)


if __name__ == "__main__":
    res = run()
    print(res["intrinsic"].to_string(index=False))
    if not res["extrinsic"].empty:
        print(res["extrinsic"].to_string(index=False))

