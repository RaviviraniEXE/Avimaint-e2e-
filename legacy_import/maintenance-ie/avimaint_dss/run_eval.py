"""Offline evaluation report for the recommender (cluster-safe leave-one-out).

Measures AGREEMENT WITH RECORDED PRACTICE (not correctness — no gold exists).

  python run_eval.py                      # configured dataset, no reranker
  python run_eval.py --full               # whole corpus (definitive)
  python run_eval.py --reranker           # add the cross-encoder reranker
  python run_eval.py --compare --full     # raw vs System-D on the SAME query set
  python run_eval.py --all --full         # the full grid: {raw, D} x {reranker off, on}

Flags
  --sample N   sample size when not --full (default 1200)
  --full       evaluate every classifiable record (matched across datasets in compare/all)
  --reranker   route retrieval through the cross-encoder reranker (retrieval.reranker_model)
  --compare    System-D vs raw, evaluated on the identical query set
  --all        the 2x2 grid {raw, D} x {reranker off, on} on the identical query set

Writes outputs/reports/eval_report.json and eval_report.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from core.corpus import load_corpus
from core.retrieval import Retriever
from core.reranker import CrossEncoderReranker
from core import evaluate as EV

ROOT = Path(__file__).resolve().parent
D_CSV = ROOT / "data" / "dashboard_dataset_D.csv"
RAW_CSV = ROOT / "data" / "Aircraft_Annotation_DataFile.csv"


def _cfg():
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def _reranker(cfg, want: bool):
    if not want:
        return None
    model = cfg["retrieval"].get("reranker_model") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rr = CrossEncoderReranker(model, cfg["retrieval"].get("reranker_blend", 0.7))
    print(f"  loading reranker: {model} …", flush=True)
    if not rr.available():
        print("  [warn] reranker failed to load — running WITHOUT it:\n    " + rr.last_error(),
              flush=True)
        return None
    print(f"  reranker ready (backend = {rr.backend()})", flush=True)
    return rr


def _bar():
    try:
        from tqdm import tqdm
        st = {"b": None}

        def prog(n, total):
            if st["b"] is None:
                st["b"] = tqdm(total=total, desc="  evaluating", unit="wo")
            st["b"].n = n
            st["b"].refresh()

        def close():
            if st["b"]:
                st["b"].n = st["b"].total
                st["b"].close()
        return prog, close
    except ImportError:
        def prog(n, total):
            print(f"    {n}/{total} ({100 * n // max(total, 1)}%)", flush=True)
        return prog, (lambda: None)


def _build(csv, normalize):
    print(f"  loading corpus ({os.path.basename(str(csv))}) …", flush=True)
    corpus = load_corpus(csv, normalize=normalize)
    print(f"  building retrieval index over {corpus.n:,} work orders …", flush=True)
    return corpus, Retriever(corpus.df)


def _run(corpus, retr, sample, reranker, query_ids, label):
    tag = label + (" + reranker" if reranker else "")
    n = "all" if query_ids is not None else (sample or "all")
    print(f"  running LOO on {n if query_ids is None else len(query_ids)} work orders "
          f"[{tag}]:", flush=True)
    prog, close = _bar()
    res = EV.evaluate(corpus.df, retr, sample=sample, reranker=reranker,
                      query_ids=query_ids, progress=prog)
    close()
    return res


def _matched_ids(dC, dR, sample, full, seed=0):
    idsD = set(dC.df[dC.df["action_family"] != "Other"]["ident"].astype(str))
    idsR = set(dR.df[dR.df["action_family"] != "Other"]["ident"].astype(str))
    shared = sorted(idsD & idsR)
    if not full and sample and len(shared) > sample:
        import random
        rng = random.Random(seed)
        shared = sorted(rng.sample(shared, sample))
    return shared


def _fmt(res, title):
    rr = "reranker ON" if res.get("reranker") else "reranker off"
    L = [f"### {title}  ({rr})", "",
         f"- evaluated: **{res['n_evaluated']:,}** (cluster-safe LOO)",
         f"- macro-recall: **{res['macro_recall']}%**  (majority {res['baseline_majority_macro']}%)",
         f"- top-1 agreement: **{res['top1_family_acc']}%**  (majority {res['baseline_majority_acc']}%)",
         f"- top-3 agreement: **{res['top3_family_acc']}%**   ·   MRR **{res['mrr']}**", "",
         "| tier | coverage | system acc | majority acc |", "|---|---|---|---|"]
    for t in ("strong", "moderate", "exploratory"):
        v = res["tiers"][t]
        L.append(f"| {t} | {v['coverage_pct']}% | {v['top1_acc']}% | {v['majority_acc']}% |")
    L += ["", "| action family | n | recall |", "|---|---|---|"]
    for f, v in sorted(res["per_family_recall"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(f"| {f} | {v['n']} | {v['recall']}% |")
    return "\n".join(L)


def _grid_table(results):
    """results: dict {(dataset, rr_on): res}. Compact side-by-side headline table."""
    L = ["", "## Summary grid", "",
         "| config | n | macro-recall | top-1 | top-3 | MRR | strong cov | strong acc |",
         "|---|---|---|---|---|---|---|---|"]
    for (ds, rr), r in results.items():
        L.append(f"| {ds}{' +rr' if rr else ''} | {r['n_evaluated']:,} | {r['macro_recall']}% | "
                 f"{r['top1_family_acc']}% | {r['top3_family_acc']}% | {r['mrr']} | "
                 f"{r['tiers']['strong']['coverage_pct']}% | {r['tiers']['strong']['top1_acc']}% |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sample", type=int, default=1200)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--reranker", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--all", action="store_true", dest="grid")
    args = ap.parse_args()
    sample = None if args.full else args.sample
    cfg = _cfg()
    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.grid or args.compare:
        print("Loading both datasets (matched query set) …")
        dC, rC = _build(D_CSV, normalize=False)
        dR, rR = _build(RAW_CSV, normalize=False)
        ids = _matched_ids(dC, dR, sample, args.full)
        print(f"  matched query set: {len(ids):,} work orders (identical for both datasets)\n")
        results, report = {}, {}
        rr_opts = [False, True] if args.grid else [args.reranker]
        rr_model = _reranker(cfg, any(rr_opts))
        for ds_name, corpus, retr in [("System-D", dC, rC), ("Raw", dR, rR)]:
            for rr_on in rr_opts:
                res = _run(corpus, retr, None, rr_model if rr_on else None, ids, ds_name)
                results[(ds_name, rr_on)] = res
                report[f"{ds_name}{'_rr' if rr_on else ''}"] = res
        # markdown
        md = [_grid_table(results), ""]
        for (ds, rr), r in results.items():
            md.append(_fmt(r, f"{ds}"))
            md.append("")
        d_key = results.get(("System-D", args.reranker if not args.grid else False))
        r_key = results.get(("Raw", args.reranker if not args.grid else False))
        if d_key and r_key:
            md.append(f"**Macro-recall delta (System-D − Raw, reranker off): "
                      f"{round(d_key['macro_recall'] - r_key['macro_recall'], 1)} pts**")
        if args.grid:
            for ds in ("System-D", "Raw"):
                off, on = results[(ds, False)], results[(ds, True)]
                md.append(f"**Reranker effect on {ds} (macro-recall): "
                          f"{round(on['macro_recall'] - off['macro_recall'], 1)} pts** "
                          f"({off['macro_recall']}% → {on['macro_recall']}%)")
        md = "\n".join(md)
    else:
        csv = args.csv or (ROOT / cfg["data"]["csv_path"])
        normalize = cfg["extraction"].get("normalize", False) if not args.csv else True
        corpus, retr = _build(csv, normalize=normalize)
        rr_model = _reranker(cfg, args.reranker)
        res = _run(corpus, retr, sample, rr_model, None, os.path.basename(str(csv)))
        report = res
        md = _fmt(res, os.path.basename(str(csv)))

    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "eval_report.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nSaved: {out_dir / 'eval_report.json'}  and  eval_report.md")


if __name__ == "__main__":
    main()

