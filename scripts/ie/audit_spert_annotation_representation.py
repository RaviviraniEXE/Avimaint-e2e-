"""Audit the representation used by the authoritative aviation IE gold and SpERT baseline.

This script proves whether the historical outputs/spert baseline was trained on
true source-raw text or on the legacy normalized annotation representation.

Required safety condition for this project:
- all 1,600 gold records must tokenize exactly like
  data/aviation/processed/normalized_corpus.csv::normalized
- the exported outputs/spert/{train,dev,test}.json must exactly reproduce the
  same 1,600 gold token sequences and frozen IDs

The result is written as a provenance artifact.  No model is trained.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
REPORT = IE / "outputs" / "reports" / "normalization_spert_matched_v2"
NORM = ROOT / "data" / "aviation" / "processed" / "normalized_corpus.csv"

import sys
sys.path.insert(0, str(IE))
from src.data.preannotate import tokenize  # noqa: E402


def load_gold() -> dict[str, dict]:
    records = {}
    for path in sorted((IE / "outputs" / "gold").glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = str(rec.get("ident", "")).strip()
                if not rid:
                    raise SystemExit(f"{path}:{line_no}: gold record has no ident")
                if rid in records:
                    raise SystemExit(f"Duplicate gold IDENT {rid}")
                records[rid] = rec
    return records


def main():
    if not NORM.exists():
        raise SystemExit(f"Missing normalized corpus: {NORM}")

    df = pd.read_csv(
        NORM, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    required = {"IDENT", "raw", "normalized"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"normalized_corpus.csv missing columns: {missing}")

    if df["IDENT"].astype(str).duplicated().any():
        raise SystemExit("Duplicate IDENT values in normalized_corpus.csv")

    norm_map = dict(zip(df["IDENT"].astype(str), df["normalized"]))
    gold = load_gold()
    if len(gold) != 1600:
        raise SystemExit(f"Expected 1600 gold records, found {len(gold)}")

    normalized_matches = 0
    normalized_bad = []
    for rid, rec in gold.items():
        if rid not in norm_map:
            raise SystemExit(f"Gold IDENT missing from normalized corpus: {rid}")
        ntoks = [t for t, _, _ in tokenize(norm_map[rid])]
        if ntoks == rec["tokens"]:
            normalized_matches += 1
        elif len(normalized_bad) < 5:
            normalized_bad.append({
                "ident": rid,
                "gold": rec["tokens"],
                "normalized": ntoks,
            })

    if normalized_matches != 1600:
        raise SystemExit(
            "Annotation-representation audit failed: "
            f"normalized-token matches={normalized_matches}/1600; "
            f"examples={normalized_bad}"
        )

    exported = {}
    export_matches = 0
    export_bad = []
    export_ids = set()
    for split in ("train", "dev", "test"):
        path = IE / "outputs" / "spert" / f"{split}.json"
        if not path.exists():
            raise SystemExit(f"Missing historical SpERT export: {path}")
        docs = json.loads(path.read_text(encoding="utf-8"))
        exported[split] = len(docs)
        for doc in docs:
            rid = str(doc.get("orig_id", doc.get("ident", ""))).strip()
            if not rid:
                raise SystemExit(f"{path}: document without orig_id/ident")
            if rid in export_ids:
                raise SystemExit(f"Duplicate IDENT across SpERT exports: {rid}")
            export_ids.add(rid)
            if rid not in gold:
                raise SystemExit(f"SpERT export IDENT is not in gold: {rid}")
            if doc.get("tokens") == gold[rid]["tokens"]:
                export_matches += 1
            elif len(export_bad) < 5:
                export_bad.append({
                    "ident": rid,
                    "gold": gold[rid]["tokens"],
                    "spert_export": doc.get("tokens"),
                })

    if export_matches != 1600 or len(export_ids) != 1600:
        raise SystemExit(
            "Historical SpERT export does not reproduce all gold records: "
            f"token_matches={export_matches}/1600 unique_ids={len(export_ids)}; "
            f"examples={export_bad}"
        )

    REPORT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "pass",
        "gold_records": 1600,
        "normalized_corpus_normalized_token_matches": normalized_matches,
        "historical_spert_export_token_matches": export_matches,
        "historical_spert_export_split_counts": exported,
        "conclusion": (
            "The historical outputs/spert baseline was trained/evaluated on the "
            "legacy normalized annotation representation, not on true source-raw text."
        ),
        "rq1_correction": (
            "The five-way matched normalization ablation therefore requires a "
            "new System-A raw SpERT model trained on outputs/gold_variants/raw. "
            "The four already-trained rules/byt5/selective_byt5/rules_then_byt5 "
            "models remain valid and are reused without retraining."
        ),
    }
    out = REPORT / "REPRESENTATION_AUDIT.json"
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 86)
    print("ANNOTATION REPRESENTATION AUDIT: PASS")
    print("=" * 86)
    print("gold records                              : 1600")
    print("gold == normalized_corpus.normalized      : 1600/1600")
    print("gold == historical SpERT export tokens    : 1600/1600")
    print("historical SpERT split counts             :", exported)
    print()
    print("CONCLUSION:")
    print("  outputs/spert is an annotation-representation baseline, NOT true System-A raw.")
    print("  Only the missing true-raw SpERT condition must be trained.")
    print("audit ->", out)
    print("=" * 86)


if __name__ == "__main__":
    main()
