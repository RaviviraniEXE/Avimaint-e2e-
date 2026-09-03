"""Prepare the final 6,169-record operational Selective-ByT5 corpus for SpERT.

Operational design decision
---------------------------
The controlled RQ1 experiment remains frozen and unchanged.  For the final DSS
pipeline, Selective ByT5 is used as an engineering trade-off: it is a
human-readable normalized representation, uses the safety/fallback mechanism,
and was the strongest normalized SpERT condition on relation micro/macro among
the evaluated normalized variants.  This is an operational selection, not a
claim that it statistically outperforms raw text.

Safety gates
------------
1. exactly 6,169 unique records;
2. deterministic operational tokenizer;
3. tokenizer parity with ALL 1,600 projected Selective-ByT5 gold records;
4. projected gold tokens must exactly match the 1,600 tokens actually exported
   to the trained Selective-ByT5 SpERT train/dev/test files;
5. original raw PROBLEM/ACTION are retained in the index for provenance.
"""
import _bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from src.data.corpus import load, representation_path


REPRESENTATION = "selective_byt5"

# Matches the token conventions seen in the projected SpERT variants:
# - slash compounds such as R/H, L/H, A/C stay intact
# - hyphen compounds such as left-hand stay intact
# - #, +, apostrophes and punctuation remain separate tokens
TOKEN_RE = re.compile(
    # Keep internal dot/slash/hyphen compounds intact when punctuation is
    # between alphanumeric segments: 12.1, CLEARNACE.035, R/H, A/C, 3-4.
    # Sentence-final punctuation remains a separate token.
    r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)+|"
    r"[A-Za-z0-9]+|"
    r"[^\w\s]",
    flags=re.UNICODE,
)


def operational_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_projected_gold() -> dict[str, dict]:
    folder = Path(f"outputs/gold_variants/{REPRESENTATION}")
    if not folder.exists():
        raise SystemExit(f"Projected gold folder missing: {folder}")

    records = {}
    for path in sorted(folder.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = str(rec.get("ident", "")).strip()
                if not rid:
                    raise SystemExit(f"{path}:{line_no}: missing ident")
                if rid in records:
                    raise SystemExit(
                        f"Duplicate projected gold IDENT: {rid}"
                    )
                records[rid] = rec
    return records


def load_trained_export_tokens() -> tuple[dict[str, list[str]], dict[str, int]]:
    folder = Path(f"outputs/spert_normalized/{REPRESENTATION}")
    exported = {}
    split_counts = {}
    for split in ("train", "dev", "test"):
        path = folder / f"{split}.json"
        if not path.exists():
            raise SystemExit(
                f"Trained Selective-ByT5 SpERT export missing: {path}"
            )
        docs = json.loads(path.read_text(encoding="utf-8-sig"))
        split_counts[split] = len(docs)
        for doc in docs:
            rid = str(
                doc.get("orig_id", doc.get("ident", ""))
            ).strip()
            if not rid:
                raise SystemExit(f"{path}: document without orig_id/ident")
            if rid in exported:
                raise SystemExit(
                    f"Duplicate IDENT across trained SpERT exports: {rid}"
                )
            exported[rid] = doc.get("tokens")
    return exported, split_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-records", type=int, default=6169)
    ap.add_argument("--expected-gold-records", type=int, default=1600)
    args = ap.parse_args()

    os.makedirs("outputs/kg", exist_ok=True)

    # Explicit: final operational representation is Selective ByT5.
    df, _, text_map, stats = load(
        representation=REPRESENTATION
    )

    if len(df) != args.expected_records:
        raise SystemExit(
            f"Canonical corpus rows={len(df)}, expected={args.expected_records}"
        )
    if df["IDENT"].astype(str).duplicated().any():
        raise SystemExit("Canonical corpus contains duplicate IDENTs.")

    projected = load_projected_gold()
    if len(projected) != args.expected_gold_records:
        raise SystemExit(
            f"Expected {args.expected_gold_records} projected "
            f"{REPRESENTATION} gold records, found {len(projected)}"
        )

    exported, split_counts = load_trained_export_tokens()
    if len(exported) != args.expected_gold_records:
        raise SystemExit(
            f"Expected {args.expected_gold_records} records in the trained "
            f"{REPRESENTATION} SpERT export, found {len(exported)}"
        )

    # Gate A: projected variant must be the exact token representation the
    # already-trained Selective-ByT5 SpERT saw.
    export_bad = []
    for rid, rec in projected.items():
        if rid not in exported or rec.get("tokens") != exported[rid]:
            export_bad.append({
                "ident": rid,
                "projected": rec.get("tokens"),
                "trained_spert_export": exported.get(rid),
            })
            if len(export_bad) >= 5:
                break
    if export_bad:
        raise SystemExit(
            "PROJECTED/TRAINED SPERT TOKEN PARITY FAILED. "
            "Do not run full-corpus inference.\n"
            + json.dumps(export_bad, ensure_ascii=False, indent=2)
        )

    # Gate B: the tokenizer used for all 6,169 records must exactly reproduce
    # every projected/trained token sequence for all 1,600 annotated records.
    token_bad = []
    mismatch_total = 0
    for rid, rec in projected.items():
        if rid not in text_map:
            raise SystemExit(
                f"Projected {REPRESENTATION} IDENT missing from full corpus: "
                f"{rid}"
            )
        tokens = operational_tokenize(text_map[rid])
        if tokens != rec.get("tokens"):
            mismatch_total += 1
            if len(token_bad) < 10:
                token_bad.append({
                    "ident": rid,
                    "projected_and_trained_tokens": rec.get("tokens"),
                    "operational_tokenizer": tokens,
                    "text": text_map[rid],
                })

    if mismatch_total:
        raise SystemExit(
            "SELECTIVE-BYT5 TOKENIZER PARITY FAILED against the exact "
            f"trained SpERT representation for {mismatch_total}/{len(projected)} "
            "annotated records. DO NOT run full-corpus inference.\n"
            + json.dumps(token_bad, ensure_ascii=False, indent=2)
        )

    dataset = []
    index = []
    seen = set()
    skipped_empty = 0

    for _, row in df.iterrows():
        ident = str(row["IDENT"]).strip()
        text = str(row.get("text") or "").strip()
        if not ident:
            raise SystemExit("Encountered empty IDENT.")
        if ident in seen:
            raise SystemExit(f"Duplicate IDENT during preparation: {ident}")
        seen.add(ident)

        if not text:
            skipped_empty += 1
            continue

        tokens = operational_tokenize(text)
        if not tokens:
            raise SystemExit(
                f"Operational text produced no tokens for IDENT={ident}"
            )

        problem_raw = str(row.get("PROBLEM", ""))
        action_raw = str(row.get("ACTION", ""))

        dataset.append({
            "tokens": tokens,
            "entities": [],
            "relations": [],
        })
        index.append({
            "ident": ident,
            "tokens": tokens,
            "representation": REPRESENTATION,
            "normalization_system": "Selective ByT5",
            "normalized_text": text,
            "problem_raw": problem_raw,
            "action_raw": action_raw,
            "source_raw_text": (
                (problem_raw.strip() + " " + action_raw.strip()).strip()
            ),
        })

    if skipped_empty:
        raise SystemExit(
            f"Found {skipped_empty} empty operational records; "
            "expected zero."
        )
    if len(dataset) != args.expected_records:
        raise SystemExit(
            f"Prepared {len(dataset)} records, expected "
            f"{args.expected_records}"
        )
    if len(seen) != args.expected_records:
        raise SystemExit(
            f"Unique IDENTs={len(seen)}, expected {args.expected_records}"
        )

    dataset_path = Path("outputs/kg/full_corpus_spert.json")
    index_path = Path("outputs/kg/full_index.jsonl")
    manifest_path = Path("outputs/kg/full_corpus_manifest.json")

    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False), encoding="utf-8"
    )
    with index_path.open("w", encoding="utf-8") as target:
        for row in index:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_path = representation_path(REPRESENTATION)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "operational_selection": True,
        "representation": REPRESENTATION,
        "normalization_system": "Selective ByT5",
        "representation_source": str(source_path),
        "representation_source_sha256": sha256(source_path),
        "prepared_records": len(dataset),
        "unique_identifiers": len(seen),
        "skipped_empty": skipped_empty,
        "raw_source_retained_in_index": True,
        "tokenizer": {
            "name": "operational_maintenance_tokenizer_v2",
            "pattern": TOKEN_RE.pattern,
            "preserves_slash_compounds": True,
            "preserves_hyphen_compounds": True,
            "preserves_decimal_and_dot_compounds": True,
        },
        "representation_parity": {
            "status": "pass",
            "projected_gold_records": len(projected),
            "trained_spert_export_records": len(exported),
            "projected_equals_trained_export": len(projected),
            "operational_tokenizer_matches": len(projected),
            "mismatches": 0,
            "trained_split_counts": split_counts,
            "projected_reference": (
                f"outputs/gold_variants/{REPRESENTATION}"
            ),
            "trained_export_reference": (
                f"outputs/spert_normalized/{REPRESENTATION}"
            ),
        },
        "prediction_input": str(dataset_path),
        "prediction_input_sha256": sha256(dataset_path),
        "token_index": str(index_path),
        "token_index_sha256": sha256(index_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("FINAL OPERATIONAL SELECTIVE-BYT5 CORPUS PREPARED")
    print("=" * 92)
    print("representation            : selective_byt5")
    print("prepared records          :", len(dataset))
    print("unique identifiers        :", len(seen))
    print("raw source retained       : YES")
    print(
        "projected == trained     : "
        f"PASS ({len(projected)}/{len(projected)})"
    )
    print(
        "tokenizer == trained     : "
        f"PASS ({len(projected)}/{len(projected)})"
    )
    print("manifest                  :", manifest_path)
    print("=" * 92)


if __name__ == "__main__":
    main()
