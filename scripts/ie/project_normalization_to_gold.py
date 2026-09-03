"""Project frozen gold spans onto full-corpus normalization outputs.

HOTFIX:
- primary systems are raw, rules, byt5, selective_byt5, rules_then_byt5
- missing prediction files / IDs are fatal (no silent raw-text fallback)
- duplicate record IDs in prediction CSVs are fatal
- relations are retained only when both endpoint entities project successfully
- row-level projection QC is always written
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import glob
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avimaint.normalization.rules import load_replacements, normalize_rules

REPLACEMENTS = load_replacements(ROOT / "data/dictionaries/abbreviations.yaml")

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[./#-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)


def tokens_with_offsets(text: str):
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def boundary_map(source: str, target: str) -> list[int]:
    mapping = [0] * (len(source) + 1)
    matcher = SequenceMatcher(None, source.casefold(), target.casefold(), autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if a1 == a0:
            continue
        if tag == "equal":
            for offset in range(a1 - a0 + 1):
                mapping[a0 + offset] = b0 + offset
        else:
            for offset in range(a1 - a0 + 1):
                mapping[a0 + offset] = round(b0 + (b1 - b0) * offset / (a1 - a0))
    mapping[0] = 0
    mapping[-1] = len(target)
    for index in range(1, len(mapping)):
        mapping[index] = max(mapping[index], mapping[index - 1])
    return mapping


def project_record(record: dict, target: str):
    source_tokens = record["tokens"]
    source = " ".join(source_tokens)
    source_offsets = []
    cursor = 0
    for token in source_tokens:
        source_offsets.append((cursor, cursor + len(token)))
        cursor += len(token) + 1

    target_offsets = tokens_with_offsets(target)
    target_tokens = [x[0] for x in target_offsets]

    # Align through the conservative canonical form. This handles reversible
    # abbreviation changes (#4 <-> number 4, CK <-> check) more reliably
    # than a direct whole-sentence character diff.
    canonical = normalize_rules(target, REPLACEMENTS)
    source_to_canonical = boundary_map(source, canonical)
    canonical_to_target = boundary_map(canonical, target)

    projected = []
    old_to_new = {}
    failures = []
    for old_index, entity in enumerate(record.get("entities", [])):
        if entity["start"] >= len(source_offsets) or entity["end"] <= entity["start"]:
            failures.append(old_index)
            continue
        char_start = source_offsets[entity["start"]][0]
        char_end = source_offsets[entity["end"] - 1][1]
        left = canonical_to_target[source_to_canonical[char_start]]
        right = canonical_to_target[source_to_canonical[char_end]]
        indices = [
            i for i, (_, s, e) in enumerate(target_offsets)
            if e > left and s < right
        ]
        if not indices:
            surface = " ".join(
                source_tokens[entity["start"]:entity["end"]]
            ).casefold()
            hit = target.casefold().find(surface)
            indices = [
                i for i, (_, s, e) in enumerate(target_offsets)
                if hit >= 0 and e > hit and s < hit + len(surface)
            ]
        if not indices:
            failures.append(old_index)
            continue
        old_to_new[old_index] = len(projected)
        projected.append(
            {
                "type": entity["type"],
                "start": min(indices),
                "end": max(indices) + 1,
            }
        )

    relations = []
    dropped_rel = 0
    for relation in record.get("relations", []):
        if relation["head"] in old_to_new and relation["tail"] in old_to_new:
            relations.append(
                {
                    "type": relation["type"],
                    "head": old_to_new[relation["head"]],
                    "tail": old_to_new[relation["tail"]],
                }
            )
        else:
            dropped_rel += 1

    output = dict(record)
    output["tokens"] = target_tokens
    output["entities"] = projected
    output["relations"] = relations

    total = len(record.get("entities", []))
    coverage = len(projected) / total if total else 1.0

    output["bio"] = ["O"] * len(target_tokens)
    for entity in projected:
        output["bio"][entity["start"]] = f"B-{entity['type']}"
        for index in range(entity["start"] + 1, entity["end"]):
            if output["bio"][index] == "O":
                output["bio"][index] = f"I-{entity['type']}"

    output["normalization_projection"] = {
        "source_text": source,
        "target_text": target,
        "entity_coverage": coverage,
        "dropped_entities": len(failures),
        "dropped_relations": dropped_rel,
    }
    return output, coverage, len(failures), dropped_rel


def _load_gold_paths(root: Path, pattern: str):
    paths = sorted(glob.glob(str(root / pattern)))
    if not paths:
        raise SystemExit("No frozen gold JSONL files found")
    return [Path(p) for p in paths]


def _gold_ids(paths):
    ids = []
    for path in paths:
        with path.open(encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                rid = str(record.get("ident", ""))
                if not rid:
                    raise SystemExit(f"Gold record without ident in {path}")
                ids.append(rid)
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate ident values found across frozen gold JSONL files")
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        default="legacy_import/maintenance-ie/outputs/gold/*.jsonl",
    )
    parser.add_argument(
        "--pred-root",
        default="outputs/normalization/full_corpus",
    )
    parser.add_argument(
        "--out-root",
        default="legacy_import/maintenance-ie/outputs/gold_variants",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["raw", "rules", "byt5", "selective_byt5", "rules_then_byt5"],
    )
    parser.add_argument("--min-coverage", type=float, default=0.97)
    args = parser.parse_args()

    root = Path.cwd()
    gold_files = _load_gold_paths(root, args.gold)
    gold_ids = _gold_ids(gold_files)
    gold_id_set = set(gold_ids)

    qc = []
    failed_coverage = []

    for system in args.systems:
        csv_path = root / args.pred_root / f"{system}.csv"
        if not csv_path.exists():
            raise SystemExit(f"Missing normalization prediction file: {csv_path}")

        predictions = pd.read_csv(csv_path, dtype=str).fillna("")
        required = {"record_id", "prediction_text"}
        missing_cols = required - set(predictions.columns)
        if missing_cols:
            raise SystemExit(
                f"{csv_path} is missing required columns: {sorted(missing_cols)}"
            )

        predictions["record_id"] = predictions["record_id"].astype(str)
        dupes = predictions.loc[
            predictions["record_id"].duplicated(keep=False), "record_id"
        ].unique().tolist()
        if dupes:
            raise SystemExit(
                f"{csv_path} contains duplicate record_id values; examples: {dupes[:10]}"
            )

        by_id = dict(
            zip(
                predictions["record_id"],
                predictions["prediction_text"].astype(str),
            )
        )
        missing_ids = sorted(gold_id_set - set(by_id))
        if missing_ids:
            raise SystemExit(
                f"{system}: missing predictions for {len(missing_ids)} frozen gold IDs; "
                f"examples={missing_ids[:10]}. Refusing silent raw-text fallback."
            )

        system_total = 0
        system_projected = 0

        for path in gold_files:
            out_dir = root / args.out_root / system
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / path.name

            with path.open(encoding="utf-8") as src, out_path.open(
                "w", encoding="utf-8"
            ) as dst:
                for line in src:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    rid = str(record.get("ident", ""))
                    target = by_id[rid]  # strict: already verified present
                    projected, coverage, dropped_e, dropped_r = project_record(
                        record, target
                    )
                    system_total += len(record.get("entities", []))
                    system_projected += len(projected.get("entities", []))
                    qc.append(
                        {
                            "system": system,
                            "record_id": rid,
                            "entity_coverage": coverage,
                            "dropped_entities": dropped_e,
                            "dropped_relations": dropped_r,
                            "prediction_found": True,
                        }
                    )
                    dst.write(json.dumps(projected, ensure_ascii=False) + "\n")

        aggregate = system_projected / max(1, system_total)
        print(
            f"{system}: projected {system_projected}/{system_total} "
            f"entities ({aggregate:.2%})"
        )
        if aggregate < args.min_coverage:
            failed_coverage.append((system, aggregate))

    qc_path = root / args.out_root / "projection_qc.csv"
    pd.DataFrame(qc).to_csv(qc_path, index=False)
    print(f"QC -> {qc_path}")

    if failed_coverage:
        raise SystemExit(
            "Projection coverage below threshold: "
            + ", ".join(f"{s}={v:.2%}" for s, v in failed_coverage)
        )


if __name__ == "__main__":
    main()
