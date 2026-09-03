"""Shared aviation corpus loader with explicit representation selection.

Historical reproducibility:
    load() == load(representation="normalized")

Operational full-corpus extraction:
    load(representation="selective_byt5")

The operational representation is read from the already-produced normalization
full-corpus artifacts.  This keeps the original source PROBLEM/ACTION columns
in the dataframe while exposing the chosen normalized text as ``text``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from src.data.dedup import add_duplicate_groups, unique_pool


LEGACY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = LEGACY_ROOT.parents[1]

CANONICAL_RAW = (
    PROJECT_ROOT / "data" / "aviation" / "raw" /
    "Aircraft_Annotation_DataFile.csv"
)
CANONICAL_NORM = (
    PROJECT_ROOT / "data" / "aviation" / "processed" /
    "normalized_corpus.csv"
)
FULL_CORPUS_ROOT = (
    PROJECT_ROOT / "outputs" / "normalization" / "full_corpus"
)

SYSTEM_FILES = {
    "raw": "raw.csv",
    "rules": "rules.csv",
    "byt5": "byt5.csv",
    "selective_byt5": "selective_byt5.csv",
    "rules_then_byt5": "rules_then_byt5.csv",
}


def _resolve_path(requested: str | Path, canonical: Path, kind: str) -> Path:
    requested = Path(requested)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(LEGACY_ROOT / requested)
    candidates.append(canonical)

    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not locate {kind} corpus file. Tried:\n  - {tried}\n"
        f"Expected canonical path:\n  - {canonical}"
    )


def resolve_corpus_paths(
    raw: str | Path = "data/raw/Aircraft_Annotation_DataFile.csv",
    norm: str | Path = "data/raw/normalized_corpus.csv",
) -> tuple[Path, Path]:
    return (
        _resolve_path(raw, CANONICAL_RAW, "raw aviation"),
        _resolve_path(norm, CANONICAL_NORM, "normalized aviation"),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    frame.columns = [c.strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def representation_path(representation: str) -> Path:
    if representation not in SYSTEM_FILES:
        raise ValueError(
            f"No full-corpus file is defined for representation "
            f"{representation!r}."
        )
    return FULL_CORPUS_ROOT / SYSTEM_FILES[representation]


def _load_system_map(
    representation: str,
    canonical_ids: set[str],
) -> dict[str, str]:
    path = representation_path(representation)
    if not path.is_file():
        raise FileNotFoundError(
            f"Full-corpus normalization artifact missing for "
            f"{representation}: {path}"
        )

    frame = _read_csv(path)
    required = {"record_id", "prediction_text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing}"
        )

    ids = frame["record_id"].astype(str)
    if ids.duplicated().any():
        examples = ids[ids.duplicated()].head(5).tolist()
        raise ValueError(
            f"Duplicate record_id values in {path}; examples={examples}"
        )

    if len(frame) != 6169:
        raise ValueError(
            f"{representation} full-corpus artifact contains "
            f"{len(frame)} records, expected 6169."
        )

    if "system" in frame.columns:
        observed = set(frame["system"].astype(str).str.strip())
        aliases = {
            "raw": {"raw"},
            "rules": {"rules"},
            "byt5": {"byt5"},
            "selective_byt5": {"selective_byt5"},
            "rules_then_byt5": {"rules_then_byt5"},
        }
        if not observed.issubset(aliases[representation]):
            raise ValueError(
                f"Unexpected system values in {path}: {sorted(observed)}"
            )

    system_ids = set(ids)
    if system_ids != canonical_ids:
        raise ValueError(
            f"{representation} IDENT coverage differs from canonical corpus: "
            f"missing={len(canonical_ids-system_ids)} "
            f"extra={len(system_ids-canonical_ids)}"
        )

    return dict(zip(ids, frame["prediction_text"].astype(str)))


def load(
    raw: str | Path = "data/raw/Aircraft_Annotation_DataFile.csv",
    norm: str | Path = "data/raw/normalized_corpus.csv",
    *,
    representation: str = "normalized",
):
    """Load aviation data and expose an explicit text representation.

    ``normalized`` preserves the legacy annotation-era behavior:
        data/aviation/processed/normalized_corpus.csv::normalized

    The evaluated operational systems use:
        outputs/normalization/full_corpus/<system>.csv::prediction_text

    Supported system representations:
        raw, rules, byt5, selective_byt5, rules_then_byt5

    Returns the historical 4-tuple:
        (df, pool, text_map, stats)
    """
    raw_path, norm_path = resolve_corpus_paths(raw, norm)

    df = _read_csv(raw_path)
    nm = _read_csv(norm_path)

    if "IDENT" not in df.columns:
        raise ValueError(f"Raw corpus has no IDENT column: {raw_path}")
    if not {"IDENT", "normalized"}.issubset(nm.columns):
        raise ValueError(
            f"Historical normalized corpus requires IDENT + normalized: "
            f"{norm_path}"
        )

    raw_ids_series = df["IDENT"].astype(str)
    norm_ids_series = nm["IDENT"].astype(str)

    if raw_ids_series.duplicated().any():
        raise ValueError("Canonical raw corpus contains duplicate IDENTs.")
    if norm_ids_series.duplicated().any():
        raise ValueError("Historical normalized corpus contains duplicate IDENTs.")

    canonical_ids = set(raw_ids_series)
    if set(norm_ids_series) != canonical_ids:
        raise ValueError(
            "Raw and historical normalized IDENT sets differ."
        )

    if representation == "normalized":
        text_map = dict(
            zip(norm_ids_series, nm["normalized"].astype(str))
        )
    elif representation in SYSTEM_FILES:
        text_map = _load_system_map(representation, canonical_ids)
    else:
        allowed = ["normalized", *SYSTEM_FILES.keys()]
        raise ValueError(
            f"Unknown representation={representation!r}; "
            f"expected one of {allowed}"
        )

    df, stats = add_duplicate_groups(df)
    df["text"] = df["IDENT"].astype(str).map(text_map).fillna("")
    pool = unique_pool(df)
    pool["text"] = pool["IDENT"].astype(str).map(text_map).fillna("")

    return df, pool, text_map, stats


def annotated_idents(gold_glob="outputs/gold/*.jsonl"):
    ids = set()
    for f in glob.glob(gold_glob):
        with open(f, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    ids.add(str(json.loads(line).get("ident")))
    return ids
