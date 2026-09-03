"""Shared corpus loading: raw logbook + normalized text + dedup + unique pool."""
from __future__ import annotations

import glob
import json

import pandas as pd

from src.data.dedup import add_duplicate_groups, unique_pool


def load(raw="data/raw/Aircraft_Annotation_DataFile.csv",
         norm="data/raw/normalized_corpus.csv"):
    df = pd.read_csv(raw, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    nm = pd.read_csv(norm, dtype=str, keep_default_na=False)
    norm_map = dict(zip(nm["IDENT"], nm["normalized"]))
    df, stats = add_duplicate_groups(df)
    df["text"] = df["IDENT"].map(norm_map).fillna("")
    pool = unique_pool(df)
    pool["text"] = pool["IDENT"].map(norm_map).fillna("")
    return df, pool, norm_map, stats


def annotated_idents(gold_glob="outputs/gold/*.jsonl"):
    ids = set()
    for f in glob.glob(gold_glob):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                ids.add(str(json.loads(line).get("ident")))
    return ids

