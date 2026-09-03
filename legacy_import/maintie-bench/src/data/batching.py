"""Split annotation records into fixed-size batches for correction workflow."""
from __future__ import annotations

import json
import os
from typing import List


def write_batches(records: List[dict], out_dir: str, prefix: str, batch_size: int):
    """Write records into JSONL batches + a human-review .txt per batch.
    Returns list of batch file paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for b, start in enumerate(range(0, len(records), batch_size), 1):
        chunk = records[start:start + batch_size]
        jpath = os.path.join(out_dir, f"{prefix}_batch{b:02d}.jsonl")
        tpath = os.path.join(out_dir, f"{prefix}_batch{b:02d}_review.txt")
        with open(jpath, "w", encoding="utf-8") as jf, open(tpath, "w", encoding="utf-8") as tf:
            for r in chunk:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
                tf.write(f"IDENT {r.get('ident','?')}   [{r.get('stratum','')}]\n")
                tf.write("TEXT : " + " ".join(r["tokens"]) + "\n")
                spans = [f"{r['tokens'][e['start']] if e['end']-e['start']==1 else ' '.join(r['tokens'][e['start']:e['end']])} = {e['type']}"
                         for e in r["entities"]]
                tf.write("SPANS: " + ("; ".join(spans) if spans else "(none pre-labeled)") + "\n")
                tf.write("-" * 70 + "\n")
        paths.append(jpath)
    return paths

