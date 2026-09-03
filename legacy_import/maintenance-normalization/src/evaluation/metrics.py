"""Normalization metrics: intrinsic (no gold) and extrinsic (vs gold + ERR)."""
from __future__ import annotations

from typing import Dict, Iterable, List

from src.utils.text import simple_tokens


def type_token_stats(texts: Iterable[str]) -> Dict[str, int]:
    types = set()
    n = 0
    for t in texts:
        toks = simple_tokens(t)
        n += len(toks)
        types.update(toks)
    return {"n_tokens": n, "vocab_size": len(types)}


def oov_rate(texts: Iterable[str], vocab: set) -> float:
    total = oov = 0
    for t in texts:
        for tok in simple_tokens(t):
            if tok.isdigit():
                continue
            total += 1
            if tok not in vocab:
                oov += 1
    return oov / total if total else 0.0


def intrinsic_report(raw, norm, stats, vocab) -> Dict[str, float]:
    rt, nt = type_token_stats(raw), type_token_stats(norm)
    n_exp = sum(s.get("n_expansions", 0) for s in stats)
    n_tok = sum(s.get("n_tokens", 0) for s in stats) or nt["n_tokens"]
    r_oov, n_oov = oov_rate(raw, vocab), oov_rate(norm, vocab)
    red = lambda b, a: ((b - a) / b) if b else 0.0
    return {
        "records": len(raw),
        "raw_vocab": rt["vocab_size"],
        "norm_vocab": nt["vocab_size"],
        "vocab_reduction_pct": round(100 * red(rt["vocab_size"], nt["vocab_size"]), 2),
        "raw_oov": round(r_oov, 4),
        "norm_oov": round(n_oov, 4),
        "oov_reduction_pct": round(100 * red(r_oov, n_oov), 2),
        "expansions": n_exp,
        "expansion_rate": round(n_exp / n_tok, 4) if n_tok else 0.0,
    }


def extrinsic_report(pred, gold, baseline=None) -> Dict[str, float]:
    try:
        import jiwer
    except ImportError as e:  # pragma: no cover
        raise ImportError("Extrinsic metrics need `pip install jiwer`.") from e
    clean = lambda xs: [x if x and x.strip() else " " for x in xs]
    p, g = clean(pred), clean(gold)
    wer, cer = jiwer.wer(g, p), jiwer.cer(g, p)
    exact = sum(1 for a, b in zip(p, g) if a.strip() == b.strip()) / len(g)
    rep = {"records": len(g), "wer": round(wer, 4), "cer": round(cer, 4),
           "exact_match": round(exact, 4)}
    if baseline is not None:
        b = clean(baseline)
        bw, bc = jiwer.wer(g, b), jiwer.cer(g, b)
        rep["err_word"] = round((bw - wer) / bw, 4) if bw else 0.0
        rep["err_char"] = round((bc - cer) / bc, 4) if bc else 0.0
    return rep

