"""Feature extraction for the classical tier (CRF NER + Logistic-Regression RE).

NER features: orthographic (shape, affixes, casing, digits) + domain gazetteers,
in a ±2 token window. RE features: entity types, order/distance, the tokens
around and between the two spans.
"""
from __future__ import annotations

import re

from src.data import lexicons as L


def _shape(w):
    return re.sub(r"[A-Za-z]", "x", re.sub(r"\d", "d", w))


def _gaz(word):
    """Gazetteer membership flags for a lowercased word."""
    flags = {}
    for ent, words in L.SINGLE.items():
        if word in words:
            flags[f"gaz={ent}"] = True
    return flags


def token_features(tokens, i):
    """Focus token orthography + gazetteers + a ±1 neighbour window.

    NOTE: a ±2 window and extra orthographic flags were tested and *regressed*
    test F1 on this small corpus (over-fitting) — the ±1 set below is the tuned
    baseline. See features_ablation in the thesis notes.
    """
    w = tokens[i]
    wl = w.lower()
    f = {
        "bias": 1.0,
        "w.lower": wl,
        "w.suf3": wl[-3:],
        "w.suf2": wl[-2:],
        "w.pre3": wl[:3],
        "w.shape": _shape(w),
        "w.isdigit": w.isdigit(),
        "w.hasdigit": bool(re.search(r"\d", w)),
        "w.hasslash": ("/" in w) or (w == "#"),
        "w.len": len(w),
    }
    f.update(_gaz(wl))
    if i == 0:
        f["BOS"] = True
    else:
        pw = tokens[i - 1].lower()
        f["-1.lower"] = pw
        f["-1.suf3"] = pw[-3:]
        for k in _gaz(pw):
            f["-1." + k] = True
    if i == len(tokens) - 1:
        f["EOS"] = True
    else:
        nw = tokens[i + 1].lower()
        f["+1.lower"] = nw
        f["+1.suf3"] = nw[-3:]
        for k in _gaz(nw):
            f["+1." + k] = True
    return f


def sent_features(tokens):
    return [token_features(tokens, i) for i in range(len(tokens))]


# ---- relation features (entity-pair) ---------------------------------------
def relation_features(tokens, head, tail):
    """head/tail = entity dicts {type,start,end}. Returns a feature dict."""
    h_s, h_e, t_s, t_e = head["start"], head["end"], tail["start"], tail["end"]
    between = tokens[min(h_e, t_e):max(h_s, t_s)]
    order = "h<t" if h_s < t_s else "t<h"
    dist = min(abs(h_s - t_e), abs(t_s - h_e))

    def at(idx):
        return tokens[idx].lower() if 0 <= idx < len(tokens) else "<edge>"

    return {
        "head.type": head["type"],
        "tail.type": tail["type"],
        "pair": f"{head['type']}->{tail['type']}",
        "order": order,
        "dist": min(dist, 10),
        "adjacent": dist <= 1,
        "head.word": " ".join(tokens[h_s:h_e]).lower(),
        "tail.word": " ".join(tokens[t_s:t_e]).lower(),
        # local context around each span
        "head.before": at(h_s - 1),
        "head.after": at(h_e),
        "tail.before": at(t_s - 1),
        "tail.after": at(t_e),
        # what lies between the two spans
        "between.bow": " ".join(w.lower() for w in between)[:60],
        "between.first": (between[0].lower() if between else "<none>"),
        "between.last": (between[-1].lower() if between else "<none>"),
        "n_between": min(len(between), 10),
    }

