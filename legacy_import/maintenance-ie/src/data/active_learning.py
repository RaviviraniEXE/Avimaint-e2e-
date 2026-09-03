"""Active-learning scoring for rare-class enrichment.

Given the current CRF, score each unlabeled record by:
    score = rare_weight * (predicted rare-class entities present?)
          + uncertainty_weight * (model uncertainty)
and return the top-N as the next batch. This targets records that both look like
they contain the hard classes AND that the model is unsure about — the standard
uncertainty + targeted-class active-learning criterion. TRAIN-pool only; records
already in gold or in the frozen test/dev are excluded by the caller.
"""
from __future__ import annotations

from src.models.crf_ner import bio_to_entities


def score_record(crf, tokens, rare_entities, rare_weight, unc_weight):
    bio = crf.predict_bio(tokens)
    ents = bio_to_entities(tokens, bio)
    n_rare = sum(1 for e in ents if e["type"] in rare_entities)
    unc = crf.uncertainty(tokens)
    return rare_weight * min(n_rare, 3) + unc_weight * unc, bio, ents, n_rare, unc


def rank_pool(crf, records, cfg):
    """records: list of {ident, tokens, ...}. Returns records sorted by AL score
    (desc), each annotated with model bio/entities/score for pre-labeling."""
    al = cfg["annotation"]["active_learning"]
    scored = []
    for r in records:
        s, bio, ents, n_rare, unc = score_record(
            crf, r["tokens"], al["rare_entities"], al["rare_weight"], al["uncertainty_weight"])
        scored.append({**r, "bio": bio, "entities": ents, "relations": [],
                       "al_score": round(s, 4), "n_rare": n_rare, "uncertainty": round(unc, 4)})
    scored.sort(key=lambda x: -x["al_score"])
    return scored

