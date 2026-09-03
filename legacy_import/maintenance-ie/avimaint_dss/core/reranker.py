"""Optional cross-encoder reranker over the top retrieved candidates.

Hybrid retrieval proposes a candidate pool, then a cross-encoder scores
(query-problem, historical-problem) pairs and the final
order blends the cross-encoder score with the retrieval score.

Robust loading — tries, in order:
  1. sentence-transformers `CrossEncoder` (handles `cross-encoder/...` hub models
     and standard HF cross-encoder folders);
  2. raw `transformers` AutoModelForSequenceClassification (handles a fine-tuned
     development-selected checkpoint dir, even when it is not saved in
     sentence-transformers format).

If both fail, `available()` returns False and `last_error()` explains why (the
sidebar shows this), instead of silently staying OFF.

Configure via retrieval.reranker_model:
  cross-encoder/ms-marco-MiniLM-L-6-v2                          (pretrained, downloads)
  path/to/development-selected/reranker                     (your trained model)
"""
from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model_name: str, blend: float = 0.7):
        self.model_name = model_name
        self.blend = blend
        self._ok = None
        self._backend = None            # "sentence-transformers" | "transformers"
        self._error = ""
        self._ce = None                 # sentence-transformers CrossEncoder
        self._tok = self._model = self._torch = None

    # ------------------------------------------------------------------ load
    def available(self) -> bool:
        if self._ok is not None:
            return self._ok
        # 1) sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self.model_name)
            self._backend = "sentence-transformers"
            self._ok = True
            return True
        except Exception as e:
            self._error = f"sentence-transformers loader: {type(e).__name__}: {e}"
        # 2) raw transformers sequence-classification (e.g. a trained local checkpoint)
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._torch = torch
            self._backend = "transformers"
            self._error = ""
            self._ok = True
            return True
        except Exception as e:
            self._error += f"  |  transformers loader: {type(e).__name__}: {e}"
        self._ok = False
        return False

    def last_error(self) -> str:
        return self._error

    def backend(self) -> str:
        return self._backend or "none"

    # --------------------------------------------------------------- scoring
    def _scores(self, pairs):
        import numpy as np
        if self._backend == "sentence-transformers":
            return np.asarray(self._ce.predict(pairs), dtype=float)
        # transformers backend — score each (query, candidate) pair
        tok, model, torch = self._tok, self._model, self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(pairs), 32):
                batch = pairs[i:i + 32]
                enc = tok([p[0] for p in batch], [p[1] for p in batch],
                          padding=True, truncation=True, max_length=256, return_tensors="pt")
                logits = model(**enc).logits
                if logits.shape[-1] == 1:                 # regression / single relevance score
                    s = logits.squeeze(-1)
                else:                                     # take the positive/last class logit
                    s = logits[:, -1]
                out.extend([float(x) for x in s.tolist()])
        return np.asarray(out, dtype=float)

    def rerank(self, query: str, candidates: list[tuple[int, str, float]]):
        """candidates = [(row_idx, candidate_text, retrieval_score)].
        Returns [(row_idx, blended_score)] sorted desc."""
        if not candidates or not self.available():
            return [(i, s) for (i, _t, s) in candidates]
        import numpy as np
        pairs = [[query, t] for (_i, t, _s) in candidates]
        ce = self._scores(pairs)
        rng = ce.max() - ce.min()
        ce_n = (ce - ce.min()) / rng if rng > 1e-9 else ce * 0.0
        out = []
        for k, (i, _t, s) in enumerate(candidates):
            out.append((i, self.blend * float(ce_n[k]) + (1 - self.blend) * float(s)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
