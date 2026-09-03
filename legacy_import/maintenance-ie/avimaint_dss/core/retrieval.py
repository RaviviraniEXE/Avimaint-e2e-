"""Leakage-safe multi-channel historical-case retrieval.

Ranking and confidence are intentionally separate. Retrieval returns ranked cases
plus transparent channel scores. Confidence is decided later from independent
historical evidence, not from a per-query normalized similarity alone.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOK = re.compile(r"[a-z0-9#]+")
def _tok(text): return _TOK.findall((text or "").lower())

@dataclass
class Hit:
    idx: int
    score: float
    text_sim: float
    struct: float
    channels: dict[str,float] = field(default_factory=dict)

class Retriever:
    MODES = {"raw_bm25","norm_bm25","norm_tfidf","structure","weighted_hybrid","rrf_hybrid"}
    def __init__(self, df: pd.DataFrame, weights=None, dense_model=None, rrf_k: int=60):
        self.df=df.reset_index(drop=True)
        self.raw_docs=self.df["problem"].fillna("").astype(str).tolist()
        self.docs=self.df["problem_norm"].fillna("").astype(str).tolist()
        self.weights=weights or {"bm25":.34,"word":.24,"char":.14,"struct":.28}
        self.rrf_k=int(rrf_k)
        self._raw_bm25=BM25Okapi([_tok(d) for d in self.raw_docs])
        self._bm25=BM25Okapi([_tok(d) for d in self.docs])
        self._word=TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=1)
        self._wm=self._word.fit_transform(self.docs)
        self._char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=1)
        self._cm=self._char.fit_transform(self.docs)
        self._comp=self.df["components"].tolist(); self._fault=self.df["faults"].tolist()
        self._entity_types=self.df.get("problem_entity_types",pd.Series([[] for _ in range(len(self.df))])).tolist()
        self._relation_types=self.df.get("problem_relation_types",pd.Series([[] for _ in range(len(self.df))])).tolist()
        self._dense=self._demb=None
        if dense_model: self._init_dense(dense_model)
    def _init_dense(self,name):
        try:
            from sentence_transformers import SentenceTransformer
            self._dense=SentenceTransformer(name)
            self._demb=self._dense.encode(self.docs,normalize_embeddings=True,show_progress_bar=False)
        except Exception: self._dense=None
    @staticmethod
    def _unit(a):
        a=np.asarray(a,dtype=float); lo=float(a.min()) if a.size else 0.; hi=float(a.max()) if a.size else 0.
        return (a-lo)/(hi-lo) if hi-lo>1e-12 else np.zeros_like(a)
    def _struct_overlap(self,q_comp,q_fault,q_entity_types=None,q_relation_types=None):
        qc,qf=set(q_comp or []),set(q_fault or []); qe,qr=set(q_entity_types or []),set(q_relation_types or [])
        out=np.zeros(len(self.docs)); enabled=(bool(qc),bool(qf),bool(qe),bool(qr)); ww=(.40,.30,.15,.15)
        denom=sum(w for w,e in zip(ww,enabled) if e)
        if not denom:return out
        for i in range(len(self.docs)):
            vals=(len(qc&set(self._comp[i]))/len(qc) if qc else 0.,len(qf&set(self._fault[i]))/len(qf) if qf else 0.,len(qe&set(self._entity_types[i]))/len(qe) if qe else 0.,len(qr&set(self._relation_types[i]))/len(qr) if qr else 0.)
            out[i]=sum(w*v for w,v,e in zip(ww,vals,enabled) if e)/denom
        return out
    @staticmethod
    def _rrf(channels,k=60):
        n=len(next(iter(channels.values()))); total=np.zeros(n)
        for arr in channels.values():
            order=np.argsort(-arr); ranks=np.empty(n,dtype=int); ranks[order]=np.arange(1,n+1)
            total += 1.0/(k+ranks)
        return total
    def search(self,query,q_comp,q_fault,top_k=25,q_entity_types=None,q_relation_types=None,*,raw_query=None,mode="weighted_hybrid",exclude_groups=None,diversify=True):
        if mode not in self.MODES: raise ValueError(f"Unknown retrieval mode {mode!r}")
        raw_query=raw_query if raw_query is not None else query
        rb=self._unit(np.asarray(self._raw_bm25.get_scores(_tok(raw_query))))
        bm=self._unit(np.asarray(self._bm25.get_scores(_tok(query))))
        wd=cosine_similarity(self._word.transform([query]),self._wm).ravel()
        ch=cosine_similarity(self._char.transform([query]),self._cm).ravel()
        st=self._struct_overlap(q_comp,q_fault,q_entity_types,q_relation_types)
        channels={"raw_bm25":rb,"bm25":bm,"word":wd,"char":ch,"struct":st}
        if mode=="raw_bm25": total=rb
        elif mode=="norm_bm25": total=bm
        elif mode=="norm_tfidf": total=.65*wd+.35*ch
        elif mode=="structure": total=st
        elif mode=="rrf_hybrid": total=self._rrf({"bm25":bm,"word":wd,"char":ch,"struct":st},self.rrf_k)
        else:
            w=self.weights; total=w.get("bm25",0)*bm+w.get("word",0)*wd+w.get("char",0)*ch+w.get("struct",0)*st
        excluded=set(str(x) for x in (exclude_groups or set()))
        order=np.argsort(-total)
        hits=[]; used=set()
        for i in order:
            if excluded and "leakage_group_id" in self.df and str(self.df.iloc[i]["leakage_group_id"]) in excluded: continue
            if diversify and "cluster_id" in self.df:
                c=str(self.df.iloc[i]["cluster_id"])
                if c in used: continue
                used.add(c)
            text=.5*bm[i]+.3*wd[i]+.2*ch[i]
            hits.append(Hit(int(i),float(total[i]),float(text),float(st[i]),{k:float(v[i]) for k,v in channels.items()}))
            if len(hits)>=top_k: break
        return hits
