"""Final DSS case library loader.

The dashboard source is the original paired work-order corpus. Retrieval structure
comes only from precomputed PROBLEM-only SpERT predictions. ACTION is retained as
historical evidence and is never used to construct query features.
"""
from __future__ import annotations
import json, os, hashlib
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from . import normalize as N
from .textnorm import normalize_text
from .extraction import spert_to_structure, rule_structure

@dataclass
class Corpus:
    df: pd.DataFrame
    mode: str
    normalizer: object|None=None
    @property
    def n(self): return len(self.df)
    @property
    def n_clusters(self): return self.df["cluster_id"].nunique()
    def cluster_support(self): return self.df.groupby("cluster_id").size()

def _read_predictions(path: Path):
    out={}
    if not path.is_file(): return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line); ident=str(r.get("ident") or r.get("id") or "")
                if ident: out[ident]=r.get("problem_pred") or r
    return out

def _read_protocol(path: Path|None):
    if not path or not path.is_file(): return {}
    return json.loads(path.read_text(encoding="utf-8"))

def load_corpus(csv_path,mode="spert",spert_url="http://127.0.0.1:8765",predictions_path=None,normalize=True,use_cache=False,protocol_path=None,require_predictions=False):
    path=Path(csv_path); df=pd.read_csv(path,encoding="utf-8-sig",dtype=str,keep_default_na=False)
    cols={c.lower().strip():c for c in df.columns}; df=df.rename(columns={cols.get("ident",df.columns[0]):"ident",cols.get("problem",df.columns[1]):"problem",cols.get("action",df.columns[2]):"action"})[["ident","problem","action"]]
    df["ident"]=df["ident"].astype(str); df["problem_clean"]=df["problem"].map(normalize_text); df["action_clean"]=df["action"].map(normalize_text); df["problem_norm"]=df["problem_clean"].map(N.normalize_problem)
    preds=_read_predictions(Path(predictions_path)) if predictions_path else {}
    if require_predictions and len(preds)!=len(df): raise RuntimeError(f"Final DSS requires {len(df)} PROBLEM-only predictions; found {len(preds)} at {predictions_path}")
    comps=[]; faults=[]; locs=[]; fams=[]; roles=[]; outs=[]; srcs=[]; ets=[]; rts=[]
    for r in df.itertuples(index=False):
        if str(r.ident) in preds: st=spert_to_structure(preds[str(r.ident)],r.action)
        else: st=rule_structure(r.problem_clean,r.action_clean)
        comps.append(st.components); faults.append(st.faults); locs.append(st.locations); fam=N.action_family(r.action); fams.append(fam); roles.append(N.action_role(fam)); outs.append(N.outcome_polarity(r.action)); srcs.append("problem_only_spert" if str(r.ident) in preds else "rule")
        ets.append(sorted({str(e.get("type")) for e in st.entities if e.get("type")})); rts.append(sorted({str(x.get("type")) for x in st.relations if x.get("type")}))
    df["components"],df["faults"],df["locations"]=comps,faults,locs; df["action_family"],df["action_role"],df["outcome"],df["source"]=fams,roles,outs,srcs; df["problem_entity_types"],df["problem_relation_types"]=ets,rts
    protocol=_read_protocol(Path(protocol_path) if protocol_path else None)
    c2e=protocol.get("case_to_evidence_cluster",{}); c2g=protocol.get("case_to_leakage_group",{}); c2s=protocol.get("case_to_split",{})
    if c2e:
        df["cluster_id"]=df["ident"].map(c2e); df["leakage_group_id"]=df["ident"].map(c2g); df["frozen_split"]=df["ident"].map(c2s)
        if df[["cluster_id","leakage_group_id","frozen_split"]].isna().any().any(): raise RuntimeError("Protocol does not cover every corpus IDENT")
    else:
        keys=df["problem_norm"].fillna(""); df["cluster_id"]=pd.factorize(keys)[0].astype(str); df["leakage_group_id"]=df["cluster_id"]; df["frozen_split"]="unspecified"
    df["component"]=df["components"].map(lambda x:x[0] if x else "(unspecified)"); df["fault"]=df["faults"].map(lambda x:x[0] if x else "(unspecified)")
    return Corpus(df=df,mode="predictions" if preds else "rule",normalizer=None)
