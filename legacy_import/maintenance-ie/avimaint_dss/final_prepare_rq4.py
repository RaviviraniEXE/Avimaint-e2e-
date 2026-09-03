from __future__ import annotations
import json, hashlib
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from core.textnorm import normalize_text
from core.normalize import action_family
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]
OUT=REPO/'outputs/runs/rq4_case_retrieval/protocol'; OUT.mkdir(parents=True,exist_ok=True)

def hid(prefix,s): return prefix+hashlib.sha256(s.encode()).hexdigest()[:16]
class UF:
    def __init__(self,n): self.p=list(range(n))
    def f(self,x):
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def u(self,a,b):
        a,b=self.f(a),self.f(b)
        if a!=b:self.p[b]=a

def main():
    df=pd.read_csv(ROOT/'data/Aircraft_Annotation_DataFile.csv',dtype=str,keep_default_na=False); df['IDENT']=df.IDENT.astype(str); df['norm']=df.PROBLEM.map(normalize_text)
    # exact evidence clusters: identical normalized problem only
    df['evidence_cluster']=df['norm'].map(lambda s:hid('EC-',s))
    # conservative near-duplicate leakage groups, problem-only. Similarity >= .87.
    v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,norm='l2'); X=v.fit_transform(df['norm']); nn=NearestNeighbors(metric='cosine',radius=.13,n_jobs=-1).fit(X); neigh=nn.radius_neighbors(X,return_distance=False)
    uf=UF(len(df))
    for i,ns in enumerate(neigh):
        for j in ns:
            if j>i: uf.u(i,int(j))
    roots=[uf.f(i) for i in range(len(df))]; root_names={r:hid('LG-',df.iloc[r]['norm']) for r in set(roots)}; df['leakage_group']=[root_names[r] for r in roots]
    df['family']=df.ACTION.map(action_family)
    # Deterministic group-hash 70/15/15 split. Because assignment is made at
    # leakage-group level, no near-duplicate group can cross partitions. With
    # ~2k groups this also preserves action-family mix without looking at query text.
    group_sizes=df.groupby("leakage_group").size().to_dict()
    target={"train":.70*len(df),"dev":.15*len(df),"test":.15*len(df)}
    counts=Counter(); assign={}
    # Largest-group-first bin packing gives sizes close to 70/15/15 while never
    # splitting a leakage group. SHA-256 is only the deterministic tie-breaker.
    gids=sorted(group_sizes,key=lambda g:(-group_sizes[g],hashlib.sha256(g.encode()).hexdigest()))
    for gid in gids:
        n=group_sizes[gid]
        choices=sorted(target,key=lambda s:((counts[s]+n-target[s])/max(target[s],1) if counts[s]+n>target[s] else -((target[s]-counts[s])/max(target[s],1)),s))
        s=choices[0]; assign[gid]=s; counts[s]+=n
    case_to_split={i:assign[g] for i,g in zip(df.IDENT,df.leakage_group)}; payload={'version':'rq4_protocol_v1','seed':'deterministic_largest_group_binpack','near_duplicate_similarity_threshold':0.87,'query_features':'PROBLEM only','action_used_for_retrieval_or_clustering':False,'case_to_evidence_cluster':dict(zip(df.IDENT,df.evidence_cluster)),'case_to_leakage_group':dict(zip(df.IDENT,df.leakage_group)),'case_to_split':case_to_split,'counts':dict(Counter(case_to_split.values())),'evidence_clusters':int(df.evidence_cluster.nunique()),'leakage_groups':int(df.leakage_group.nunique())}
    (OUT/'split_v1.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    pd.DataFrame({'ident':df.IDENT,'problem':df.PROBLEM,'action':df.ACTION,'evidence_cluster_id':df.evidence_cluster,'leakage_group_id':df.leakage_group,'split':df.IDENT.map(case_to_split)}).to_csv(OUT/'protocol_cases.csv',index=False)
    print(json.dumps({k:payload[k] for k in ('counts','evidence_clusters','leakage_groups')},indent=2)); print('WROTE',OUT/'split_v1.json')
if __name__=='__main__': main()
