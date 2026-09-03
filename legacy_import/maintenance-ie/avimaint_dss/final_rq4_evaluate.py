from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd,yaml
from core.corpus import load_corpus
from core.retrieval import Retriever
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]
MODES=['raw_bm25','norm_bm25','norm_tfidf','structure','weighted_hybrid','rrf_hybrid']
def load():
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8')); pred=(ROOT/cfg['data']['problem_predictions_path']).resolve(); protocol=(ROOT/cfg['data']['protocol_path']).resolve(); return load_corpus(ROOT/cfg['data']['csv_path'],predictions_path=pred,protocol_path=protocol,require_predictions=True).df

def family_rank(cand,hits,true):
    meta={}
    for h in hits:
        r=cand.iloc[h.idx]; fam=str(r.action_family)
        if fam=='Other' or r.outcome in ('negative','mixed'):continue
        d=meta.setdefault(fam,{}); c=str(r.cluster_id); d[c]=max(d.get(c,0.),float(h.score))
    ranked=sorted(meta,key=lambda f:(len(meta[f]),sum(meta[f].values())),reverse=True); return ranked,meta

def evaluate(cand,q,mode):
    C=cand.reset_index(drop=True); R=Retriever(C); rows=[]
    for x in q[q.action_family!='Other'].itertuples(index=False):
        hits=R.search(x.problem_norm,x.components,x.faults,top_k=50,q_entity_types=x.problem_entity_types,q_relation_types=x.problem_relation_types,raw_query=x.problem,mode=mode,exclude_groups={str(x.leakage_group_id)},diversify=True)
        ranked,meta=family_rank(C,hits,x.action_family); true=str(x.action_family); rank=ranked.index(true)+1 if true in ranked else 0; pred=ranked[0] if ranked else ''
        top=hits[0].score if hits else 0.; second=hits[1].score if len(hits)>1 else 0.; support=len(meta.get(pred,{}))
        rows.append({'ident':x.ident,'leakage_group_id':x.leakage_group_id,'true_family':true,'predicted_family':pred,'rank':rank,'top1_correct':pred==true,'top3_correct':0<rank<=3,'top5_correct':0<rank<=5,'rr':1/rank if rank else 0.,'ndcg5':1/math.log2(rank+1) if 0<rank<=5 else 0.,'top_score':top,'margin':top-second,'support_clusters':support})
    d=pd.DataFrame(rows); pf=d.groupby('true_family').top1_correct.mean() if len(d) else pd.Series(dtype=float); return {'queries':len(d),'hit_at_1':float(d.top1_correct.mean()),'hit_at_3':float(d.top3_correct.mean()),'hit_at_5':float(d.top5_correct.mean()),'mrr':float(d.rr.mean()),'ndcg_at_5':float(d.ndcg5.mean()),'macro_action_family_recall':float(pf.mean()) if len(pf) else 0.,'interpretation':'agreement with recorded historical action family, not technical correctness'},d

def main():
    a=argparse.ArgumentParser(); a.add_argument('--partition',choices=['dev','test'],default='dev'); a.add_argument('--confirm-final-test',action='store_true'); z=a.parse_args(); out=REPO/'outputs/runs/rq4_case_retrieval'/z.partition; out.mkdir(parents=True,exist_ok=True); lock=REPO/'outputs/runs/rq4_case_retrieval/FINAL_TEST_LOCK.json'
    if z.partition=='test':
        if not z.confirm_final_test: raise SystemExit('TEST is gated. Re-run with --confirm-final-test only after DEV selection is frozen.')
        if lock.exists(): raise SystemExit('FINAL TEST already locked; no rerun allowed.')
    df=load(); train=df[df.frozen_split=='train']; q=df[df.frozen_split==z.partition]; results={}; devsel=None
    for mode in MODES:
        m,p=evaluate(train,q,mode); results[mode]=m; p.to_csv(out/f'predictions_{mode}.csv',index=False); print(mode,m)
    if z.partition=='dev':
        devsel=max(MODES,key=lambda m:(results[m]['macro_action_family_recall'],results[m]['mrr'],results[m]['hit_at_1'])); (out/'RQ4_DEV_SELECTION.json').write_text(json.dumps({'selected_mode':devsel,'selection_rule':'max macro recall, then MRR, then Hit@1','metrics':results},indent=2),encoding='utf-8')
    else:
        sel=json.loads((REPO/'outputs/runs/rq4_case_retrieval/dev/RQ4_DEV_SELECTION.json').read_text(encoding='utf-8'))['selected_mode']; (out/'RQ4_FINAL_TEST.json').write_text(json.dumps({'selected_mode':sel,'selected_mode_metrics':results[sel],'all_frozen_comparisons':results},indent=2),encoding='utf-8'); lock.write_text(json.dumps({'locked':True,'selected_mode':sel},indent=2),encoding='utf-8')
    (out/'metrics_all.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
if __name__=='__main__': main()
