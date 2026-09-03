from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]
def ece(y,p,bins=10):
    e=0.; edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if m.any():e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)
def main():
    devdir=REPO/'outputs/runs/rq4_case_retrieval/dev'; testdir=REPO/'outputs/runs/rq4_case_retrieval/test'; sel=json.loads((devdir/'RQ4_DEV_SELECTION.json').read_text(encoding='utf-8'))['selected_mode']; lock=REPO/'outputs/runs/rq4_case_retrieval/FINAL_TEST_LOCK.json'
    if not lock.exists(): raise SystemExit('Run the gated final RQ4 TEST first.')
    dev=pd.read_csv(devdir/f'predictions_{sel}.csv'); test=pd.read_csv(testdir/f'predictions_{sel}.csv'); cols=['top_score','margin','support_clusters']; Xd=dev[cols].astype(float).to_numpy(); yd=dev.top1_correct.astype(int).to_numpy(); Xt=test[cols].astype(float).to_numpy(); yt=test.top1_correct.astype(int).to_numpy()
    if len(np.unique(yd))<2: raise SystemExit('DEV correctness has one class; calibration not identifiable.')
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42)).fit(Xd,yd); p=model.predict_proba(Xt)[:,1]; test['historical_agreement_probability']=p
    risk=[]
    order=np.argsort(-p); yy=yt[order]
    for c in np.linspace(.1,1,10): n=max(1,int(np.ceil(len(yy)*c))); risk.append({'coverage':float(c),'agreement_accuracy':float(yy[:n].mean()),'risk':float(1-yy[:n].mean())})
    out=REPO/'outputs/runs/rq5_planning_support'; out.mkdir(parents=True,exist_ok=True); test.to_csv(out/'calibrated_test_predictions.csv',index=False); metrics={'selected_retrieval_mode':sel,'calibration_population':'DEV only','evaluation_population':'locked TEST','ece_10_bins':ece(yt,p),'brier_score':float(brier_score_loss(yt,p)),'risk_coverage':risk,'interpretation':'Probability estimates historical action-family agreement only. It is NOT probability of technical correctness, safety, or regulatory applicability.'}; (out/'RQ5_PLANNING_SUPPORT.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
