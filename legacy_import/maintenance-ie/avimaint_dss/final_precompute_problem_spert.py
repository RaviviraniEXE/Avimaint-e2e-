from pathlib import Path
import json,time,pandas as pd
from core.extraction import SpERTClient
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]; OUT=REPO/'outputs/runs/rq4_case_retrieval/problem_only_spert'; OUT.mkdir(parents=True,exist_ok=True); PATH=OUT/'predictions_problem_only.jsonl'
def main():
    df=pd.read_csv(ROOT/'data/Aircraft_Annotation_DataFile.csv',dtype=str,keep_default_na=False); df.IDENT=df.IDENT.astype(str); client=SpERTClient('http://127.0.0.1:8765',timeout=60)
    if not client.health(): raise SystemExit('SpERT service is not ready. Start FINAL_03_START_MATCHED_SPERT.bat first.')
    done={}
    if PATH.is_file():
        for l in PATH.read_text(encoding='utf-8').splitlines():
            if l.strip(): r=json.loads(l); done[str(r['ident'])]=r
    with PATH.open('a',encoding='utf-8') as f:
        for n,r in enumerate(df.itertuples(index=False),1):
            if r.IDENT in done: continue
            pred=client.predict(r.PROBLEM)
            if pred is None: raise SystemExit(f'Inference failed at IDENT={r.IDENT}')
            f.write(json.dumps({'ident':r.IDENT,'problem_text':r.PROBLEM,'problem_pred':pred},ensure_ascii=False)+'\n'); f.flush()
            if n%100==0: print(f'{n}/{len(df)}')
    rows=[json.loads(x) for x in PATH.read_text(encoding='utf-8').splitlines() if x.strip()]; ids=[str(x['ident']) for x in rows]
    if len(rows)!=6169 or len(set(ids))!=6169: raise SystemExit(f'Expected 6169 unique predictions, got rows={len(rows)} unique={len(set(ids))}')
    print('PROBLEM-ONLY SPERT VERIFIED: 6169/6169 ->',PATH)
if __name__=='__main__': main()
