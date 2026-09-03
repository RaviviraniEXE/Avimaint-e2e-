from pathlib import Path
import hashlib,json,shutil,datetime
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[2]; SRC=REPO/'outputs/runs'; DST=REPO/'outputs/frozen/final_rq4_rq5'
def h(p):
 d=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
 return d.hexdigest()
def main():
 req=[SRC/'rq4_case_retrieval/test/RQ4_FINAL_TEST.json',SRC/'rq4_case_retrieval/FINAL_TEST_LOCK.json',SRC/'rq5_planning_support/RQ5_PLANNING_SUPPORT.json']
 if any(not x.exists() for x in req): raise SystemExit('RQ4 final TEST and RQ5 planning-support outputs must exist before freeze.')
 if DST.exists(): shutil.rmtree(DST)
 for name in ('rq4_case_retrieval','rq5_planning_support'): shutil.copytree(SRC/name,DST/name)
 (DST/'FREEZE_MANIFEST.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete','thesis_rq4':'leakage-safe historical-case retrieval','thesis_rq5':'auditable planning support','no_post_test_tuning':True},indent=2),encoding='utf-8')
 rows=[]
 for p in sorted(DST.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS.txt':rows.append(f'{h(p)}  {p.relative_to(DST).as_posix()}')
 (DST/'SHA256SUMS.txt').write_text('\n'.join(rows)+'\n',encoding='ascii'); print('FROZEN',DST,'files=',len(rows))
if __name__=='__main__':main()
