from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"matplotlib is required for report-only figure generation: {exc}")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORTS = OUT / "reports"
SPERT = OUT / "spert"
FINAL = REPORTS / "final_benchmark"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"

TIER1 = REPORTS / "ie_results__maintie_tier1.json"
NEURAL = REPORTS / "ie_results__maintie_neural.json"
TIER1_MANIFEST = REPORTS / "ie_results__maintie_tier1_manifest.json"
NEURAL_MANIFEST = REPORTS / "ie_results__maintie_neural_manifest.json"
OVERLAP = REPORTS / "maintie_overlap_audit.json"
SPAN_ABL = REPORTS / "tables" / "span_ner_ablation.csv"
GOLD = SPERT / "test.json"
PRED = REPORTS / "spert_test.json"

ENTITY_TYPES = ["PhysicalObject", "Activity", "State", "Process", "Property"]
RELATION_TYPES = ["hasPatient", "hasPart", "isA", "hasAgent", "contains", "hasProperty"]

def load_json(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f

def ent_tuple(e):
    return (str(e["type"]), int(e["start"]), int(e["end"]))

def rel_tuple(doc, r, with_types=True):
    ents = doc.get("entities", [])
    h = ents[int(r["head"])]
    t = ents[int(r["tail"])]
    if with_types:
        hs = ent_tuple(h)
        ts = ent_tuple(t)
    else:
        hs = (int(h["start"]), int(h["end"]))
        ts = (int(t["start"]), int(t["end"]))
    return (str(r["type"]), hs, ts)

def score_entities(gold_docs, pred_docs):
    tp = fp = fn = 0
    per = {}
    for typ in ENTITY_TYPES:
        ttp = tfp = tfn = 0
        for g,p in zip(gold_docs,pred_docs):
            gs = {ent_tuple(e) for e in g.get("entities", []) if e["type"] == typ}
            ps = {ent_tuple(e) for e in p.get("entities", []) if e["type"] == typ}
            ttp += len(gs & ps); tfp += len(ps-gs); tfn += len(gs-ps)
        pp,rr,ff=prf(ttp,tfp,tfn)
        per[typ]={"precision":pp,"recall":rr,"f1":ff,"support":ttp+tfn,
                  "tp":ttp,"fp":tfp,"fn":tfn}
        tp+=ttp; fp+=tfp; fn+=tfn
    p,r,f=prf(tp,fp,fn)
    return {"micro_p":p,"micro_r":r,"micro_f1":f,
            "macro_f1":sum(per[t]["f1"] for t in ENTITY_TYPES)/len(ENTITY_TYPES),
            "per_class":per,"support":tp+fn,"predicted":tp+fp,
            "tp":tp,"fp":fp,"fn":fn}

def score_relations(gold_docs, pred_docs, with_types=True):
    tp = fp = fn = 0
    per = {}
    for typ in RELATION_TYPES:
        ttp=tfp=tfn=0
        for g,p in zip(gold_docs,pred_docs):
            gs={rel_tuple(g,r,with_types) for r in g.get("relations",[]) if r["type"]==typ}
            ps={rel_tuple(p,r,with_types) for r in p.get("relations",[]) if r["type"]==typ}
            ttp+=len(gs & ps); tfp+=len(ps-gs); tfn+=len(gs-ps)
        pp,rr,ff=prf(ttp,tfp,tfn)
        per[typ]={"precision":pp,"recall":rr,"f1":ff,"support":ttp+tfn,
                  "tp":ttp,"fp":tfp,"fn":tfn}
        tp+=ttp; fp+=tfp; fn+=tfn
    p,r,f=prf(tp,fp,fn)
    return {"micro_p":p,"micro_r":r,"micro_f1":f,
            "macro_f1":sum(per[t]["f1"] for t in RELATION_TYPES)/len(RELATION_TYPES),
            "per_class":per,"support":tp+fn,"predicted":tp+fp,
            "tp":tp,"fp":fp,"fn":fn}

def nested_gold_entities(doc):
    es=[ent_tuple(e) for e in doc.get("entities",[])]
    nested=set()
    for i,a in enumerate(es):
        _,as_,ae=a
        for j,b in enumerate(es):
            if i==j: continue
            _,bs,be=b
            if max(as_,bs) < min(ae,be):
                nested.add(a)
                break
    return nested

def nested_recall(gold_docs,pred_docs):
    n=hit=0
    for g,p in zip(gold_docs,pred_docs):
        ng=nested_gold_entities(g)
        ps={ent_tuple(e) for e in p.get("entities",[])}
        n += len(ng)
        hit += len(ng & ps)
    return hit/n if n else 0.0, hit, n

def read_baseline_result(path):
    return load_json(path)

def one_model_result(obj, key):
    d=obj[key]
    return {
        "entity_micro_f1": float(d["entity"]["micro_f1"]),
        "entity_macro_f1": float(d["entity"]["macro_f1"]),
        "relation_micro_f1": float(d["relation"]["micro_f1"]),
        "relation_macro_f1": float(d["relation"]["macro_f1"]),
        "entity_per_class": d["entity"]["per_class"],
        "relation_per_class": d["relation"]["per_class"],
    }

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def bar_plot(path_base, labels, series, ylabel, title, ylim=(0,1)):
    x=list(range(len(labels)))
    n=max(1,len(series))
    width=0.8/n
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.3),5.5))
    for k,(name,vals) in enumerate(series.items()):
        offset=(k-(n-1)/2)*width
        ax.bar([v+offset for v in x],vals,width=width,label=name)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=30,ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path_base)+".png",dpi=180)
    fig.savefig(str(path_base)+".pdf")
    plt.close(fig)

def main():
    print("="*74)
    print(" MAINTIE FINAL BENCHMARK REPORT - EXISTING ARTIFACTS ONLY / NO TRAINING")
    print("="*74)

    gold=load_json(GOLD)
    pred=load_json(PRED)
    if not isinstance(gold,list) or not isinstance(pred,list):
        raise SystemExit("Gold/prediction files must be JSON arrays.")
    if len(gold)!=108 or len(pred)!=108:
        raise SystemExit(f"Expected 108 TEST records; got gold={len(gold)} pred={len(pred)}")
    for i,(g,p) in enumerate(zip(gold,pred)):
        if g.get("tokens") != p.get("tokens"):
            raise SystemExit(f"Token alignment mismatch at TEST record {i}")

    tier1=read_baseline_result(TIER1)
    neural=read_baseline_result(NEURAL)
    overlap=load_json(OVERLAP)

    e=score_entities(gold,pred)
    r=score_relations(gold,pred,with_types=True)
    r_no_nec=score_relations(gold,pred,with_types=False)
    nr,nhit,ngold=nested_recall(gold,pred)

    spert={
        "entity": e,
        "relation_strict_with_entity_classification": r,
        "relation_span_only_without_entity_classification": r_no_nec,
        "nested_entity_recall": nr,
        "nested_entity_hits": nhit,
        "nested_entity_support": ngold,
        "evaluation_gold":"original_full_spans_and_relations",
        "test_records":108,
    }

    FINAL.mkdir(parents=True,exist_ok=True)
    TABLES.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    (FINAL/"spert_test_metrics.json").write_text(json.dumps(spert,indent=2),encoding="utf-8")

    models={
        "CRF+LogReg": one_model_result(tier1,"Tier1_CRF_LogReg"),
        "BiLSTM+Neural": one_model_result(neural,"Tier2_BiLSTM_Neural"),
        "DistilBERT": one_model_result(neural,"Tier3_Transformer"),
        "SpERT": {
            "entity_micro_f1":e["micro_f1"],
            "entity_macro_f1":e["macro_f1"],
            "relation_micro_f1":r["micro_f1"],
            "relation_macro_f1":r["macro_f1"],
            "entity_per_class":{k:[v["precision"],v["recall"],v["f1"]] for k,v in e["per_class"].items()},
            "relation_per_class":{k:[v["precision"],v["recall"],v["f1"]] for k,v in r["per_class"].items()},
        },
    }

    overall_rows=[]
    for name,d in models.items():
        overall_rows.append({
            "model":name,
            "entity_micro_f1":round(d["entity_micro_f1"],6),
            "entity_macro_f1":round(d["entity_macro_f1"],6),
            "strict_relation_micro_f1":round(d["relation_micro_f1"],6),
            "relation_macro_f1":round(d["relation_macro_f1"],6),
        })
    write_csv(TABLES/"overall_metrics.csv",overall_rows,list(overall_rows[0].keys()))

    entity_rows=[]
    for typ in ENTITY_TYPES:
        row={"entity_type":typ,"support":e["per_class"][typ]["support"]}
        for name,d in models.items():
            vals=d["entity_per_class"].get(typ,[0,0,0])
            row[name]=round(float(vals[2]),6)
        entity_rows.append(row)
    write_csv(TABLES/"entity_per_class_f1.csv",entity_rows,
              ["entity_type","support"]+list(models.keys()))

    relation_rows=[]
    for typ in RELATION_TYPES:
        row={"relation_type":typ,"support":r["per_class"][typ]["support"]}
        for name,d in models.items():
            vals=d["relation_per_class"].get(typ,[0,0,0])
            row[name]=round(float(vals[2]),6)
        relation_rows.append(row)
    write_csv(TABLES/"relation_per_class_f1.csv",relation_rows,
              ["relation_type","support"]+list(models.keys()))

    test_audit=overlap["test"]
    ceiling_rows=[
        {"metric":"entity_recall","bio_theoretical_ceiling":test_audit["bio_entity_recall_ceiling"],
         "Tier1":tier1["Tier1_CRF_LogReg"]["entity"]["micro_r"],
         "Tier2":neural["Tier2_BiLSTM_Neural"]["entity"]["micro_r"],
         "Tier3":neural["Tier3_Transformer"]["entity"]["micro_r"],
         "SpERT":e["micro_r"]},
        {"metric":"relation_recall","bio_theoretical_ceiling":test_audit["bio_relation_recall_ceiling"],
         "Tier1":tier1["Tier1_CRF_LogReg"]["relation"]["micro_r"],
         "Tier2":neural["Tier2_BiLSTM_Neural"]["relation"]["micro_r"],
         "Tier3":neural["Tier3_Transformer"]["relation"]["micro_r"],
         "SpERT":r["micro_r"]},
    ]
    write_csv(TABLES/"bio_ceiling_vs_achieved_recall.csv",ceiling_rows,
              ["metric","bio_theoretical_ceiling","Tier1","Tier2","Tier3","SpERT"])

    span_rows=[]
    if SPAN_ABL.exists():
        with SPAN_ABL.open(newline="",encoding="utf-8-sig") as f:
            span_rows=list(csv.DictReader(f))
        shutil.copy2(SPAN_ABL,TABLES/"span_ner_ablation.csv")
    span_rows_plus = span_rows + [{
        "model":"SpERT",
        "entity_micro_f1":f"{e['micro_f1']:.6f}",
        "entity_macro_f1":f"{e['macro_f1']:.6f}",
        "nested_entity_recall":f"{nr:.6f}",
        "n_nested_gold":str(ngold),
    }]
    if span_rows_plus:
        fields=["model","entity_micro_f1","entity_macro_f1","nested_entity_recall","n_nested_gold"]
        write_csv(TABLES/"span_and_spert_nested_recall.csv",span_rows_plus,fields)

    bar_plot(FIGURES/"overall_four_model_comparison",
             ["Entity micro-F1","Entity macro-F1","Strict relation micro-F1","Relation macro-F1"],
             {n:[d["entity_micro_f1"],d["entity_macro_f1"],d["relation_micro_f1"],d["relation_macro_f1"]] for n,d in models.items()},
             "F1","MaintIE frozen TEST: four-model comparison")

    bar_plot(FIGURES/"entity_per_class_f1_all_models",ENTITY_TYPES,
             {n:[float(d["entity_per_class"].get(t,[0,0,0])[2]) for t in ENTITY_TYPES] for n,d in models.items()},
             "F1","MaintIE frozen TEST: entity per-class F1")

    bar_plot(FIGURES/"relation_per_class_f1_all_models",RELATION_TYPES,
             {n:[float(d["relation_per_class"].get(t,[0,0,0])[2]) for t in RELATION_TYPES] for n,d in models.items()},
             "F1","MaintIE frozen TEST: strict relation per-class F1")

    if span_rows_plus:
        labels=[row["model"] for row in span_rows_plus]
        vals=[float(row["nested_entity_recall"]) for row in span_rows_plus]
        bar_plot(FIGURES/"nested_entity_recall",labels,{"Nested entity recall":vals},
                 "Recall",f"MaintIE nested/overlapping entity recovery (n={ngold})")

    ceiling_labels=["Entity recall","Relation recall"]
    bar_plot(FIGURES/"bio_ceiling_vs_achieved_recall",ceiling_labels,
             {
                "BIO ceiling":[test_audit["bio_entity_recall_ceiling"],test_audit["bio_relation_recall_ceiling"]],
                "Tier1":[tier1["Tier1_CRF_LogReg"]["entity"]["micro_r"],tier1["Tier1_CRF_LogReg"]["relation"]["micro_r"]],
                "Tier2":[neural["Tier2_BiLSTM_Neural"]["entity"]["micro_r"],neural["Tier2_BiLSTM_Neural"]["relation"]["micro_r"]],
                "Tier3":[neural["Tier3_Transformer"]["entity"]["micro_r"],neural["Tier3_Transformer"]["relation"]["micro_r"]],
                "SpERT":[e["micro_r"],r["micro_r"]],
             },"Recall","MaintIE TEST: BIO representation ceiling vs achieved recall")

    manifest={
        "status":"complete",
        "training_performed":False,
        "generated_at_utc":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "frozen_split":{"train":860,"dev":108,"test":108},
        "schema":{"entities":5,"relations":6},
        "evaluation_gold":"original_full_spans_and_relations",
        "bio_policy":"earliest_then_longest_nonoverlap_for_BIO_models_only",
        "sources":{
            "tier1":str(TIER1.relative_to(ROOT)),
            "neural":str(NEURAL.relative_to(ROOT)),
            "overlap_audit":str(OVERLAP.relative_to(ROOT)),
            "span_ablation":str(SPAN_ABL.relative_to(ROOT)) if SPAN_ABL.exists() else None,
            "spert_gold_test":str(GOLD.relative_to(ROOT)),
            "spert_predictions":str(PRED.relative_to(ROOT)),
        },
        "spert_test_summary":{
            "entity_micro_f1":e["micro_f1"],
            "entity_macro_f1":e["macro_f1"],
            "strict_relation_micro_f1":r["micro_f1"],
            "relation_macro_f1":r["macro_f1"],
            "isA_f1":r["per_class"]["isA"]["f1"],
            "nested_entity_recall":nr,
            "nested_entity_support":ngold,
        },
        "final_dir":str(FINAL.relative_to(ROOT)),
    }
    (FINAL/"FINAL_MAINTIE_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    print()
    print("SPERT FROZEN TEST")
    print(f"  entity   P={e['micro_p']:.4f} R={e['micro_r']:.4f} F1={e['micro_f1']:.4f} macro={e['macro_f1']:.4f}")
    print(f"  relation P={r['micro_p']:.4f} R={r['micro_r']:.4f} F1={r['micro_f1']:.4f} macro={r['macro_f1']:.4f}")
    print(f"  isA      F1={r['per_class']['isA']['f1']:.4f} support={r['per_class']['isA']['support']}")
    print(f"  nested entity recall={nr:.4f} ({nhit}/{ngold})")
    print()
    print("FOUR-MODEL TEST SUMMARY")
    for row in overall_rows:
        print(f"  {row['model']:<16} entity={row['entity_micro_f1']:.4f} strict_relation={row['strict_relation_micro_f1']:.4f}")
    print()
    print(f"Saved -> {FINAL}")
    print("NO TRAINING WAS PERFORMED.")

if __name__=="__main__":
    main()
