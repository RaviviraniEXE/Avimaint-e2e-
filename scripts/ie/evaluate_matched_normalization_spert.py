"""Final evaluation for representation-matched normalization -> SpERT ablation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
IE = ROOT / "legacy_import" / "maintenance-ie"
RAW = IE / "outputs" / "spert"
BASE = IE / "outputs" / "spert_normalized"
REPORT = IE / "outputs" / "reports" / "normalization_spert_matched"
SYSTEMS = ["raw", "rules", "byt5", "selective_byt5", "rules_then_byt5"]
NORM_SYSTEMS = SYSTEMS[1:]
EXPECTED_N = 225
EXPECTED_RAW = {
    "entity_micro_f1": 0.9520,
    "entity_macro_f1": 0.9063,
    "strict_relation_micro_f1": 0.8537,
    "relation_macro_f1": 0.7898,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    z = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, z


def entity_key(e):
    return (str(e["type"]), int(e["start"]), int(e["end"]))


def relation_key(doc, r):
    ents = doc.get("entities", [])
    h = r.get("head")
    t = r.get("tail")
    hk = entity_key(h) if isinstance(h, dict) else entity_key(ents[int(h)])
    tk = entity_key(t) if isinstance(t, dict) else entity_key(ents[int(t)])
    return (str(r["type"]), hk, tk)


def counts(g, p):
    inter = g & p
    return sum(inter.values()), sum((p-g).values()), sum((g-p).values())


def evaluate(gold, pred):
    if len(gold) != len(pred):
        raise SystemExit(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    ge, pe, gr, pr = Counter(), Counter(), Counter(), Counter()
    ent_labels, rel_labels = set(), set()
    invalid_rel = 0
    for i, (gd, pd) in enumerate(zip(gold, pred)):
        for e in gd.get("entities", []):
            ge[(i,) + entity_key(e)] += 1; ent_labels.add(e["type"])
        for e in pd.get("entities", []):
            pe[(i,) + entity_key(e)] += 1
        for r in gd.get("relations", []):
            try: gr[(i,) + relation_key(gd, r)] += 1; rel_labels.add(r["type"])
            except Exception: invalid_rel += 1
        for r in pd.get("relations", []):
            try: pr[(i,) + relation_key(pd, r)] += 1
            except Exception: invalid_rel += 1
    etp, efp, efn = counts(ge, pe); ep, er, ef = f1(etp, efp, efn)
    rtp, rfp, rfn = counts(gr, pr); rp, rr, rf = f1(rtp, rfp, rfn)
    ent_rows=[]; ent_f=[]
    for label in sorted(ent_labels):
        g=Counter({k:v for k,v in ge.items() if k[1]==label}); p=Counter({k:v for k,v in pe.items() if k[1]==label})
        tp,fp,fn=counts(g,p); pp,rrr,ff=f1(tp,fp,fn); ent_f.append(ff)
        ent_rows.append({"class":label,"precision":pp,"recall":rrr,"f1":ff,"support":sum(g.values())})
    rel_rows=[]; rel_f=[]
    for label in sorted(rel_labels):
        g=Counter({k:v for k,v in gr.items() if k[1]==label}); p=Counter({k:v for k,v in pr.items() if k[1]==label})
        tp,fp,fn=counts(g,p); pp,rrr,ff=f1(tp,fp,fn); rel_f.append(ff)
        rel_rows.append({"class":label,"precision":pp,"recall":rrr,"f1":ff,"support":sum(g.values())})
    return {
        "entity_precision":ep,"entity_recall":er,"entity_micro_f1":ef,"entity_macro_f1":sum(ent_f)/len(ent_f),
        "strict_relation_precision":rp,"strict_relation_recall":rr,"strict_relation_micro_f1":rf,"relation_macro_f1":sum(rel_f)/len(rel_f),
        "gold_entities":sum(ge.values()),"gold_relations":sum(gr.values()),"pred_entities":sum(pe.values()),"pred_relations":sum(pr.values()),
        "invalid_relation_endpoints":invalid_rel,"entity_per_class":ent_rows,"relation_per_class":rel_rows,
    }


def ids(docs):
    out=[]
    for i,d in enumerate(docs):
        rid=d.get("orig_id",d.get("ident"))
        if rid is None: raise SystemExit(f"test document {i} has no orig_id/ident")
        out.append(str(rid))
    return out


def parse_conf(path: Path):
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        m=re.match(r"^\s*([^#;\[=][^=]*?)\s*=\s*(.*?)\s*$",line)
        if m: out[m.group(1).strip()]=m.group(2).strip()
    return out


def newest_model(export: Path):
    models=list((export/"save").rglob("final_model")) if (export/"save").exists() else []
    models=[p for p in models if p.is_dir()]
    if not models: raise SystemExit(f"No final_model under {export/'save'}")
    return max(models,key=lambda p:p.stat().st_mtime)


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    prep=load_json(BASE/"PREP_MANIFEST.json")
    gold={}; pred={}; export={"raw":RAW, **{s:BASE/s for s in NORM_SYSTEMS}}
    for s in SYSTEMS:
        gp=export[s]/"test.json"; pp=export[s]/"predictions_test.json"
        if not gp.exists() or not pp.exists(): raise SystemExit(f"{s}: missing test or prediction artifact")
        gold[s]=load_json(gp); pred[s]=load_json(pp)
        if len(gold[s])!=EXPECTED_N or len(pred[s])!=EXPECTED_N: raise SystemExit(f"{s}: expected 225 test docs/preds")
    raw_ids=ids(gold["raw"])
    for s in NORM_SYSTEMS:
        if ids(gold[s]) != raw_ids: raise SystemExit(f"{s}: frozen TEST order/membership differs from raw")

    raw_support=(sum(len(d.get("entities",[])) for d in gold["raw"]),sum(len(d.get("relations",[])) for d in gold["raw"]))
    for s in NORM_SYSTEMS:
        sup=(sum(len(d.get("entities",[])) for d in gold[s]),sum(len(d.get("relations",[])) for d in gold[s]))
        if sup != raw_support: raise SystemExit(f"{s}: TEST gold support {sup} differs from raw {raw_support}")

    compare_keys=["model_type","model_path","tokenizer_path","train_batch_size","eval_batch_size","neg_entity_count","neg_relation_count","epochs","lr","lr_warmup","weight_decay","max_grad_norm","max_span_size","rel_filter_threshold","size_embedding","prop_drop","final_eval","store_predictions","store_examples","seed"]
    raw_conf=parse_conf(RAW/"avimaint_spert.conf")
    conf_checks={}
    for s in NORM_SYSTEMS:
        c=parse_conf(export[s]/"avimaint_spert.conf")
        diffs={k:{"raw":raw_conf.get(k),"variant":c.get(k)} for k in compare_keys if c.get(k)!=raw_conf.get(k)}
        conf_checks[s]={"same_hyperparameters":not bool(diffs),"differences":diffs}
        if diffs: raise SystemExit(f"{s}: hyperparameter mismatch vs raw config: {diffs}")

    metrics={}; overall=[]; ent_rows=[]; rel_rows=[]
    for s in SYSTEMS:
        m=evaluate(gold[s],pred[s]); metrics[s]=m
        overall.append({"variant":s,"entity_micro_f1":m["entity_micro_f1"],"entity_macro_f1":m["entity_macro_f1"],"strict_relation_micro_f1":m["strict_relation_micro_f1"],"relation_macro_f1":m["relation_macro_f1"],"n_test":EXPECTED_N,"gold_entities":m["gold_entities"],"gold_relations":m["gold_relations"]})
        ent_rows.extend({"variant":s,**x} for x in m["entity_per_class"])
        rel_rows.extend({"variant":s,**x} for x in m["relation_per_class"])

    rawm=metrics["raw"]
    parity={}; parity_ok=True
    for k,exp in EXPECTED_RAW.items():
        tol=0.001 if "micro" in k else 0.002
        obs=rawm[k]; ok=abs(obs-exp)<=tol; parity_ok &= ok
        parity[k]={"expected":exp,"observed":obs,"abs_difference":abs(obs-exp),"tolerance":tol,"pass":ok}
    if not parity_ok: raise SystemExit("RAW parity failed. Do not interpret matched normalization results.")

    base_row=overall[0]
    for row in overall:
        for k in ["entity_micro_f1","entity_macro_f1","strict_relation_micro_f1","relation_macro_f1"]:
            row["delta_vs_raw_"+k]=row[k]-base_row[k]

    models={}
    for s in SYSTEMS:
        mp=newest_model(export[s])
        models[s]={
            "normalization_system":s,
            "export_dir":str(export[s].relative_to(ROOT)),
            "final_model_path":str(mp.relative_to(ROOT)),
            "test_prediction_path":str((export[s]/"predictions_test.json").relative_to(ROOT)),
            "config_path":str((export[s]/"avimaint_spert.conf").relative_to(ROOT)),
            "metrics":{k:metrics[s][k] for k in ["entity_micro_f1","entity_macro_f1","strict_relation_micro_f1","relation_macro_f1"]},
        }

    REPORT.mkdir(parents=True,exist_ok=True)
    write_csv(REPORT/"matched_normalization_spert_ablation.csv",overall)
    write_csv(REPORT/"entity_per_class.csv",ent_rows)
    write_csv(REPORT/"relation_per_class.csv",rel_rows)
    dump_json(REPORT/"MODEL_REGISTRY.json",models)
    manifest={
        "status":"complete_no_post_test_retuning_allowed",
        "generated_at_utc":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "experiment":"representation_matched_normalization_to_full_spert",
        "systems":SYSTEMS,
        "normalized_models_trained":4,
        "raw_model_retrained":False,
        "raw_model_policy":"reuse authoritative pre-existing raw full-9x11 SpERT baseline",
        "frozen_split":{"train":1275,"dev":100,"test":225,"membership_source":"outputs/splits.json"},
        "schema":{"entities":9,"relations":11},
        "same_hyperparameters_vs_raw":conf_checks,
        "raw_parity_gate":{"pass":parity_ok,"checks":parity},
        "metrics":{s:{k:v for k,v in metrics[s].items() if k not in {"entity_per_class","relation_per_class"}} for s in SYSTEMS},
        "model_registry":"MODEL_REGISTRY.json",
        "preparation_manifest":prep,
        "interpretation_policy":{
            "primary_question":"Does normalization help the strongest IE architecture when train/dev/test representations are matched?",
            "test_metrics_calculated_only_after_all_four_normalized_models_and_predictions_exist":True,
            "post_test_hyperparameter_tuning":"forbidden",
            "automatic_winner_selection_for_deployment":False,
            "earlier_frozen_raw_model_on_normalized_test_experiment":"retain only as inference-time representation-shift sensitivity analysis",
        },
    }
    dump_json(REPORT/"FINAL_MATCHED_NORMALIZATION_SPERT_MANIFEST.json",manifest)

    print("="*108)
    print("REPRESENTATION-MATCHED NORMALIZATION -> FULL SpERT RESULTS")
    print("="*108)
    print(f"{'variant':18s} {'ent micro':>10s} {'ent macro':>10s} {'rel micro':>10s} {'rel macro':>10s} {'dEntMicro':>11s} {'dRelMicro':>11s}")
    print("-"*108)
    for r in overall:
        print(f"{r['variant']:18s} {r['entity_micro_f1']:10.4f} {r['entity_macro_f1']:10.4f} {r['strict_relation_micro_f1']:10.4f} {r['relation_macro_f1']:10.4f} {r['delta_vs_raw_entity_micro_f1']:11.4f} {r['delta_vs_raw_strict_relation_micro_f1']:11.4f}")
    print("-"*108)
    print("RAW parity: PASS")
    print("No automatic deployment winner was selected from TEST.")
    print(f"Reports -> {REPORT}")
    print(f"Model registry -> {REPORT/'MODEL_REGISTRY.json'}")

if __name__=="__main__": main()
