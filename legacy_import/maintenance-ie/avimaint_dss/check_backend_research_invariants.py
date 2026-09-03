from __future__ import annotations
import inspect
from pathlib import Path
import yaml
from core.corpus import load_corpus
from core.retrieval import Retriever, Hit
from core.calibration import RQ5AgreementCalibrator
ROOT=Path(__file__).resolve().parent
cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
def accepts(fn,name):
    sig=inspect.signature(fn)
    return name in sig.parameters or any(p.kind==inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
for required in ("weights","dense_model","rrf_k"):
    assert accepts(Retriever.__init__,required), f"Retriever.__init__ lacks required parameter: {required}"
for required in ("top_k","q_entity_types","q_relation_types","raw_query","mode","diversify"):
    assert accepts(Retriever.search,required), f"Retriever.search lacks required parameter: {required}"

hit_fields = set(getattr(Hit, "__dataclass_fields__", {})) | set(getattr(Hit, "__annotations__", {}))
for required in ("idx", "score", "text_sim", "struct", "channels"):
    assert required in hit_fields, f"Retriever Hit lacks required field: {required}"
pred=cfg["data"].get("problem_predictions_path") or None
protocol=cfg["data"].get("protocol_path") or None
corpus=load_corpus(ROOT/cfg["data"]["csv_path"], mode=cfg["extraction"]["mode"], spert_url=cfg["extraction"]["spert_url"], predictions_path=(ROOT/pred).resolve() if pred else None, protocol_path=(ROOT/protocol).resolve() if protocol else None, require_predictions=cfg["extraction"].get("require_problem_predictions",False), normalize=False)
expected_full=int(cfg.get("research",{}).get("expected_corpus_rows",6169))
expected_train=int(cfg.get("research",{}).get("expected_train_rows",4319))
split=str(cfg.get("research",{}).get("validated_candidate_split","train"))
assert len(corpus.df)==expected_full, f"Corpus drift: expected {expected_full}, found {len(corpus.df)}"
assert "frozen_split" in corpus.df.columns, "frozen_split column missing"
required_columns = {"ident", "problem", "problem_norm", "action", "action_family", "outcome", "cluster_id", "components", "faults"}
missing_columns = sorted(required_columns - set(corpus.df.columns))
assert not missing_columns, f"Corpus is missing required DSS columns: {missing_columns}"
train_rows=int((corpus.df["frozen_split"]==split).sum())
assert train_rows==expected_train, f"Evidence split drift: expected {expected_train} {split} rows, found {train_rows}"
cal_rel=cfg["research"]["rq5_dev_predictions_path"]
cal=RQ5AgreementCalibrator((ROOT/cal_rel).resolve())
assert cal.available(), cal.status()
print("BACKEND_RESEARCH_INVARIANTS_OK")
print("CORPUS_ROWS",len(corpus.df))
print("EVIDENCE_SPLIT",split,train_rows)
print("RQ5_CALIBRATOR",cal.status())
print("RETRIEVER_CONTRACT_OK")
