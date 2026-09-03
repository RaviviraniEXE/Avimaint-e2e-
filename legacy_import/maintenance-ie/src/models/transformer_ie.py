"""Hugging Face token- and pair-classification baselines for Tier 3.

Training uses DEV-loss early stopping exactly as before.  This version adds live
epoch/batch progress, train loss, DEV loss, DEV F1 diagnostics, patience state and
machine-readable training summaries.  It also releases non-selected transformer
models from GPU memory between tuning configurations.
"""
from __future__ import annotations

from copy import deepcopy
import random

from src.progress import EpochProgress, trace_event

NONE = "__NONE__"


class TransformerNER:
    def __init__(self, tags, model_name="distilbert-base-uncased", max_len=128, seed=42):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        random.seed(seed); torch.manual_seed(seed)
        self.tags=list(tags); self.tag_to_id={x:i for i,x in enumerate(self.tags)}
        self.model_name=model_name; self.max_len=max_len; self.seed=seed
        self.tokenizer=AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model=AutoModelForTokenClassification.from_pretrained(
            model_name,num_labels=len(tags),id2label={i:x for i,x in enumerate(tags)},label2id=self.tag_to_id)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.model.to(self.device)
        self.training_summary={}; self.training_history=[]

    def _encode(self, rows, labelled=True):
        import torch
        encoded=self.tokenizer([r["tokens"] for r in rows],is_split_into_words=True,padding=True,
                               truncation=True,max_length=self.max_len,return_tensors="pt")
        labels=[]
        if labelled:
            for bi,row in enumerate(rows):
                word_ids=encoded.word_ids(batch_index=bi); last=None; aligned=[]
                for wid in word_ids:
                    if wid is None: aligned.append(-100)
                    elif wid!=last: aligned.append(self.tag_to_id.get(row["bio"][wid],self.tag_to_id["O"]))
                    else: aligned.append(-100)
                    last=wid
                labels.append(aligned)
            encoded["labels"]=torch.tensor(labels)
        return encoded

    def _dev_entity_f1(self, dev):
        if not dev: return None
        from src.evaluate import entity_scores
        from src.models.crf_ner import bio_to_entities
        tags=self.predict(dev)
        gold=[{"tokens":r["tokens"],"entities":bio_to_entities(r["tokens"],r["bio"]),"relations":[]} for r in dev]
        pred=[{"tokens":r["tokens"],"entities":bio_to_entities(r["tokens"],p),"relations":[]} for r,p in zip(dev,tags)]
        return float(entity_scores(gold,pred)["micro_f1"])

    def fit(self, records, dev=None, epochs=10, lr=3e-5, batch_size=16, patience=3):
        import torch
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=lr)
        best,best_state,stale=float("inf"),None,0; rng=random.Random(self.seed)
        best_epoch=0; stopped_epoch=0
        monitor=EpochProgress("Transformer NER",epochs=epochs,patience=patience)
        trace_event("model_fit_context",model="Transformer NER",encoder=self.model_name,device=str(self.device),
                    records=len(records),dev_records=len(dev or []),batch_size=batch_size,lr=float(lr),
                    max_len=self.max_len,patience=patience,early_stopping_metric="dev_loss")
        for epoch in range(1,epochs+1):
            order=list(range(len(records))); rng.shuffle(order); self.model.train()
            total_batches=(len(order)+batch_size-1)//batch_size
            monitor.start_epoch(epoch,total_batches)
            running=0.0; batches=0
            for start in range(0,len(order),batch_size):
                batch=[records[i] for i in order[start:start+batch_size]]; enc=self._encode(batch)
                enc={k:v.to(self.device) for k,v in enc.items()}; optimizer.zero_grad()
                loss=self.model(**enc).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),1); optimizer.step()
                batches+=1; running+=float(loss.item()); monitor.batch(epoch,batches,total_batches,running/batches)
            check=dev or records; self.model.eval(); losses=[]
            with torch.no_grad():
                for start in range(0,len(check),batch_size):
                    enc={k:v.to(self.device) for k,v in self._encode(check[start:start+batch_size]).items()}
                    losses.append(float(self.model(**enc).loss))
            current=sum(losses)/max(1,len(losses)); train_loss=running/max(1,batches)
            dev_f1=self._dev_entity_f1(dev)
            improved=current<best-1e-5
            if improved: best,best_state,stale,best_epoch=current,deepcopy(self.model.state_dict()),0,epoch
            else: stale+=1
            self.training_history.append({"epoch":epoch,"train_loss":round(train_loss,8),"dev_loss":round(current,8),
                                          "dev_f1":None if dev_f1 is None else round(dev_f1,8),
                                          "stale":stale,"improved":bool(improved)})
            monitor.finish_epoch(epoch,train_loss,current,dev_f1,best,stale,improved,best_epoch)
            stopped_epoch=epoch
            if stale>=patience:
                monitor.early_stop(epoch,best_epoch,best); break
        if best_state is not None:self.model.load_state_dict(best_state)
        self.training_summary={"model":"Transformer NER","encoder":self.model_name,"device":str(self.device),
                               "max_epochs":int(epochs),"stopped_epoch":int(stopped_epoch),"best_epoch":int(best_epoch),
                               "best_dev_loss":round(float(best),8),"patience":int(patience),
                               "early_stopped":bool(stopped_epoch<epochs),"early_stopping_metric":"dev_loss",
                               "batch_size":int(batch_size),"lr":float(lr),"max_len":int(self.max_len),
                               "history":list(self.training_history)}
        monitor.close(stopped_epoch,best_epoch,best)
        return self

    def predict(self,records,batch_size=32):
        import torch
        out=[]; self.model.eval()
        with torch.no_grad():
            for start in range(0,len(records),batch_size):
                chunk=records[start:start+batch_size]; enc=self._encode(chunk,False)
                device_enc={k:v.to(self.device) for k,v in enc.items()}; logits=self.model(**device_enc).logits.argmax(-1).cpu()
                for bi,row in enumerate(chunk):
                    word_ids=enc.word_ids(batch_index=bi); tags=[]; seen=set()
                    for ti,wid in enumerate(word_ids):
                        if wid is not None and wid not in seen:
                            tags.append(self.tags[int(logits[bi,ti])]); seen.add(wid)
                    tags += ["O"]*(len(row["tokens"])-len(tags)); out.append(tags[:len(row["tokens"])])
        return out

    @classmethod
    def tuned(cls,tags,train,dev,model_name="distilbert-base-uncased",max_len=128,lrs=None,**fit_args):
        import torch
        from src.evaluate import entity_scores
        from src.models.crf_ner import bio_to_entities
        best_model,best=None,{"micro_f1":-1.0,"history":[]}; lrs=[float(x) for x in (lrs or [3e-5])]
        gold=[{"tokens":r["tokens"],"entities":bio_to_entities(r["tokens"],r["bio"]),"relations":[]} for r in dev]
        for idx,lr in enumerate(lrs,1):
            print(f"\n[Transformer NER tuning] config {idx}/{len(lrs)} | lr={lr:g} | encoder={model_name}",flush=True)
            model=cls(tags,model_name,max_len).fit(train,dev,lr=lr,**fit_args)
            pred=[{"tokens":r["tokens"],"entities":bio_to_entities(r["tokens"],p),"relations":[]} for r,p in zip(dev,model.predict(dev))]
            score=float(entity_scores(gold,pred)["micro_f1"])
            item={"lr":lr,"micro_f1":score,"training_summary":model.training_summary}; best["history"].append(item)
            is_best=score>best["micro_f1"]
            print(f"[Transformer NER tuning] DEV entity micro-F1={score:.4f}{' | BEST' if is_best else ''}",flush=True)
            if is_best:
                if best_model is not None:
                    try: best_model.model.to("cpu")
                    except Exception: pass
                best_model=model; best.update({"lr":lr,"micro_f1":score,"training_summary":model.training_summary})
            else:
                try:model.model.to("cpu")
                except Exception:pass
                del model
                if torch.cuda.is_available():torch.cuda.empty_cache()
        return best_model,best


class TransformerRE:
    def __init__(self,schema,entity_types,relation_types,model_name="distilbert-base-uncased",max_len=128,seed=42):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.schema=schema; self.labels=[NONE,*relation_types]; self.label_to_id={x:i for i,x in enumerate(self.labels)}
        self.model_name=model_name; self.max_len=max_len; self.seed=seed
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,use_fast=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=len(self.labels),
                    id2label={i:x for i,x in enumerate(self.labels)},label2id=self.label_to_id)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.model.to(self.device)
        self.training_summary={}; self.training_history=[]

    def _allowed(self,h,t):
        return [n for n,s in self.schema.get("relations",{}).items() if n in self.label_to_id and h in s.get("head",[]) and t in s.get("tail",[])]

    def _examples(self,rows,labelled):
        out=[]
        for di,row in enumerate(rows):
            entities=row.get("entities",[]); gold={(r["head"],r["tail"]):r["type"] for r in row.get("relations",[])}
            for hi,h in enumerate(entities):
                for ti,t in enumerate(entities):
                    if hi==ti or not self._allowed(h["type"],t["type"]):continue
                    tokens=list(row["tokens"]); inserts=[(h["start"],"[HEAD]"),(h["end"],"[/HEAD]"),(t["start"],"[TAIL]"),(t["end"],"[/TAIL]")]
                    for idx,mark in sorted(inserts,reverse=True):tokens.insert(idx,mark)
                    text=" ".join(tokens)+f" [PAIR] {h['type']} {t['type']}"
                    out.append((di,hi,ti,h,t,text,gold.get((hi,ti),NONE) if labelled else NONE))
        return out

    def _dev_relation_f1(self,dev):
        if not dev:return None
        from src.evaluate import relation_scores
        inputs=[{"tokens":r["tokens"],"entities":r.get("entities",[])} for r in dev]
        rels=self.predict(inputs)
        gold=[{"tokens":r["tokens"],"entities":r.get("entities",[]),"relations":r.get("relations",[])} for r in dev]
        pred=[{"tokens":r["tokens"],"entities":r.get("entities",[]),"relations":rp} for r,rp in zip(dev,rels)]
        return float(relation_scores(gold,pred)["micro_f1"])

    def fit(self,records,dev=None,epochs=10,lr=3e-5,batch_size=8,patience=3):
        import torch
        train=self._examples(records,True)
        if not train:return self
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=lr); best,best_state,stale=float("inf"),None,0; rng=random.Random(self.seed)
        best_epoch=0; stopped_epoch=0
        check=self._examples(dev or records,True)
        monitor=EpochProgress("Transformer RE",epochs=epochs,patience=patience)
        trace_event("model_fit_context",model="Transformer RE",encoder=self.model_name,device=str(self.device),
                    records=len(records),train_pairs=len(train),dev_records=len(dev or []),batch_size=batch_size,
                    lr=float(lr),max_len=self.max_len,patience=patience,early_stopping_metric="dev_loss")
        for epoch in range(1,epochs+1):
            rng.shuffle(train);self.model.train(); total_batches=(len(train)+batch_size-1)//batch_size
            monitor.start_epoch(epoch,total_batches); running=0.0;batches=0
            for start in range(0,len(train),batch_size):
                chunk=train[start:start+batch_size]; enc=self.tokenizer([x[5] for x in chunk],padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
                enc={k:v.to(self.device) for k,v in enc.items()}; labels=torch.tensor([self.label_to_id[x[6]] for x in chunk],device=self.device)
                optimizer.zero_grad();loss=self.model(**enc,labels=labels).loss;loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1);optimizer.step()
                batches+=1;running+=float(loss.item());monitor.batch(epoch,batches,total_batches,running/batches)
            self.model.eval();losses=[]
            with torch.no_grad():
                for start in range(0,len(check),batch_size):
                    chunk=check[start:start+batch_size];enc=self.tokenizer([x[5] for x in chunk],padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
                    enc={k:v.to(self.device) for k,v in enc.items()};labels=torch.tensor([self.label_to_id[x[6]] for x in chunk],device=self.device)
                    losses.append(float(self.model(**enc,labels=labels).loss))
            current=sum(losses)/max(1,len(losses));train_loss=running/max(1,batches);dev_f1=self._dev_relation_f1(dev)
            improved=current<best-1e-5
            if improved:best,best_state,stale,best_epoch=current,deepcopy(self.model.state_dict()),0,epoch
            else:stale+=1
            self.training_history.append({"epoch":epoch,"train_loss":round(train_loss,8),"dev_loss":round(current,8),
                                          "dev_f1":None if dev_f1 is None else round(dev_f1,8),"stale":stale,"improved":bool(improved)})
            monitor.finish_epoch(epoch,train_loss,current,dev_f1,best,stale,improved,best_epoch);stopped_epoch=epoch
            if stale>=patience:
                monitor.early_stop(epoch,best_epoch,best);break
        if best_state is not None:self.model.load_state_dict(best_state)
        self.training_summary={"model":"Transformer RE","encoder":self.model_name,"device":str(self.device),
                               "max_epochs":int(epochs),"stopped_epoch":int(stopped_epoch),"best_epoch":int(best_epoch),
                               "best_dev_loss":round(float(best),8),"patience":int(patience),
                               "early_stopped":bool(stopped_epoch<epochs),"early_stopping_metric":"dev_loss",
                               "batch_size":int(batch_size),"lr":float(lr),"max_len":int(self.max_len),
                               "train_pairs":len(train),"history":list(self.training_history)}
        monitor.close(stopped_epoch,best_epoch,best)
        return self

    def predict(self,records,batch_size=32):
        import torch
        examples=self._examples(records,False);out=[[] for _ in records];self.model.eval()
        with torch.no_grad():
            for start in range(0,len(examples),batch_size):
                chunk=examples[start:start+batch_size];enc=self.tokenizer([x[5] for x in chunk],padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
                pred=self.model(**{k:v.to(self.device) for k,v in enc.items()}).logits.argmax(-1).tolist()
                for ex,lid in zip(chunk,pred):
                    label=self.labels[lid]
                    if label!=NONE and label in self._allowed(ex[3]["type"],ex[4]["type"]):out[ex[0]].append({"type":label,"head":ex[1],"tail":ex[2]})
        return out
