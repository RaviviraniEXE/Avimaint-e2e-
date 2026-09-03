"""Hugging Face token- and pair-classification baselines for Tier 3."""
from __future__ import annotations

from copy import deepcopy
import random


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

    def fit(self, records, dev=None, epochs=10, lr=3e-5, batch_size=16, patience=3):
        import torch
        from tqdm.auto import tqdm
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=lr)
        best,best_state,stale=float("inf"),None,0; rng=random.Random(self.seed)
        total_batches=(len(records)+batch_size-1)//batch_size
        print(f"[Tier3 NER] device={self.device} | train_docs={len(records)} | batches/epoch={total_batches}")
        for epoch in range(1,epochs+1):
            order=list(range(len(records))); rng.shuffle(order); self.model.train(); running=0.0
            bar=tqdm(range(0,len(order),batch_size),total=total_batches,
                     desc=f"Tier3 NER epoch {epoch:02d}/{epochs}",unit="batch",dynamic_ncols=True,leave=True)
            for step,start in enumerate(bar,1):
                batch=[records[i] for i in order[start:start+batch_size]]; enc=self._encode(batch)
                enc={k:v.to(self.device) for k,v in enc.items()}; optimizer.zero_grad()
                loss=self.model(**enc).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),1); optimizer.step()
                running+=float(loss.item());bar.set_postfix(train_loss=f"{running/step:.4f}")
            check=dev or records; self.model.eval(); losses=[]
            with torch.no_grad():
                val_bar=tqdm(range(0,len(check),batch_size),total=(len(check)+batch_size-1)//batch_size,
                             desc=f"Tier3 NER dev   {epoch:02d}/{epochs}",unit="batch",dynamic_ncols=True,leave=False)
                for start in val_bar:
                    enc={k:v.to(self.device) for k,v in self._encode(check[start:start+batch_size]).items()}
                    losses.append(float(self.model(**enc).loss))
            current=sum(losses)/max(1,len(losses))
            if current<best-1e-5: best,best_state,stale=current,deepcopy(self.model.state_dict()),0
            else: stale+=1
            print(f"[Tier3 NER] epoch={epoch:02d} dev_loss={current:.6f} best={best:.6f} patience={stale}/{patience}")
            if stale>=patience:
                print(f"[Tier3 NER] early stopping at epoch {epoch}; restoring best checkpoint")
                break
        if best_state is not None:self.model.load_state_dict(best_state)
        return self

    def predict(self,records,batch_size=32):
        import torch
        from tqdm.auto import tqdm
        out=[]; self.model.eval()
        with torch.no_grad():
            bar=tqdm(range(0,len(records),batch_size),total=(len(records)+batch_size-1)//batch_size if records else 0,
                     desc="Tier3 NER predict",unit="batch",dynamic_ncols=True,leave=True)
            for start in bar:
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
        from src.evaluate import entity_scores
        from src.models.crf_ner import bio_to_entities
        best_model,best=None,{"micro_f1":-1.0}; lrs=lrs or [3e-5]
        gold=[{"tokens":r["tokens"],"entities":bio_to_entities(r["tokens"],r["bio"]),"relations":[]} for r in dev]
        for lr in lrs:
            model=cls(tags,model_name,max_len).fit(train,dev,lr=float(lr),**fit_args)
            pred=[{"tokens":r["tokens"],"entities":bio_to_entities(r["tokens"],p),"relations":[]} for r,p in zip(dev,model.predict(dev))]
            score=entity_scores(gold,pred)["micro_f1"]
            if score>best["micro_f1"]:best_model,best=model,{"lr":float(lr),"micro_f1":score}
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

    def fit(self,records,dev=None,epochs=10,lr=3e-5,batch_size=8,patience=3):
        import torch
        from tqdm.auto import tqdm
        train=self._examples(records,True)
        if not train:return self
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=lr); best,best_state,stale=float("inf"),None,0; rng=random.Random(self.seed)
        total_batches=(len(train)+batch_size-1)//batch_size
        print(f"[Tier3 RE] device={self.device} | train_pairs={len(train)} | batches/epoch={total_batches}")
        for epoch in range(1,epochs+1):
            rng.shuffle(train);self.model.train();running=0.0
            bar=tqdm(range(0,len(train),batch_size),total=total_batches,
                     desc=f"Tier3 RE  epoch {epoch:02d}/{epochs}",unit="batch",dynamic_ncols=True,leave=True)
            for step,start in enumerate(bar,1):
                chunk=train[start:start+batch_size]; enc=self.tokenizer([x[5] for x in chunk],padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
                enc={k:v.to(self.device) for k,v in enc.items()}; labels=torch.tensor([self.label_to_id[x[6]] for x in chunk],device=self.device)
                optimizer.zero_grad();loss=self.model(**enc,labels=labels).loss;loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1);optimizer.step()
                running+=float(loss.item());bar.set_postfix(train_loss=f"{running/step:.4f}")
            check=self._examples(dev or records,True);self.model.eval();losses=[]
            with torch.no_grad():
                val_bar=tqdm(range(0,len(check),batch_size),total=(len(check)+batch_size-1)//batch_size,
                             desc=f"Tier3 RE  dev   {epoch:02d}/{epochs}",unit="batch",dynamic_ncols=True,leave=False)
                for start in val_bar:
                    chunk=check[start:start+batch_size];enc=self.tokenizer([x[5] for x in chunk],padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
                    enc={k:v.to(self.device) for k,v in enc.items()};labels=torch.tensor([self.label_to_id[x[6]] for x in chunk],device=self.device)
                    losses.append(float(self.model(**enc,labels=labels).loss))
            current=sum(losses)/max(1,len(losses))
            if current<best-1e-5:best,best_state,stale=current,deepcopy(self.model.state_dict()),0
            else: stale+=1
            print(f"[Tier3 RE] epoch={epoch:02d} dev_loss={current:.6f} best={best:.6f} patience={stale}/{patience}")
            if stale>=patience:
                print(f"[Tier3 RE] early stopping at epoch {epoch}; restoring best checkpoint")
                break
        if best_state is not None:self.model.load_state_dict(best_state)
        return self

    def predict(self,records,batch_size=32):
        import torch
        from tqdm.auto import tqdm
        examples=self._examples(records,False);out=[[] for _ in records];self.model.eval()
        with torch.no_grad():
            bar=tqdm(range(0,len(examples),batch_size),total=(len(examples)+batch_size-1)//batch_size if examples else 0,
                     desc="Tier3 RE  predict",unit="batch",dynamic_ncols=True,leave=True)
            for start in bar:
                chunk=examples[start:start+batch_size];enc=self.tokenizer([x[5] for x in chunk],padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
                pred=self.model(**{k:v.to(self.device) for k,v in enc.items()}).logits.argmax(-1).tolist()
                for ex,lid in zip(chunk,pred):
                    label=self.labels[lid]
                    if label!=NONE and label in self._allowed(ex[3]["type"],ex[4]["type"]):out[ex[0]].append({"type":label,"head":ex[1],"tail":ex[2]})
        return out
