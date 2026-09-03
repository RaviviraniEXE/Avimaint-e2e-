"""Neural relation classifier over entity pairs for Tier 2.

Early stopping remains DEV-loss based.  Live progress reports current epoch,
batch progress, train loss, DEV loss, DEV relation F1 (gold-entity diagnostic),
patience and ETA without changing the optimisation protocol.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

from src.progress import EpochProgress, trace_event

NONE = "__NONE__"


class NeuralRelationClassifier:
    def __init__(self, schema: dict, vocab: dict[str, int], entity_types: list[str],
                 relation_types: list[str], seed: int = 42, embedding_dim: int = 100,
                 hidden_dim: int = 160, dropout: float = 0.4, lr: float = 1e-3):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed); random.seed(seed)
        self.schema, self.vocab = schema, vocab
        self.entity_types = entity_types
        self.labels = [NONE, *relation_types]
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.entity_to_id = {label: i for i, label in enumerate(entity_types)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = dict(seed=seed, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                           dropout=dropout, lr=lr)
        self.training_summary: dict = {}
        self.training_history: list[dict] = []

        class PairNet(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.word = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
                inner.ent = nn.Embedding(max(1, len(entity_types)), 24)
                inner.mlp = nn.Sequential(nn.Linear(embedding_dim * 3 + 48 + 2, hidden_dim),
                                          nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(hidden_dim, len(self.labels)))
            def forward(inner, doc, head, tail, etypes, scalars):
                emb = inner.word(doc)
                mask = (doc != 0).float().unsqueeze(-1)
                context = (emb * mask).sum(1) / mask.sum(1).clamp_min(1)
                h = inner.word(head).mean(1); t = inner.word(tail).mean(1)
                e = inner.ent(etypes).reshape(etypes.shape[0], -1)
                return inner.mlp(torch.cat([context, h, t, e, scalars], dim=1))
        self.net = PairNet().to(self.device)

    def _allowed(self, head: str, tail: str) -> list[str]:
        return [name for name, spec in self.schema.get("relations", {}).items()
                if name in self.label_to_id and head in spec.get("head", []) and tail in spec.get("tail", [])]

    def _examples(self, rows: list[dict], labelled: bool):
        examples = []
        for di, row in enumerate(rows):
            entities = row.get("entities", [])
            gold = {(r["head"], r["tail"]): r["type"] for r in row.get("relations", [])}
            for hi, head in enumerate(entities):
                for ti, tail in enumerate(entities):
                    if hi == ti or not self._allowed(head["type"], tail["type"]): continue
                    label = gold.get((hi, ti), NONE) if labelled else NONE
                    examples.append((di, hi, ti, row["tokens"], head, tail, label))
        return examples

    def _batch(self, examples):
        import torch
        max_doc = max(len(x[3]) for x in examples); max_head = max(x[4]["end"]-x[4]["start"] for x in examples)
        max_tail = max(x[5]["end"]-x[5]["start"] for x in examples)
        doc = torch.zeros((len(examples), max_doc), dtype=torch.long)
        htok = torch.zeros((len(examples), max_head), dtype=torch.long)
        ttok = torch.zeros((len(examples), max_tail), dtype=torch.long)
        etypes = torch.zeros((len(examples), 2), dtype=torch.long)
        scalars = torch.zeros((len(examples), 2), dtype=torch.float32)
        labels = torch.zeros(len(examples), dtype=torch.long)
        unk = self.vocab.get("<UNK>", 1)
        for i, (_, _, _, tokens, h, t, label) in enumerate(examples):
            ids = [self.vocab.get(x.lower(), unk) for x in tokens]; doc[i,:len(ids)] = torch.tensor(ids)
            hs = ids[h["start"]:h["end"]]; ts = ids[t["start"]:t["end"]]
            htok[i,:len(hs)] = torch.tensor(hs); ttok[i,:len(ts)] = torch.tensor(ts)
            etypes[i] = torch.tensor([self.entity_to_id.get(h["type"],0), self.entity_to_id.get(t["type"],0)])
            distance = min(20, abs(h["start"]-t["start"]))/20.0
            scalars[i] = torch.tensor([distance, float(h["start"] < t["start"])])
            labels[i] = self.label_to_id.get(label, 0)
        return tuple(x.to(self.device) for x in (doc, htok, ttok, etypes, scalars, labels))

    def _dev_relation_f1(self, dev: list[dict] | None) -> float | None:
        if not dev:
            return None
        from src.evaluate import relation_scores
        inputs = [{"tokens": r["tokens"], "entities": r.get("entities", [])} for r in dev]
        rels = self.predict(inputs)
        gold = [{"tokens": r["tokens"], "entities": r.get("entities", []), "relations": r.get("relations", [])}
                for r in dev]
        pred = [{"tokens": r["tokens"], "entities": r.get("entities", []), "relations": rp}
                for r, rp in zip(dev, rels)]
        return float(relation_scores(gold, pred)["micro_f1"])

    def fit(self, records: list[dict], dev: list[dict] | None = None, epochs: int = 40,
            patience: int = 6, batch_size: int = 64):
        import torch
        train = self._examples(records, True)
        if not train: return self
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.params["lr"])
        counts = [1] * len(self.labels)
        for x in train: counts[self.label_to_id.get(x[-1],0)] += 1
        weights = torch.tensor([len(train)/c for c in counts], dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights/weights.mean())
        best_loss, best_state, stale = float("inf"), None, 0; rng = random.Random(self.params["seed"])
        best_epoch, stopped_epoch = 0, 0
        monitor = EpochProgress("BiLSTM Neural RE", epochs=epochs, patience=patience)
        trace_event("model_fit_context", model="BiLSTM Neural RE", device=str(self.device),
                    records=len(records), train_pairs=len(train), dev_records=len(dev or []),
                    batch_size=batch_size, lr=self.params["lr"], patience=patience,
                    early_stopping_metric="dev_loss")

        check = self._examples(dev or records, True)
        for epoch in range(1, epochs + 1):
            rng.shuffle(train); self.net.train()
            total_batches = (len(train) + batch_size - 1) // batch_size
            monitor.start_epoch(epoch, total_batches)
            running_loss = 0.0; batches = 0
            for start in range(0,len(train),batch_size):
                d,h,t,e,s,y=self._batch(train[start:start+batch_size]); optimizer.zero_grad()
                loss=loss_fn(self.net(d,h,t,e,s),y); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(),5); optimizer.step()
                batches += 1; running_loss += float(loss.item())
                monitor.batch(epoch, batches, total_batches, running_loss / batches)
            self.net.eval(); losses=[]
            with torch.no_grad():
                for start in range(0,len(check),batch_size):
                    d,h,t,e,s,y=self._batch(check[start:start+batch_size]); losses.append(float(loss_fn(self.net(d,h,t,e,s),y)))
            current=sum(losses)/max(1,len(losses)); train_loss=running_loss/max(1,batches)
            dev_f1=self._dev_relation_f1(dev)
            improved = current < best_loss-1e-5
            if improved:
                best_loss,best_state,stale,best_epoch=current,deepcopy(self.net.state_dict()),0,epoch
            else:
                stale+=1
            self.training_history.append({"epoch":epoch,"train_loss":round(train_loss,8),
                                          "dev_loss":round(current,8),
                                          "dev_f1":None if dev_f1 is None else round(dev_f1,8),
                                          "stale":stale,"improved":bool(improved)})
            monitor.finish_epoch(epoch,train_loss,current,dev_f1,best_loss,stale,improved,best_epoch)
            stopped_epoch=epoch
            if stale>=patience:
                monitor.early_stop(epoch,best_epoch,best_loss); break
        if best_state is not None: self.net.load_state_dict(best_state)
        self.training_summary={"model":"BiLSTM Neural RE","device":str(self.device),
                               "max_epochs":int(epochs),"stopped_epoch":int(stopped_epoch),
                               "best_epoch":int(best_epoch),"best_dev_loss":round(float(best_loss),8),
                               "patience":int(patience),"early_stopped":bool(stopped_epoch<epochs),
                               "early_stopping_metric":"dev_loss","batch_size":int(batch_size),
                               "train_pairs":len(train),"params":dict(self.params),
                               "history":list(self.training_history)}
        monitor.close(stopped_epoch,best_epoch,best_loss)
        return self

    def predict(self, records: list[dict], batch_size: int = 128) -> list[list[dict]]:
        import torch
        examples=self._examples(records,False); output=[[] for _ in records]
        self.net.eval()
        with torch.no_grad():
            for start in range(0,len(examples),batch_size):
                chunk=examples[start:start+batch_size]
                d,h,t,e,s,_=self._batch(chunk); pred=self.net(d,h,t,e,s).argmax(1).tolist()
                for ex,label_id in zip(chunk,pred):
                    label=self.labels[label_id]
                    if label!=NONE and label in self._allowed(ex[4]["type"],ex[5]["type"]):
                        output[ex[0]].append({"type":label,"head":ex[1],"tail":ex[2]})
        return output

    def save(self,path:str):
        import torch
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"state":self.net.state_dict(),"vocab":self.vocab,"labels":self.labels,
                    "entity_types":self.entity_types,"params":self.params,
                    "training_summary":self.training_summary},target)
