"""Neural relation classifier over entity pairs for Tier 2."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random


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

    def fit(self, records: list[dict], dev: list[dict] | None = None, epochs: int = 40,
            patience: int = 6, batch_size: int = 64):
        import torch
        from tqdm.auto import tqdm
        train = self._examples(records, True)
        if not train: return self
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.params["lr"])
        counts = [1] * len(self.labels)
        for x in train: counts[self.label_to_id.get(x[-1],0)] += 1
        weights = torch.tensor([len(train)/c for c in counts], dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights/weights.mean())
        best_loss, best_state, stale = float("inf"), None, 0; rng = random.Random(self.params["seed"])
        total_batches = (len(train) + batch_size - 1) // batch_size
        print(f"[Tier2 RE] device={self.device} | train_pairs={len(train)} | batches/epoch={total_batches}")
        for epoch in range(1, epochs + 1):
            rng.shuffle(train); self.net.train(); running = 0.0
            bar = tqdm(range(0,len(train),batch_size), total=total_batches,
                       desc=f"Tier2 RE  epoch {epoch:02d}/{epochs}", unit="batch",
                       dynamic_ncols=True, leave=True)
            for step, start in enumerate(bar, 1):
                d,h,t,e,s,y=self._batch(train[start:start+batch_size]); optimizer.zero_grad()
                loss=loss_fn(self.net(d,h,t,e,s),y); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(),5); optimizer.step()
                running += float(loss.item()); bar.set_postfix(train_loss=f"{running/step:.4f}")
            check=self._examples(dev or records,True); self.net.eval()
            with torch.no_grad():
                losses=[]
                val_bar = tqdm(range(0,len(check),batch_size),
                               total=(len(check)+batch_size-1)//batch_size,
                               desc=f"Tier2 RE  dev   {epoch:02d}/{epochs}", unit="batch",
                               dynamic_ncols=True, leave=False)
                for start in val_bar:
                    d,h,t,e,s,y=self._batch(check[start:start+batch_size]); losses.append(float(loss_fn(self.net(d,h,t,e,s),y)))
            current=sum(losses)/max(1,len(losses))
            if current < best_loss-1e-5:
                best_loss,best_state,stale=current,deepcopy(self.net.state_dict()),0
            else:
                stale+=1
            print(f"[Tier2 RE] epoch={epoch:02d} dev_loss={current:.6f} best={best_loss:.6f} patience={stale}/{patience}")
            if stale>=patience:
                print(f"[Tier2 RE] early stopping at epoch {epoch}; restoring best checkpoint")
                break
        if best_state is not None: self.net.load_state_dict(best_state)
        return self

    def predict(self, records: list[dict], batch_size: int = 128) -> list[list[dict]]:
        import torch
        from tqdm.auto import tqdm
        examples=self._examples(records,False); output=[[] for _ in records]
        self.net.eval()
        with torch.no_grad():
            bar = tqdm(range(0,len(examples),batch_size),
                       total=(len(examples)+batch_size-1)//batch_size if examples else 0,
                       desc="Tier2 RE  predict", unit="batch", dynamic_ncols=True, leave=True)
            for start in bar:
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
                    "entity_types":self.entity_types,"params":self.params},target)
