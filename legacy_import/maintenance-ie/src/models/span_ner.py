"""Span-classification NER heads used by the MaintIE nesting ablation.

Unlike BIO tagging, these models score every candidate span independently and
therefore may return overlapping or nested entities.
"""
from __future__ import annotations

from copy import deepcopy
import random


NONE = "__NONE__"


def _candidates(row, max_span, training=False, seed=42):
    gold={(e["start"],e["end"]):e["type"] for e in row.get("entities",[])}
    positives=[]; negatives=[]
    for start in range(len(row["tokens"])):
        for end in range(start+1,min(len(row["tokens"]),start+max_span)+1):
            item=(start,end,gold.get((start,end),NONE))
            (positives if item[2]!=NONE else negatives).append(item)
    if training and positives:
        rng=random.Random(seed+len(row["tokens"]));rng.shuffle(negatives)
        negatives=negatives[:max(20,5*len(positives))]
    return positives+negatives


class BiLSTMSpanNER:
    def __init__(self,vocab,entity_types,max_span=10,pretrained=None,seed=42,
                 word_dim=100,hidden_dim=192,dropout=0.4):
        import torch
        import torch.nn as nn
        self.vocab=vocab;self.labels=[NONE,*entity_types];self.label_to_id={x:i for i,x in enumerate(self.labels)}
        self.max_span=max_span;self.seed=seed;self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class Net(nn.Module):
            def __init__(inner):
                super().__init__();inner.emb=nn.Embedding(len(vocab),word_dim,padding_idx=0)
                if pretrained is not None and tuple(pretrained.shape)==tuple(inner.emb.weight.shape):inner.emb.weight.data.copy_(torch.as_tensor(pretrained))
                inner.lstm=nn.LSTM(word_dim,hidden_dim//2,batch_first=True,bidirectional=True)
                inner.out=nn.Sequential(nn.Linear(hidden_dim*3+1,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,len(self.labels)))
            def encode(inner,ids):return inner.lstm(inner.emb(ids))[0]
            def score(inner,h,start,end):
                rows=[]
                for bi,(s,e) in enumerate(zip(start.tolist(),end.tolist())):
                    rows.append(torch.cat([h[bi,s],h[bi,e-1],h[bi,s:e].mean(0),torch.tensor([(e-s)/self.max_span],device=h.device)]))
                return inner.out(torch.stack(rows))
        torch.manual_seed(seed);self.net=Net().to(self.device)

    def _examples(self,rows,training=False):
        return [(row,s,e,label) for row in rows for s,e,label in _candidates(row,self.max_span,training,self.seed)]

    def _batch(self,examples):
        import torch
        n=max(len(x[0]["tokens"]) for x in examples);ids=torch.zeros((len(examples),n),dtype=torch.long)
        unk=self.vocab.get("<UNK>",1)
        for i,(row,_,_,_) in enumerate(examples):
            x=[self.vocab.get(t.lower(),unk) for t in row["tokens"]];ids[i,:len(x)]=torch.tensor(x)
        return (ids.to(self.device),torch.tensor([x[1] for x in examples],device=self.device),
                torch.tensor([x[2] for x in examples],device=self.device),
                torch.tensor([self.label_to_id[x[3]] for x in examples],device=self.device))

    def fit(self,records,dev=None,epochs=30,lr=1e-3,batch_size=32,patience=6):
        import torch
        train=self._examples(records,True);opt=torch.optim.AdamW(self.net.parameters(),lr=lr);lossfn=torch.nn.CrossEntropyLoss()
        best,best_state,stale=float("inf"),None,0;rng=random.Random(self.seed)
        for _ in range(epochs):
            rng.shuffle(train);self.net.train()
            for i in range(0,len(train),batch_size):
                ids,s,e,y=self._batch(train[i:i+batch_size]);opt.zero_grad();loss=lossfn(self.net.score(self.net.encode(ids),s,e),y);loss.backward();opt.step()
            check=self._examples(dev or records,True);self.net.eval();vals=[]
            with torch.no_grad():
                for i in range(0,len(check),batch_size):
                    ids,s,e,y=self._batch(check[i:i+batch_size]);vals.append(float(lossfn(self.net.score(self.net.encode(ids),s,e),y)))
            current=sum(vals)/max(1,len(vals))
            if current<best-1e-5:best,best_state,stale=current,deepcopy(self.net.state_dict()),0
            else:
                stale+=1
                if stale>=patience:break
        if best_state:self.net.load_state_dict(best_state)
        return self

    def predict(self,records,batch_size=64):
        import torch
        output=[];self.net.eval()
        with torch.no_grad():
            for row in records:
                examples=[(row,s,e,l) for s,e,l in _candidates(row,self.max_span)];entities=[]
                for i in range(0,len(examples),batch_size):
                    chunk=examples[i:i+batch_size];ids,s,e,_=self._batch(chunk);p=self.net.score(self.net.encode(ids),s,e).softmax(-1);labels=p.argmax(-1).tolist()
                    for ex,lid,confidence in zip(chunk,labels,p.max(-1).values.tolist()):
                        if lid and confidence>=0.5:entities.append({"type":self.labels[lid],"start":ex[1],"end":ex[2]})
                output.append(entities)
        return output


class TransformerSpanNER:
    def __init__(self,entity_types,model_name="distilbert-base-uncased",max_len=128,max_span=10,seed=42):
        import torch
        import torch.nn as nn
        from transformers import AutoModel,AutoTokenizer
        self.labels=[NONE,*entity_types];self.label_to_id={x:i for i,x in enumerate(self.labels)};self.max_span=max_span;self.max_len=max_len;self.seed=seed
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,use_fast=True);self.encoder=AutoModel.from_pretrained(model_name)
        hidden=self.encoder.config.hidden_size;self.head=nn.Sequential(nn.Linear(hidden*3+1,hidden),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden,len(self.labels)))
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu");self.encoder.to(self.device);self.head.to(self.device);torch.manual_seed(seed)

    def _encode_rows(self,rows):
        enc=self.tokenizer([r["tokens"] for r in rows],is_split_into_words=True,padding=True,truncation=True,max_length=self.max_len,return_tensors="pt")
        maps=[]
        for bi,row in enumerate(rows):
            ids=enc.word_ids(batch_index=bi);maps.append([next((i for i,x in enumerate(ids) if x==w),None) for w in range(len(row["tokens"]))])
        return enc,maps

    def _score(self,rows,spans):
        import torch
        enc,maps=self._encode_rows(rows);hidden=self.encoder(**{k:v.to(self.device) for k,v in enc.items()}).last_hidden_state;features=[]
        for bi,(s,e) in enumerate(spans):
            positions=[maps[bi][w] for w in range(s,e) if maps[bi][w] is not None] or [0]
            pos=torch.tensor(positions,device=self.device);features.append(torch.cat([hidden[bi,pos[0]],hidden[bi,pos[-1]],hidden[bi,pos].mean(0),torch.tensor([(e-s)/self.max_span],device=self.device)]))
        return self.head(torch.stack(features))

    def _examples(self,records,training=False):return [(r,s,e,l) for r in records for s,e,l in _candidates(r,self.max_span,training,self.seed)]

    def fit(self,records,dev=None,epochs=30,lr=3e-5,batch_size=8,patience=4):
        import torch
        params=list(self.encoder.parameters())+list(self.head.parameters());opt=torch.optim.AdamW(params,lr=lr);lossfn=torch.nn.CrossEntropyLoss();train=self._examples(records,True)
        best,best_state,stale=float("inf"),None,0;rng=random.Random(self.seed)
        for _ in range(epochs):
            rng.shuffle(train);self.encoder.train();self.head.train()
            for i in range(0,len(train),batch_size):
                c=train[i:i+batch_size];opt.zero_grad();loss=lossfn(self._score([x[0] for x in c],[(x[1],x[2]) for x in c]),torch.tensor([self.label_to_id[x[3]] for x in c],device=self.device));loss.backward();opt.step()
            check=self._examples(dev or records,True);self.encoder.eval();self.head.eval();vals=[]
            with torch.no_grad():
                for i in range(0,len(check),batch_size):
                    c=check[i:i+batch_size];vals.append(float(lossfn(self._score([x[0] for x in c],[(x[1],x[2]) for x in c]),torch.tensor([self.label_to_id[x[3]] for x in c],device=self.device))))
            current=sum(vals)/max(1,len(vals))
            if current<best-1e-5:best,best_state,stale=current,(deepcopy(self.encoder.state_dict()),deepcopy(self.head.state_dict())),0
            else:
                stale+=1
                if stale>=patience:break
        if best_state:self.encoder.load_state_dict(best_state[0]);self.head.load_state_dict(best_state[1])
        return self

    def predict(self,records,batch_size=16):
        output=[];self.encoder.eval();self.head.eval()
        import torch
        with torch.no_grad():
            for row in records:
                cand=_candidates(row,self.max_span);entities=[]
                for i in range(0,len(cand),batch_size):
                    c=cand[i:i+batch_size];p=self._score([row]*len(c),[(x[0],x[1]) for x in c]).softmax(-1);labs=p.argmax(-1).tolist()
                    for (s,e,_),lid,conf in zip(c,labs,p.max(-1).values.tolist()):
                        if lid and conf>=.5:entities.append({"type":self.labels[lid],"start":s,"end":e})
                output.append(entities)
        return output
