# Pipeline Map — AviMaint-DSS-IE

The whole system, end to end. Everything below the dashed line is driven by
`config/schema.yaml` (the schema + annotation plan), which is why the pipeline is
**dynamic**: change the schema and every stage adapts.

```mermaid
flowchart TD
    subgraph CFG["config/schema.yaml  —  single source of truth (dynamic)"]
      SC["entities · relations (head→tail) · annotation plan · seed"]
    end

    subgraph DATA["1 · Data preparation"]
      RAW["data/raw/*.csv<br/>raw + normalized work orders"]
      COR["src/data/corpus.py<br/>normalize · dedup · unique pool"]
      RAW --> COR
    end

    subgraph ANNO["2 · Corpus building (bootstrap + active learning)"]
      P1["01_make_pilot<br/>random pilot, pre-labeled"]
      LS["Label Studio<br/>human correction"]
      AUD["audit_gold.py<br/>QA: structural + semantic"]
      IMP["02_import_gold<br/>LS export → gold/*.jsonl"]
      AR1["04_active_round --mode random<br/>next random batch (pre-labeled)"]
      FRZ["03_freeze_test<br/>freeze test 225 / dev 100 @ 800"]
      AR2["04_active_round --mode active<br/>mine RARE + uncertain (train only)"]
      GOLD["outputs/gold/*.jsonl<br/>+ outputs/splits.json (frozen)"]

      P1 --> LS --> AUD --> IMP --> GOLD
      GOLD --> AR1 --> LS
      GOLD --> FRZ --> AR2 --> LS
    end

    subgraph MODELS["3 · Three-tier IE models"]
      EMB["08_make_embeddings<br/>domain FastText (unsupervised)"]
      T1["Tier 1 · CRF + LogReg<br/>src/models/crf_ner, relation_logreg"]
      T2["Tier 2 · BiLSTM-CRF + neural RE<br/>char-CNN + pretrained + early stop"]
      T3["Tier 3 · Transformer (BERT)<br/>token-class NER + span-pooling RE"]
      T3b["Tier 3b · SpERT (external)<br/>06_export_spert"]
      GOLD --> T1
      GOLD --> T2
      EMB --> T2
      GOLD --> T3
      GOLD --> T3b
    end

    subgraph EVAL["4 · Evaluation & reporting (generated on run)"]
      TE["05_train_eval<br/>P/R/F1 on FROZEN test"]
      RP["09_report<br/>tables · confusion · comparison · learning curve"]
      OUT["outputs/reports/figures + tables"]
      T1 --> TE
      T2 --> TE
      T3 --> TE
      TE --> RP --> OUT
    end

    subgraph USE["5 · Downstream"]
      DSS["Planning / decision-support dashboard<br/>(solution recommender)"]
      OUT --> DSS
    end

    SC -. drives .-> ANNO
    SC -. drives .-> MODELS
    SC -. drives .-> EVAL
    COR --> P1
```

## How to read it

- **Top:** `config/schema.yaml` is the control panel. It defines the label set,
  the relation constraints, the batch sizes, and the seed. The dashed arrows show
  that it drives annotation, modelling, and evaluation alike.
- **Left-to-right within a stage** is the data flow; **the loop** in stage 2
  (`gold → active_round → Label Studio → import → gold`) is the human-in-the-loop
  bootstrap that grows the corpus.
- **The freeze** (`03_freeze_test`) is the honesty guarantee: once the test/dev
  sets are frozen at 800 records, active learning only ever grows the *training*
  pool, so reported F1 is never inflated.
- **Stage 4 artifacts are generated on run**, not shipped — run `09_report` and the
  figures/tables appear under `outputs/reports/`.

Open `pipeline_map.html` in a browser to see this rendered.

