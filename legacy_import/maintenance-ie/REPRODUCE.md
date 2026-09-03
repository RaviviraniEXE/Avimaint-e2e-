# Reproducing the benchmark

All models are evaluated on ONE frozen test set (225 records, `outputs/splits.json`)
and scored by ONE scorer (`src/evaluate.py`). Your Tiers 1-3 train inside the main
environment; each external published model (SpERT, REBEL) trains in its OWN isolated
environment and exports predictions, which the main pipeline imports and scores. This
keeps every number comparable while letting each external repo keep its own dependencies.

```
              TRAIN (per-env)                    SCORE (one place)
  maintie  ── Tier 1 CRF ───────────┐
           ── Tier 2 BiLSTM ────────┤
           ── Tier 3 Transformer ───┤──►  09_report.py  ──►  outputs/reports/
  spert    ── SpERT ── preds.json ──┤        (src/evaluate.py)      tables/overall_metrics.csv
  rebel    ── REBEL ── preds.json ──┘                              figures/  + metrics_full.json
```

---

## Environments (create once with Miniconda)

### 1. `maintie` — your pipeline (Tiers 1-3) + all scoring/reporting
```
conda create -n maintie python=3.11 -y
conda activate maintie
cd "D:\information extraction\maintenance-ie"
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy transformers scikit-learn gensim sklearn-crfsuite pandas matplotlib pyyaml tqdm spacy tensorboardX
pip freeze > env-maintie.lock.txt
```

### 2. `spert` — official SpERT only
```
conda create -n spert python=3.11 -y
conda activate spert
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy==2.2.6 transformers==4.36.2 scikit-learn tqdm jinja2 tensorboardX spacy
pip freeze > env-spert.lock.txt
```

### 3. `rebel` — REBEL only (add when needed)
```
conda create -n rebel python=3.11 -y
conda activate rebel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pytorch-lightning omegaconf
pip freeze > env-rebel.lock.txt
```

Keep the three `env-*.lock.txt` files in the repo — they go in the thesis appendix.

---

## Run order

### A. Your pipeline (env: `maintie`)
```
conda activate maintie
cd "D:\information extraction\maintenance-ie"
# ensure config/schema.yaml -> models.transformer.encoder: distilbert-base-uncased
python scripts/09_report.py --tiers 1 2 3 --tune --save-models --bootstrap 1000 --run-id gold1600
python scripts/10_significance.py --tiers 1 2 3 --seeds 3
```

### B. SpERT (env: `spert`) — train once, export predictions
```
conda activate maintie
python scripts/06_export_spert.py                 # writes outputs/spert/{train,dev,test}.json + types + conf
conda activate spert
python spert/spert.py train --config spert/configs/avimaint_spert.conf
conda activate maintie
python scripts/06b_import_spert_preds.py "outputs/spert/log/avimaint_spert/<timestamp>/predictions_test_epoch_20.json"
# -> writes outputs/reports/spert_test.json
```

### C. REBEL (env: `rebel`) — later, same pattern
Train REBEL in its env, convert its output to the pipeline format
(`[{tokens, entities, relations}]`), save as `outputs/reports/rebel_test.json`.

### D. The single comparison (env: `maintie`)
```
conda activate maintie
python scripts/09_report.py --tiers 1 2 3 --bootstrap 1000 \
       --spert outputs/reports/spert_test.json \
       --external REBEL=outputs/reports/rebel_test.json \
       --run-id gold1600_full
```

This scores Tiers 1-3 (retrained) plus SpERT and REBEL (imported) on the same test set
and writes the combined comparison to:
```
outputs/reports/tables/overall_metrics.csv     <- every model, one table
outputs/reports/figures/                        <- comparison charts with CIs
outputs/reports/metrics_full.json               <- full per-class breakdown
```

Drop `--external ...` until REBEL exists; drop `--spert ...` too and you get just Tiers 1-3.

---

## Where results live (one place)

`outputs/reports/tables/overall_metrics.csv` is the master comparison — one row per model
(Tier1_CRF_LogReg, Tier2_BiLSTM_Neural, Tier3_Transformer, Tier3b_SpERT, REBEL), with
entity/relation micro-F1 and 95% CIs. Everything else (figures, per-class tables,
significance CSVs) hangs off the same run. That CSV is what goes in the thesis results table.

