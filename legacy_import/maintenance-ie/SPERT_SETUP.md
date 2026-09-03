# Tier 3b — running the official SpERT on your dataset (for the MaintIE benchmark)

SpERT is a published joint entity+relation model (Eberts & Ulges, 2020). For a
benchmark study you run the **authors' official code** — not a reimplementation —
so results are faithful and comparable to MaintIE (which also used SpERT). Your
pipeline already exports the data and reads SpERT's predictions back; these are the
exact steps.

> Use a **separate virtual environment** for SpERT. Its dependencies (older
> `transformers`/`torch`) will clash with this project's newer ones — keep them apart.

---

## 1. Export your data + config (from THIS project)

```bash
python scripts/06_export_spert.py
```
Writes, using your frozen split:
```
outputs/spert/train.json          outputs/spert/dev.json
outputs/spert/test.json           outputs/spert/avimaint_types.json
outputs/spert/avimaint_spert.conf   <- ready-to-edit SpERT config
```

## 2. Clone SpERT and make its environment

```bash
git clone https://github.com/lavis-nlp/spert.git
cd spert
python -m venv .spert-venv
# Windows: .spert-venv\Scripts\activate   |   Linux/Mac: source .spert-venv/bin/activate
pip install -r requirements.txt
```

## 3. Point the config at your exported data

Open the generated `avimaint_spert.conf` and set **absolute paths** (safest across
folders), e.g. on Windows:
```
train_path = D:/information extraction/maintenance-ie/outputs/spert/train.json
valid_path = D:/information extraction/maintenance-ie/outputs/spert/test.json
types_path = D:/information extraction/maintenance-ie/outputs/spert/avimaint_types.json
predictions_path = D:/information extraction/maintenance-ie/outputs/spert/predictions_test.json
model_path = bert-base-cased
tokenizer_path = bert-base-cased
```
Notes:
- `valid_path = test.json` so `store_predictions` writes predictions for your frozen
  test set (that's what the benchmark scores).
- `model_path = bert-base-cased` is SpERT's standard backbone. (The generator fills in
  your Tier-3 encoder; for SpERT specifically, `bert-base-cased` is the safe, native
  choice — SpERT's code is BERT-oriented.)
- 4 GB GPU: keep `train_batch_size = 2`, `max_span_size = 10`; if you hit out-of-memory
  lower `max_pairs` (e.g. 500) or `neg_relation_count`.

## 4. Train + predict (in the SpERT repo, its venv)

```bash
python ./spert.py train --config avimaint_spert.conf
```
With `final_eval = true` and `store_predictions = true`, SpERT trains, evaluates on
the test set, and writes `predictions_test.json`.

## 5. Bring predictions back into your pipeline

Back in THIS project (its own venv):
```bash
python scripts/06b_import_spert_preds.py outputs/spert/predictions_test.json
python scripts/09_report.py --tiers 1 2 3 --spert outputs/reports/spert_test.json --run-id gold1400_spert
```
SpERT now appears as **Tier3_SpERT** in every table and figure — per-class F1,
confusion, the overall comparison (with CIs), and the model registry — right
alongside Tiers 1–3.

---

## What's what (so it's clear)

- **Tier 3 · Transformer** — *your* in-pipeline model (HuggingFace DistilBERT + task
  heads). No external setup; runs with `--tiers 1 2 3`.
- **Tier 3b · SpERT** — the *official external* model, run once via the steps above
  for benchmark parity with MaintIE. Optional but recommended for the benchmark study.

Both use your exact frozen test split, so all tiers are directly comparable.
