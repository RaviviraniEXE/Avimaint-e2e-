# AviMaint-DSS-IE — Information Extraction for Maintenance Work Orders

Master's thesis pipeline: **structured information extraction and planning support
from aircraft maintenance work orders.** Given short, noisy free-text logbook
entries, the system extracts *what part*, *what fault*, *what action*, *what
outcome* and the *relations* between them — the structured basis for a maintenance
decision-support dashboard.

The pipeline is **schema-driven and dynamic**: a single file, `config/schema.yaml`,
defines the entity types, relation types (with head/tail constraints) and the
annotation plan. Every script — Label Studio config, BIO tag set, candidate
relation pairs, active-learning targets, evaluation — reads that schema, so you can
retarget the whole system (e.g. to the full **MaintIE** schema) by editing one file,
with no code changes.

---

## 1. What the system does (in one picture)

```
raw work orders ──► normalize ──► de-duplicate ──► ANNOTATE (bootstrap + active learning)
                                                        │
                                                        ▼
                                          gold corpus (entities + relations)
                                                        │
                    ┌───────────────────────────────────┼────────────────────────────┐
                    ▼                                   ▼                              ▼
       Tier 1: CRF + LogReg   Tier 2: BiLSTM-CRF + neural RE   Tier 3: Transformer (BERT)   [+ SpERT]
                    └───────────────────────────────────┼────────────────────────────┘
                                                        ▼
                                    evaluate on FROZEN test  ──►  figures + tables
                                                        ▼
                                    structured output ──► planning / decision dashboard
```

A full, detailed pipeline map is in **[PIPELINE.md](PIPELINE.md)** (and
`pipeline_map.html` renders it in a browser).

---

## 2. The three components of the thesis

1. **Normalization** — cleans raw logbook text (abbreviations, spelling) before IE.
   Compared as Raw / Rules / ByT5 / Hybrid (separate module).
2. **Information Extraction** (this repo's core) — a **three-tier NER+RE benchmark**
   plus the annotation-corpus-building workflow. The tiers are, in increasing
   modernity: **Tier 1** classical CRF + logistic-regression RE (strong small-data
   baseline); **Tier 2** BiLSTM-CRF with char-CNN + domain FastText + early stopping;
   **Tier 3** a **transformer encoder** — BERT-family token-classification NER + a
   SpERT/PURE-style span-pooling relation head, the current standard for IE. The
   encoder is set in `config/schema.yaml` (`models.transformer.encoder`), so the
   same tier runs `distilbert` on a small GPU or a larger/domain model — and doubles
   as the model for a later **MaintIE benchmark** by swapping encoder + schema.
   SpERT can be added as an external Tier-3b via `06_export_spert.py`.
3. **Planning support / dashboard** — consumes the extracted entities/relations
   (the solution recommender). Built on the IE output.

---

## 3. The schema (`config/schema.yaml`) — the single source of truth

**9 entity types:** `MAINT_ITEM`, `ACTION`, `FAULT`, `ABN_PROC`, `OUTCOME`, `LOC`,
`TECH_OBS`, `OP_CTX`, `REFERENCE`.

**11 relation types** (each with allowed head→tail entity types), e.g.
`ACTION_ON_ITEM`, `ISSUE_ON_ITEM`, `HAS_LOCATION`, `ACTION_ADDRESSES_ISSUE`
(repairs), `ACTION_INVESTIGATES_ISSUE` (diagnostics), `ACTION_RESULTS_IN_OUTCOME`,
`ACTION_FOLLOWS_REFERENCE`.

**Annotation plan** (sizes are editable): a **random** pilot + correction rounds
build an unbiased corpus; a random **test/dev split is frozen at 800 records** and
never touched again; **active-learning** rounds then enrich the rare classes into
the *training* pool only, so the test F1 stays honest.

> **Dynamic / MaintIE note.** To switch to another schema (e.g. the full MaintIE
> hierarchy), edit only `config/schema.yaml` — add/rename entities, relations and
> their head/tail constraints. `src/schema.py` re-derives the BIO tag set, the
> candidate relation pairs, and the Label Studio labelling config automatically.

---

## 4. Repository layout

```
config/schema.yaml         ← the schema + annotation plan (edit this to retarget)
data/raw/                  ← raw + normalized corpus CSVs
outputs/
  gold/                    ← 1,600 audited records (800 random + 800 rare-enriched) ← keep!
  splits.json              ← FROZEN test/dev assignment                          ← keep!
  labeling_config.xml      ← paste into Label Studio
src/
  schema.py                ← loads schema; derives BIO tags, candidate relations
  features.py              ← CRF + LogReg feature functions
  evaluate.py              ← entity/relation P/R/F1 (micro/macro/per-class)
  reporting.py             ← tables, confusion matrices, comparison figures
  data/                    ← corpus, dedup, sampling, active learning, Label Studio I/O
  models/                  ← crf_ner, bilstm_crf, relation_logreg, relation_bilstm,
                             embeddings (FastText), spert_export
scripts/00..09             ← the runnable pipeline (see §5)
tests/                     ← pipeline smoke tests (pytest)
```

The scripts are numbered in run order. `scripts/audit_gold.py` is a QA tool you run
on any corrected batch before importing it.

---

## 5. The pipeline, step by step

Run everything from the project root, inside your virtualenv.

```bash
pip install -r requirements.txt      # core + torch + pytorch-crf + gensim
```

| # | Command | What it does | Output |
|---|---------|--------------|--------|
| 1 | `python scripts/01_make_pilot.py` | Draw a random pilot from the pool, pre-label it | `outputs/rounds/pilot_tasks.json` |
| — | *(correct in Label Studio, then export)* | Human correction | LS export JSON |
| Q | `python scripts/audit_gold.py <export>.json` | QA: structural + semantic checks on a batch | console report |
| 2 | `python scripts/02_import_gold.py --export <export>.json --name pilot` | Convert LS export → gold | `outputs/gold/pilot.jsonl` |
| 0 | `python scripts/00_gold_status.py` | Show corpus size + per-class support | console |
| 4 | `python scripts/04_active_round.py --name round1 --n 500 --mode random` | Train on gold, pre-label the next **random** batch | `outputs/rounds/round1_tasks.json` |
| 3 | `python scripts/03_freeze_test.py` | **At 800 gold:** freeze random test(225)/dev(100) | `outputs/splits.json` |
| 4 | `python scripts/04_active_round.py --name rare1 --n 300 --mode active` | **After freeze:** mine rare + uncertain records | `outputs/rounds/rare1_tasks.json` |
| 8 | `python scripts/08_make_embeddings.py` | Train domain FastText on the full corpus (once) | `outputs/embeddings/domain_ft.model` |
| 5 | `python scripts/05_train_eval.py --tiers 1 2 3` | Train tiers (3 = transformer), evaluate on frozen test | `outputs/reports/ie_results*.{json,csv}` |
| 9 | `python scripts/09_report.py --tiers 1 2 3 --run-id gold800` | **Generate all tables + figures** (all tiers) | `outputs/reports/figures/`, `tables/` |
| 6 | `python scripts/06_export_spert.py` | Export gold to SpERT (external Tier-3b) JSON | SpERT data files |
| 7 | `python scripts/07_normalization_ie.py` | Normalization × IE study | report |

The active-learning loop repeats: `04_active_round (active)` → correct in Label
Studio → `02_import_gold` → `05_train_eval` / `09_report`. Every training step
**re-reads all of `outputs/gold/*.jsonl`**, so the models automatically use the
full, growing corpus — nothing to edit between rounds.

**Your current state:** the corpus is already at **800 labeled records**
(`outputs/gold/`), the test/dev split is **already frozen** (`outputs/splits.json`),
and Tier-1/Tier-2 have been evaluated. You do **not** need to re-label. Continue at
step 4 (`--mode active`) to grow the rare classes, or run step 9 to regenerate all
result artifacts.

---

## 6. Result figures & tables (generated on run — not shipped stale)

`scripts/09_report.py` writes, to `outputs/reports/`:

- **Figures** (`figures/*.png`): overall model comparison and a **learning curve**
  (F1 vs corpus size) — both drawn with **bootstrap 95 % confidence-interval error
  bars** — plus per-class entity & relation F1, entity confusion matrices, and
  class-support distributions.
- **Tables** (`tables/*.csv`): per-class precision/recall/F1/**support** per model,
  `overall_metrics.csv` with **CI columns**, full-corpus support, and
  `validity_diagnostics.csv` (see below).
- `metrics_full.json`: every number, including CIs, in one file.

**Versioned model saving.** Add `--save-models` and every trained tier is persisted
under `outputs/models/<run-id>__<timestamp>/` — a **new versioned folder each run**,
so retraining after a change never overwrites an earlier model. Each version has a
`model_card.json` (run-id, timestamp, tuned?, encoder, train size, tiers, F1 + CI,
diagnostics); `outputs/models/registry.csv` lists every version for comparison and
`latest.txt` points to the newest. Use it on your final runs (Tier-3 transformers
are ~250 MB each, so leave it off while iterating).

**Confidence intervals & validity (built in).** Every micro-F1 is reported with a
bootstrap 95 % CI (resample the test docs `--bootstrap N` times, default 1000), so
you never over-claim a difference smaller than the noise. `validity_diagnostics.csv`
additionally reports, each run, whether the model is *really learning*: the
**memorization baseline** (per-word majority tag — the floor), the model's
**context gain** over it, the **train/test generalization gap**, and the
**test-set OOV rate**. Together these show the results are genuine generalization,
not memorization or leakage.

Re-run `09_report.py` after each annotation round (use a distinct `--run-id`) to
extend the learning curve. When SpERT is trained, add
`--spert outputs/reports/spert_test.json` and Tier 3 joins every figure and table.

---

## 7. Model tuning — what we checked

Every tier can **choose its own best hyperparameters on the dev set** — run any
training/report script with `--tune` and each model grid-searches the grids in
`config/schema.yaml → models.tuning` (fully editable), selecting the config with
the best dev F1 before scoring the frozen test:

- **CRF** → `c1`,`c2` (L1/L2)   • **LogReg RE** → `C`
- **BiLSTM** → `lr`,`dropout`   • **Transformer** → `lr`

Without `--tune`, sensible defaults run fast for iteration; with `--tune` you get
the honest "best-selected" numbers for the thesis. Historical pilot observations
from the first 800 random records are retained only as baseline context and must
be re-tested on the final frozen 1,600-record corpus:

Each tier was assessed against best practice, not just left at defaults:

- **Tier 1 CRF** — L-BFGS with `all_possible_transitions`, orthographic + gazetteer
  features in a ±1 window. We tested a ±2 window, extra orthographic flags, and
  dev-grid-tuning of `c1/c2`; this over-fit the historical pilot. The final
  launcher reselects hyperparameters on the 100-record development partition
  before evaluating the 225-record test partition.
- **Tier 1 RE (LogReg)** — `class_weight="balanced"`; features enriched with the
  tokens before/after each span and the first/last token between them
  (**relation F1 0.708 → 0.749** end-to-end). Kept.
- **Tier 2 BiLSTM / neural RE** — char-CNN + domain FastText + dev early stopping,
  now with **gradient clipping** (norm 5.0) for LSTM stability (Reimers & Gurevych 2017).
- **Tier 3 transformer** — AdamW + gradient clipping (norm 1.0), dev early stopping.

## 8. Reproducibility notes

- All randomness is seeded (`annotation.seed` in the schema).
- The frozen split (`outputs/splits.json`) is the honest test protocol — active
  learning never adds to it.
- Neural Tier-2 numbers can vary by a few thousandths between runs (training noise);
  treat the `09_report.py` run as canonical since it saves all artifacts from one
  training pass.
- Reference dataset/schema lineage: MaintNet (Akhbardeh 2020), MaintNorm & MaintIE
  (Bikaun 2024), ByT5 (Xue 2022).
