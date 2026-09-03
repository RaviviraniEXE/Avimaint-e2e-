# AviMaint-DSS — final dashboard

A clean, explainable maintenance decision-support dashboard for the MaintNet
aviation corpus. Four pages: **Overview · Diagnose · Insights · Planning**.
Deterministic (no LLM). Runs on the raw CSV out of the box; uses your trained
SpERT model for richer structure when its service is running.

## 0. One-click (recommended)

Double-click **`START_AVIMAINT.bat`**. It installs everything (including
torch/transformers for SpERT), starts the SpERT model service in its own window,
waits for it to load, then opens the dashboard. The sidebar flips to
**SpERT: ON** automatically once the model is ready. That's the whole setup.

The manual steps below are only if you want to run pieces separately.

## 1. Install

```bat
cd /d "D:\information extraction\maintenance-ie\avimaint_dss"
pip install -r requirements.txt
```

## 2. Run the dashboard (works immediately)

**Easiest:** double-click **`run_dashboard.bat`**.

**Or from a terminal — note `python -m`** (this avoids the "'streamlit' is not
recognized" PATH error you saw; `streamlit` alone only works if its Scripts
folder is on PATH, `python -m streamlit` always works):

```bat
python -m streamlit run app.py
```

Opens at http://localhost:8501. It loads the bundled
`data/Aircraft_Annotation_DataFile.csv` (6,169 work orders), builds structure
with the deterministic lexicon, and every page works. First launch caches the
processed corpus to `data/cache/` so later launches are instant.

## 3. Turn on your trained SpERT (richer structure)

The dashboard reads structure from a small local SpERT service (bundled in
`services/`, reusing your proven runtime). It needs your ML deps in the same
Python you're using:

```bat
pip install torch transformers safetensors
```

**Start the service** — double-click **`run_spert.bat`**, or run:

```bat
python services\spert_query_service.py --project-root "D:\information extraction\maintenance-ie"
```

It auto-finds all three pieces you already have:
- SpERT code at `D:\information extraction\spert\` (the `spert\models.py` package),
- your checkpoint at `outputs\spert\save\avimaint_spert\2026-07-28_11-39-03.560233\final_model`,
- the type file at `outputs\spert\avimaint_types.json`.

Leave that window open (it prints `SpERT query service ready at http://127.0.0.1:8765`).
To just verify it loads without starting the server, add `--check-only`.

**Then point the dashboard at it** — in `config.yaml` set:

```yaml
extraction:
  mode: spert           # was: rule
```

Delete the `data\cache\` folder and reload the dashboard. The sidebar will show
**"Structure: SpERT (live)"** and MAINT_ITEM / FAULT / ABN_PROC entities come
straight from your trained model. If the service isn't running it falls back to
the lexicon and says so — it never breaks.

### Order of operations for the full SpERT setup
1. Terminal A: `run_spert.bat` → wait for "service ready at …:8765".
2. Set `extraction.mode: spert` in `config.yaml`, delete `data\cache\`.
3. Terminal B: `run_dashboard.bat`.

> Alternatively, if you already have SpERT predictions for the corpus, drop them
> in as JSONL (`{"ident": ..., "problem_pred": {"entities": [...], "relations":
> [...]}}`) and set `data.predictions_path` in `config.yaml`. No re-running the
> model.

## 3b. Normalization (uses your thesis lists)

The dashboard normalizes **PROBLEM and ACTION separately** (abbreviations,
misspellings, symbols) before matching — so it works on the *normalized* text
without needing your combined problem+action file. It's ON by default
(`config.yaml → extraction.normalize: true`).

Out of the box it uses a strong built-in aviation map (`L/H`→left, `eng`→engine,
`gskt`→gasket, `R&R`→removed and replaced, …). To use **your exact System-B
lists**, drop these CSVs into the `data/` folder and delete `data/cache/`:

```
data/abbreviations.csv   # header: abbrev,expansion
data/misspellings.csv    # header: wrong,correct
data/keep.csv            # header: word          (optional; words left untouched)
```

They override the built-ins. Raw text is always preserved for the case
provenance shown in Diagnose; only matching/extraction use the normalized text.

## 4. Point at your own data

In `config.yaml`, set `data.csv_path` to any CSV with `IDENT, PROBLEM, ACTION`
columns.

## What each page does (six pages)

- **Overview** — KPI tiles + recurring problems + top components + top actions.
- **Diagnose** — type a problem → **what the model extracted** (entities + relations
  + a per-problem knowledge graph) → structured recommendation with the **confidence
  ladder** (Strong / Moderate / Exploratory) → **recorded strategies** (corroborated
  vs single-case) → supporting cases by IDENT. Widening fallback (problem → component
  → fault) means you get a graded answer, never "insufficient evidence".
- **Insights** — recurring/chronic faults, component Pareto, fault modes, action
  frequency, component×fault heatmap, problem→action, outcome mix. Counts, not rates.
- **Knowledge Graph** — the whole corpus as an interactive component→fault→action
  network (drag to explore; sliders control node count and edge threshold).
- **Planning** — recurring-fault register → grounded **job card** assembled only
  from recorded actions, each step traceable to source work orders.
- **Evaluation** — cluster-safe **leave-one-out**: agreement with recorded practice
  (macro-recall, top-1/top-3, MRR), coverage-at-confidence per tier vs a majority
  baseline, and per-family recall. Honest by design — it measures agreement with
  history, not correctness (no gold exists).
- **Guide** — in-app explanation of how everything works.

### Offline evaluation report (for the thesis)
```
python run_eval.py --full        # full corpus, definitive numbers
python run_eval.py --compare     # raw vs System-D, side by side (normalization delta)
```
Writes `outputs/reports/eval_report.md` + `.json`. The `--compare` run directly
tests your central research question on the recommender task (early result: on
this task normalization gave no measurable gain — consistent with the open
question your normalization chapter cites; run `--full` to confirm).

### Data & models
- Default dataset is your **System-D normalized** `data/dashboard_dataset_D.csv`
  (`extraction.normalize: false`). For raw text, point `data.csv_path` back at
  `Aircraft_Annotation_DataFile.csv` and set `normalize: true`.
- **SpERT** (sidebar shows ON/OFF) supplies real entities + relations when running.
- **Reranker** — set `retrieval.reranker_model` (e.g.
  `cross-encoder/ms-marco-MiniLM-L-6-v2` or a development-selected checkpoint) to add a
  cross-encoder reranking stage. Needs `pip install sentence-transformers`.

## Tuning

- `config.yaml → recommender.top_k` — retrieval depth.
- `config.yaml → insights.recurring_min` — how many work orders make a "chronic"
  fault.
- Confidence thresholds live in `core/recommend.py`
  (`STRONG_SCORE`, `MODERATE_SCORE`, `MIN_SCORE`).
- Optional dense channel: `pip install sentence-transformers`, then set
  `retrieval.dense_model: sentence-transformers/all-mpnet-base-v2`.

## Layout

```
app.py              Streamlit app + 4 pages
config.yaml         paths, mode, thresholds
core/
  normalize.py      canonicalisation, action/fault families, lexicons
  extraction.py     SpERT client + rule fallback -> canonical structure
  corpus.py         load CSV/predictions, derive structure, cluster, cache
  insights.py       frequency analytics (count-based)
  retrieval.py      BM25 + word/char TF-IDF + structured overlap (+ optional dense)
  recommend.py      confidence ladder + widening fallback
  watchlist.py      recurring register + grounded job card
ui/
  theme.py          palette + KPI tiles + badges
  charts.py         Plotly charts (validated palette)
```

This maps 1:1 to the build blueprint (avimaint-dss-blueprint). The optional LLM
layer (Phase 2) is deliberately not built yet — the dashboard is fully
functional and defensible without it.

## SpERT + transformers version conflict (fixed)

SpERT was written for an older `transformers`; the reranker (`sentence-transformers`)
needs a newer one, and downgrading conflicts. Two things resolve this:

1. **Automatic code shim** (already in `services/spert_runtime.py`): patches the
   weight-tying call so the trained SpERT checkpoint loads under the *modern*
   transformers the dashboard uses. Just start SpERT normally (`run_spert.bat`) —
   no downgrade needed.
2. **Isolated env fallback** — if the shim ever fails on your machine, run
   **`setup_spert.bat`**. It builds a separate `.venv_spert` with a SpERT-compatible
   `transformers` (from `requirements-spert.txt`) and runs the service there. The
   dashboard talks to it over HTTP (port 8765), so the two environments never clash.

Either way, keep the dashboard in its own env with the reranker.
