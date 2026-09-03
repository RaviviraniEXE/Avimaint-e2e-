# AviMaint-DSS — two-environment setup (SpERT + dashboard) — EASY STEPS

Two SEPARATE conda envs so they never fight over `transformers`:

| Env | What runs in it | transformers | Created by |
|---|---|---|---|
| **`spert`** (you already have it) | SpERT model service on :8765 | 4.36.2 | already exists |
| **`avimaint-dash`** (new, created for you) | dashboard **+ reranker** | 5.x | `setup_dashboard_conda.bat` |

They only talk over HTTP (port 8765), so the two `transformers` versions never
touch each other. This is what fixes the `all_tied_weights_keys` error.

Run everything from an **Anaconda Prompt (miniconda3)** so the `conda` command
works. First: `cd /d "D:\information extraction\maintenance-ie\avimaint_dss"`

---

## ONE-TIME: create the dashboard env (only run once)

```
setup_dashboard_conda.bat
```

Creates a fresh env **`avimaint-dash`** (python 3.11) and installs everything the
dashboard and reranker need from `requirements.txt` (streamlit, option-menu,
sentence-transformers, pyvis, plotly, bm25 …). Your `spert` env is not touched.

You only do this once. After that, just use the two run steps below.

---

## EVERY TIME: run it (two Anaconda Prompt windows)

**Window 1 — SpERT service** (uses your existing `spert` env):
```
run_spert_conda.bat
```
Wait until it prints `SpERT query service ready at http://127.0.0.1:8765`.
Leave this window open.

> The bat is already set to `SPERT_ENV=spert` (your env from `conda env list`).
> If you ever rename the env, edit that one line at the top of `run_spert_conda.bat`.

**Window 2 — dashboard + reranker** (uses the new `avimaint-dash` env):
```
run_dashboard_conda.bat
```
Opens at http://localhost:8501. The top status chips flip to **SpERT: ON** and
**Reranker: ON** automatically once each is ready.

That's it. Two windows, two envs, no conflict.

---

## Order of operations (quick reference)
1. (once) `setup_dashboard_conda.bat`
2. Window 1: `run_spert_conda.bat`  → wait for "ready at …:8765"
3. Window 2: `run_dashboard_conda.bat`

## Notes
- **Don't have/need SpERT running?** Skip Window 1. The dashboard still works
  fully on the deterministic lexicon; the SpERT chip just stays OFF.
- **Reranker** lives in `avimaint-dash` only — it needs transformers 5.x, which
  is exactly why it must NOT share the `spert` env.
- **First reranker run** downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` once
  (needs internet that one time); after that it's cached offline.
- One-env alternative (no reranker) is in `ONE_ENV.md`.

