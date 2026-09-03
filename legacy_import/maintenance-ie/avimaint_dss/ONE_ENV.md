# Run everything in ONE conda env (your SpERT env) — simplest, no conflict

The dashboard does NOT use transformers; only the optional reranker does. So you
can run the whole thing in your existing SpERT env (transformers 4.36.2).

From an Anaconda Prompt:
```
conda activate <your-spert-env>        # e.g. maintie   (see: conda env list)
cd /d "D:\information extraction\maintenance-ie\avimaint_dss"
python -m pip install -r requirements-core.txt
```
That adds only the dashboard packages (streamlit, navbar, plotly, pyvis, bm25 …).
It will NOT change transformers, so SpERT keeps working.

Then, in this SAME env, two windows:
```
# window 1 - SpERT service
python services\spert_query_service.py --project-root "D:\information extraction\maintenance-ie"

# window 2 - dashboard
python -m streamlit run app.py
```
In config.yaml set `retrieval.reranker_model: ""` (reranker off) so nothing tries
to load sentence-transformers.

## If you DO want the reranker in this same env
sentence-transformers is the only thing that can pull transformers 5.x. Try a
pinned version that keeps transformers 4.36.2:
```
python -m pip install "sentence-transformers==2.7.0" --no-deps
python -m pip check
```
If `pip check` is clean and the sidebar shows Reranker on, great. If it errors or
wants to upgrade transformers, skip it — the reranker is optional and did not
clearly improve results in evaluation. (Or keep the reranker in the separate
avimaint-dash env from CONDA_SETUP.md.)

