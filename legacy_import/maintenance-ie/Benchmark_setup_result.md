# Benchmark — setup & results snapshot

Persistent record so any future session can pick up. Full guide lives in the repo as
`PROJECT_GUIDE.md` and `REPRODUCE.md`.

## Data (never lose)
- `outputs/gold/*.jsonl` — 1,600 hand-corrected gold records (pilot 300, round1 500, rare1/2/3 = 300/300/200)
- `outputs/splits.json` — frozen test(225)/dev(100) split; never change
- `config/schema.yaml` — 9 entities, 11 relations (incl. ACTION_INVESTIGATES_ISSUE vs ACTION_ADDRESSES_ISSUE)

## Environments (Miniconda, 3 isolated)
- `maintie` — Tiers 1-3 + scoring (numpy==2.2.6, transformers==4.36.2, torch cu121)
- `spert` — official SpERT only (transformers==4.36.2)
- `rebel` — REBEL only (later)
Each has a `requirements-*.txt` in the repo.

## Comparison hub
`python scripts/09_report.py --tiers 1 2 3 --spert <preds> --external REBEL=<preds> --run-id <id>`
→ `outputs/reports/tables/overall_metrics.csv` (master table), figures, metrics_full.json.
External models train in their own env, export predictions, scored by src/evaluate.py on the
same frozen test set.

## Current results (gold 1600, test=225)
| Model | Entity F1 | Relation F1 |
|---|---|---|
| Tier1 CRF | 0.934 | 0.780 |
| Tier2 BiLSTM | 0.943 | 0.685 |
| Tier3 DistilBERT | 0.941 | 0.829 |
| SpERT BERT | 0.943 | 0.841 |
Significance: relations SpERT>CRF>BiLSTM (all sig); entities tie. Entities saturated;
relations discriminate. DAPT gave no significant micro-F1 gain.

## Weak classes (next: error analysis)
Relations: OCCURS_UNDER_CONTEXT ~0.41, ACTION_INVESTIGATES_ISSUE ~0.52 (vs ADDRESSES 0.92).
Entity: OP_CTX ~0.63. These low-support/subtle classes drag macro-F1.

## Key code changes made (this session)
- transformer_ie.py: `return_token_type_ids=False` (runs on transformers 4.36)
- _bootstrap.py: warning suppression + banner/step helpers
- 09_report.py: stage banners, final results table, generic `--external NAME=path` flag
- 08b_domain_pretrain.py (DAPT), 10_significance.py (paired bootstrap + seeds), evaluate.py (paired_bootstrap)

## Next steps
1. Confusion-matrix error analysis on weak relation classes
2. MaintIE benchmark (run this pipeline on MaintIE data, same schema)
3. REBEL seq2seq baseline (optional)
