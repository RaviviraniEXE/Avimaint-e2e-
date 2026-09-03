# maintie-bench — running the pipeline on the MaintIE corpus

This is a copy of the maintenance-IE pipeline configured for the **MaintIE**
benchmark corpus (Bikaun et al., *MaintIE*, LREC-COLING 2024), so the same four
model families (CRF, BiLSTM-CRF, Transformer, SpERT) run on MaintIE's data in
MaintIE's own schema. It is a separate project from your aviation work — its own
data, its own results, its own frozen split.

## What's already been done (in this package)

- **Data downloaded** from `github.com/nlp-tlp/maintie`:
  `data/raw/gold_release.json` (1,076 gold records) + `data/raw/maintie_scheme.json`.
- **Schema** written for MaintIE's **coarse (level-1)** granularity — 5 entities
  (PhysicalObject, Activity, State, Process, Property) and 6 relations
  (hasPart, contains, hasProperty, hasAgent, hasPatient, isA) — in `config/schema.yaml`.
- **Converter** `scripts/00_convert_maintie.py` already run → produced:
  - `outputs/gold/maintie.jsonl` — 1,076 records in the pipeline gold format
  - `outputs/splits.json` — MaintIE's **official** 80/10/10 file-order split
    (**860 train / 108 dev / 108 test** — the paper's 108-text test set)
- **Verified**: Tier 1 runs end-to-end (entity F1 ≈ 0.76, relation F1 ≈ 0.48 at coarse level).

So you can go straight to running the report.

## Run it (env: maintie, from this folder)

```
conda activate maintie
cd "D:\information extraction\maintie-bench"
python scripts/09_report.py --tiers 1 2 3 --tune --save-models --bootstrap 1000 --run-id maintie_coarse
python scripts/10_significance.py --tiers 1 2 3 --seeds 3
```

For SpERT on MaintIE (same flat gold, so all models are comparable):
```
python scripts/06_export_spert.py            # exports THIS project's MaintIE gold
# train SpERT in the spert env with configs pointing at maintie-bench/outputs/spert/
python scripts/06b_import_spert_preds.py <predictions.json>
python scripts/09_report.py --tiers 1 2 3 --spert outputs/reports/spert_test.json --run-id maintie_full
```

Results land in `maintie-bench/outputs/reports/` — completely separate from aviation.

## Important notes for the thesis

**Nested entities.** ~30% of MaintIE records have overlapping/nested entity spans.
BIO tagging (used by the CRF / BiLSTM / transformer-NER tiers) cannot represent
overlaps, so those tiers structurally cannot predict nested entities — their recall
is capped on those cases. Gold is kept **full** (all entities + relations), so SpERT
(span-based, handles overlaps) is scored against the same gold and can capture them.
This is a genuine, reportable finding: *span-based joint models handle MaintIE's
nesting; sequence-labeling models cannot.*

**Granularity.** This package uses MaintIE's **coarse (5-class)** level, matching
their "Entity Classes: 5" published results (SpERT entity ≈ 87.4, relation ≈ 67).
To also run the finer levels, regenerate with level 2 or 3 — the converter's
`coarse_entity()` truncates to level 1; change it to keep more `/`-segments and
update `config/schema.yaml` accordingly. (Ask if you want the 32-class variant.)

**Comparison is same-schema, per-dataset.** Compare model *rankings* and reproduce
MaintIE's published numbers here; compare aviation and MaintIE qualitatively, not by
raw F1 (different schemas/difficulty — not directly comparable).

## Regenerating from scratch (if ever needed)

```
python scripts/00_convert_maintie.py     # data/raw/gold_release.json -> gold + split
```
The split is deterministic (MaintIE's official file order), so it always reproduces
the same 860/108/108.

