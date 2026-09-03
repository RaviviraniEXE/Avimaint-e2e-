# Maintenance Work-Order Normalization

Normalization component of the master's thesis *"Development and Evaluation of an
NLP-Based Concept for Structured Information Extraction and Planning Support in
Maintenance Work Orders."* **See `RUN_STEPS.txt` for the exact commands.**

## Data & resources

- **Logbook**: `data/raw/Aircraft_Annotation_DataFile.csv` — MaintNet aviation
  logbook, 6,169 records (`IDENT, PROBLEM, ACTION`). Text is truncated at 60/55
  characters *in the original RIT source* (not fixable — inherent to the dataset).
- **Expert resources** (`data/raw/expert/`, from Nadine Amin's cleaned release):
  the expert-cleaned dataset (used as **gold**), a 100-entry abbreviation list,
  a 114-entry misspelling list, and 26 keep-as-is abbreviations.

Processing unit = one **combined** text per record (`PROBLEM + ACTION`), matching
Amin's gold. Numbers are kept as **digits** (best for information extraction);
`numbers: words` in the config reproduces Amin's speech-style gold exactly.

## The four systems

| System | Module | Method |
|--------|--------|--------|
| **A — Raw** | `system_a_raw.py` | control: unicode + whitespace + lowercase |
| **B — Rules** | `system_b_rules.py` | Amin abbreviation + misspelling + keep lists, symbols, digits |
| **C — Transformer** | `system_c_byt5.py` | ByT5 char-level seq2seq (trained on silver+gold) |
| **D — Hybrid** | `system_d_hybrid.py` | rules then ByT5 |

## Current results (Systems A + B, held-out test = 926 records)

| system | oov reduction | exact match | word-ERR | char-ERR |
|---|---|---|---|---|
| A · Raw | 0% | 0.04 | 0.00 | 0.00 |
| B · Rule-based | **89.8%** | **0.62** | **0.82** | **0.82** |

System B removes ~82% of the raw error against the expert gold. The remaining
gap is truncation reconstruction (`COWLI`→cowling) and deliberately-skipped
ambiguous abbreviations — exactly what System C is meant to close.

## Design notes

- **Rules built from expert lists, not guesses.** Ambiguous abbreviations (COMP,
  INSP, IN, SEC) are skipped; risky misspelling corrections whose source is a
  real word (OFF→OF) are skipped. Every decision is logged to
  `outputs/reports/dictionary_build_report.txt`.
- **Leak-free.** `split_gold.py` makes disjoint train/dev/test; ByT5 never trains
  on the test rows; extrinsic metrics score on the held-out test split.
- **Alignment preserved** (`alignment_B_rules.jsonl`) so IE annotations can be
  projected raw↔normalized — the basis for the thesis' normalization→IE study.
- **All results saved** automatically (see `RUN_STEPS.txt` §5), including a
  `results_log.csv` that accumulates every run.

## Comparability to prior work

Uses **MaintNet** data (Akhbardeh et al., 2020) + **Amin**'s expert cleaning as
gold, and reports the **Error Reduction Rate** that **MaintNorm** (Bikaun et al.,
2024) headlines. The novel contribution is downstream: feed each system's output
into the IE stage (CRF → BiLSTM-CRF → SpERT) and compare F1.

