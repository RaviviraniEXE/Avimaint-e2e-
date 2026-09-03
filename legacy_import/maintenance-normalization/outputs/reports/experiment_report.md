# Normalization experiment — system comparison

Number convention: **digits**. Gold: Amin expert-cleaned dataset.
Extrinsic scored on: **held-out test split**.

| System | Method |
|---|---|
| A · Raw | no normalization (control) |
| B · Rule-based | Amin abbreviation + misspelling + keep lists |
| C · Transformer | ByT5 char-level seq2seq |
| D · Hybrid | rules then ByT5 |

## Intrinsic metrics (all records)

| system | records | raw_vocab | norm_vocab | vocab_reduction_pct | raw_oov | norm_oov | oov_reduction_pct | expansions | expansion_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A · Raw | 6169 | 2246 | 2246 | 0.0 | 0.1056 | 0.1056 | 0.0 | 0 | 0.0 |
| B · Rule-based | 6169 | 2246 | 2140 | 4.72 | 0.1056 | 0.0108 | 89.76 | 17196 | 0.1882 |
| C · Transformer (ByT5) | 6169 | 2246 | 1954 | 13.0 | 0.1056 | 0.0037 | 96.45 | 0 | 0.0 |
| D · Hybrid | 6169 | 2246 | 1934 | 13.89 | 0.1056 | 0.0032 | 96.96 | 17196 | 0.1882 |

## Extrinsic metrics (vs Amin gold)

| system | records | wer | cer | exact_match | err_word | err_char |
| --- | --- | --- | --- | --- | --- | --- |
| A · Raw | 926 | 0.3228 | 0.2023 | 0.0389 | 0.0 | 0.0 |
| B · Rule-based | 926 | 0.0588 | 0.0372 | 0.6188 | 0.8178 | 0.8161 |
| C · Transformer (ByT5) | 926 | 0.0337 | 0.0257 | 0.8229 | 0.8955 | 0.8728 |
| D · Hybrid | 926 | 0.0474 | 0.0293 | 0.7268 | 0.8532 | 0.8553 |

`wer`/`cer` lower is better; `exact_match`, `err_word`, `err_char` higher is better. **ERR** = fraction of raw error removed (MaintNorm's metric; they report 95.8%).
