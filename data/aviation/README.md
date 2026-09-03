# Aviation data placement

Raw data is excluded from Git and from this scaffold.

- `raw/Aircraft_Annotation_DataFile.csv`: immutable source with `IDENT`, `PROBLEM`, `ACTION`.
- `reference/amin_cleaned_dataset.csv`: included expert-expanded reference to audit.
- `interim/`: generated pair audit and human-review sheet.
- `processed/`: approved training pairs and optional silver training pairs.

Never replace the raw file with normalized text. All generated rows must preserve
the source `IDENT` and field name.
