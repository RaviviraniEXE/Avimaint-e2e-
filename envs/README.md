# Environment Catalogue

Each folder defines one isolated Conda environment and its pip requirements.

| Key | Conda name | Responsibility |
|---|---|---|
| core | avimaint-core | Contracts, adapters, validation, splits, metrics |
| normalization | avimaint-normalization | Rule, ByT5 and hybrid normalization |
| ie-classical | avimaint-ie-classical | CRF and logistic regression |
| ie-neural | avimaint-ie-neural | BiLSTM and transformer IE |
| spert | avimaint-spert | SpERT only |
| retrieval | avimaint-retrieval | Sparse, dense and hybrid retrieval |
| dashboard | avimaint-dashboard | Streamlit and visualization |
| dev | avimaint-dev | Tests, linting, notebooks |

Create them with scripts/setup/setup_one.ps1 or setup_one.sh. The scripts also install this repository as an editable package without pulling dependencies from another environment.

