
## Summary grid

| config | n | macro-recall | top-1 | top-3 | MRR | strong cov | strong acc |
|---|---|---|---|---|---|---|---|
| System-D | 5,336 | 38.3% | 78.7% | 90.4% | 0.78 | 85.0% | 86.4% |
| System-D +rr | 5,336 | 38.2% | 78.8% | 90.4% | 0.784 | 80.7% | 84.1% |
| Raw | 5,336 | 36.7% | 80.2% | 91.3% | 0.783 | 82.2% | 87.8% |
| Raw +rr | 5,336 | 36.6% | 80.2% | 91.2% | 0.786 | 81.5% | 86.0% |

### System-D  (reranker off)

- evaluated: **5,336** (cluster-safe LOO)
- macro-recall: **38.3%**  (majority 14.3%)
- top-1 agreement: **78.7%**  (majority 79.2%)
- top-3 agreement: **90.4%**   ·   MRR **0.78**

| tier | coverage | system acc | majority acc |
|---|---|---|---|
| strong | 85.0% | 86.4% | 81.1% |
| moderate | 4.0% | 50.0% | 43.9% |
| exploratory | 11.0% | 30.1% | 77.6% |

| action family | n | recall |
|---|---|---|
| Replace | 4227 | 86.8% |
| Adjust | 452 | 69.9% |
| Diagnose | 250 | 64.8% |
| Service | 154 | 15.6% |
| Inspect | 151 | 4.0% |
| Repair | 92 | 27.2% |
| Calibrate | 10 | 0.0% |

### System-D  (reranker ON)

- evaluated: **5,336** (cluster-safe LOO)
- macro-recall: **38.2%**  (majority 14.3%)
- top-1 agreement: **78.8%**  (majority 79.2%)
- top-3 agreement: **90.4%**   ·   MRR **0.784**

| tier | coverage | system acc | majority acc |
|---|---|---|---|
| strong | 80.7% | 84.1% | 78.1% |
| moderate | 8.3% | 93.0% | 92.3% |
| exploratory | 11.1% | 29.9% | 77.5% |

| action family | n | recall |
|---|---|---|
| Replace | 4227 | 86.9% |
| Adjust | 452 | 69.9% |
| Diagnose | 250 | 65.2% |
| Service | 154 | 15.6% |
| Inspect | 151 | 4.0% |
| Repair | 92 | 26.1% |
| Calibrate | 10 | 0.0% |

### Raw  (reranker off)

- evaluated: **5,336** (cluster-safe LOO)
- macro-recall: **36.7%**  (majority 14.3%)
- top-1 agreement: **80.2%**  (majority 79.1%)
- top-3 agreement: **91.3%**   ·   MRR **0.783**

| tier | coverage | system acc | majority acc |
|---|---|---|---|
| strong | 82.2% | 87.8% | 81.8% |
| moderate | 4.3% | 47.8% | 45.3% |
| exploratory | 13.5% | 43.7% | 73.7% |

| action family | n | recall |
|---|---|---|
| Replace | 4220 | 89.2% |
| Adjust | 475 | 65.1% |
| Diagnose | 259 | 59.1% |
| Service | 165 | 14.5% |
| Inspect | 111 | 1.8% |
| Repair | 96 | 27.1% |
| Calibrate | 10 | 0.0% |

### Raw  (reranker ON)

- evaluated: **5,336** (cluster-safe LOO)
- macro-recall: **36.6%**  (majority 14.3%)
- top-1 agreement: **80.2%**  (majority 79.1%)
- top-3 agreement: **91.2%**   ·   MRR **0.786**

| tier | coverage | system acc | majority acc |
|---|---|---|---|
| strong | 81.5% | 86.0% | 79.7% |
| moderate | 4.9% | 84.4% | 83.7% |
| exploratory | 13.5% | 43.6% | 73.6% |

| action family | n | recall |
|---|---|---|
| Replace | 4220 | 89.2% |
| Adjust | 475 | 65.1% |
| Diagnose | 259 | 59.8% |
| Service | 165 | 14.5% |
| Inspect | 111 | 1.8% |
| Repair | 96 | 26.0% |
| Calibrate | 10 | 0.0% |

**Macro-recall delta (System-D − Raw, reranker off): 1.6 pts**
**Reranker effect on System-D (macro-recall): -0.1 pts** (38.3% → 38.2%)
**Reranker effect on Raw (macro-recall): -0.1 pts** (36.7% → 36.6%)
