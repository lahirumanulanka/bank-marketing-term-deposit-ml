# Bank Marketing Term Deposit Prediction

End-to-end machine learning project to predict term deposit subscription (y) using the UCI Bank Marketing datasets (original + additional).

## Project Structure
```
├── dataset/                # Original raw dataset copies (immutable reference)
├── data/
│   ├── raw/                # Working copy of original data
│   ├── interim/            # Data after cleaning / encoding steps
│   ├── processed/          # Final feature matrices ready for modeling
├── notebooks/              # Jupyter notebooks for EDA, modeling prototypes
├── src/                    # Reusable, testable python package code
│   ├── data/               # Data loading & cleaning modules
│   ├── features/           # Feature engineering & transformations
│   ├── models/             # Model definitions & training utilities
│   ├── pipeline/           # End-to-end training / inference pipelines
│   ├── evaluation/         # Metrics, error analysis, comparison
│   ├── visualization/      # Plotting utilities
├── config/                 # YAML/JSON configuration files (data, model, logging)
├── models/                 # Persisted trained model artifacts (DO NOT COMMIT large files)
├── experiments/            # MLflow or experiment tracking outputs
├── deployment/             # Dockerfile, app code (FastAPI/Flask), infra scripts
├── monitoring/             # Model drift, data quality monitoring scripts
├── scripts/                # CLI helper scripts (train, evaluate, deploy)
├── tests/                  # Unit & integration tests
├── reports/                # Generated reports
│   └── figures/            # Saved plots (EDA, metrics, SHAP)
├── docs/                   # Extended documentation (literature review, design)
```

## Key Tasks Mapping
| Coursework Task | Folder(s) |
|-----------------|-----------|
| Dataset Justification & Literature Review | `docs/`, `README.md` |
| EDA & Preprocessing | `notebooks/`, `src/data/`, `src/features/`, `reports/figures/` |
| Model Development | `src/models/`, `src/pipeline/`, `config/model_*.yaml` |
| Evaluation & Comparison | `src/evaluation/`, `reports/` |
| Interpretability | `src/evaluation/`, `reports/figures/`, `notebooks/` |
| Critical Reflection | `docs/limitations.md` |
| Deployment | `deployment/`, `monitoring/` |

## Getting Started
1. Create virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run basic data profiling:
```
python scripts/profile_data.py --config config/data_config.yaml
```
4. Train models:
```
python scripts/train.py --config config/model_logreg.yaml
```

## Experiment Tracking
MLflow will log to `experiments/mlruns` (configure in `config/mlflow.yaml`).

## Deployment Plan (High-Level)
- Package best model & preprocessing pipeline (joblib)
- Serve via FastAPI + Docker
- CI/CD: GitHub Actions builds & pushes image
- Monitoring: Collect request/response + drift metrics -> Prometheus/Grafana

## Next Steps
- Fill literature review in `docs/literature_review.md`
- Implement data cleaning in `src/data/cleaning.py`
- Build baseline Logistic Regression model
- Add SHAP explainability

---
(Placeholder README – expand as project evolves)
