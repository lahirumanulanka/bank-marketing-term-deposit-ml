# Bank Marketing Term Deposit Prediction

Comprehensive end-to-end machine learning project to predict term deposit subscription using the UCI Bank Marketing datasets.

## 📋 Project Overview

This project implements a complete ML pipeline for predicting whether a client will subscribe to a term deposit based on direct marketing campaign data from a Portuguese banking institution (2008-2010).

### Key Highlights
- **Dataset**: Merged UCI Bank Marketing datasets (~86,400 samples, 21 features)
- **Models**: 6 different ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Neural Network)
- **MLflow Tracking**: All experiments tracked and versioned
- **Interpretability**: SHAP, LIME, and feature importance analysis
- **Production Ready**: Complete deployment strategy with Docker, Kubernetes, CI/CD

## 🎯 Business Objective

Predict client subscription to term deposits to:
- Reduce marketing costs by targeting likely subscribers
- Improve customer experience by reducing unwanted calls
- Optimize resource allocation and campaign timing
- Increase conversion rates and revenue

## 📊 Project Structure
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

## 📚 Complete Task Notebooks

All 8 coursework tasks are implemented in comprehensive Jupyter notebooks:

### ✅ [Notebook 1: Dataset Justification & Literature Review](notebooks/01_dataset_justification_and_literature_review.ipynb)
- Dataset source and structure documentation
- Business problem definition and significance
- Literature survey of 5+ peer-reviewed studies
- Comparison with existing research

### ✅ [Notebook 2: Data Merging & Preprocessing](notebooks/02_data_merging_and_preprocessing.ipynb)
- Loading bank-full.csv (45,211 rows) and bank-additional-full.csv (41,188 rows)
- Column alignment and dataset merging strategy
- Final merged dataset: 86,399 rows × 21 columns
- Data saved to `data/raw/` and `data/interim/`

### ✅ [Notebook 3: Exploratory Data Analysis](notebooks/03_exploratory_data_analysis.ipynb)
- Comprehensive EDA with 15+ visualizations
- Missing values analysis and handling strategy
- Outlier detection using IQR method
- Class imbalance analysis (~88:12 ratio)
- **Feature Engineering** - Created 5 new features:
  - `contact_frequency`: Campaign contact categorization
  - `previous_campaign_success`: Past interaction outcomes
  - `age_group`: Life stage segmentation
  - `has_economic_data`: Data source indicator
  - `duration_category`: Call length categorization

### ✅ [Notebook 4: Model Development](notebooks/04_model_development.ipynb)
Implemented 6 machine learning models:
1. **Logistic Regression** (Linear Model) - Baseline with balanced weights
2. **Random Forest** (Tree-based) - 100 estimators, depth 10
3. **XGBoost** (Boosting) - Optimized for imbalanced data
4. **LightGBM** (Boosting) - Fast gradient boosting
5. **CatBoost** (Boosting) - Automatic categorical handling
6. **Neural Network** (PyTorch) - 4-layer architecture with dropout

All models tracked in MLflow with parameters and metrics.

### ✅ [Notebook 5: Evaluation & Comparison](notebooks/05_evaluation_and_comparison.ipynb)
- Multiple metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices for all models
- ROC and Precision-Recall curves comparison
- Error analysis for misclassified samples
- Hyperparameter tuning with GridSearchCV
- Threshold optimization for business requirements
- SMOTE for handling class imbalance

### ✅ [Notebook 6: Interpretability & Insights](notebooks/06_interpretability_and_insights.ipynb)
- Feature importance analysis
- **SHAP** values for global and local explanations
- **LIME** for individual prediction explanations
- Permutation importance
- Partial dependence plots
- **Business insights** and actionable recommendations
- Marketing strategy optimization

### ✅ [Notebook 7: Critical Reflection](notebooks/07_critical_reflection.ipynb)
- Dataset limitations (temporal, geographic, features)
- Ethical implications (privacy, discrimination, transparency)
- Bias analysis (selection, historical, measurement)
- Fairness evaluation across demographics
- Generalizability concerns
- **Future extensions**:
  - Deep learning (LSTM, Transformers, GNN)
  - Causal inference and uplift modeling
  - Reinforcement learning for dynamic campaigns
  - Federated learning for privacy

### ✅ [Notebook 8: Deployment Strategy](notebooks/08_deployment_strategy.ipynb)
- Model packaging and serialization
- **FastAPI** application with health checks
- **Docker** containerization (Dockerfile + docker-compose)
- **Kubernetes** deployment manifests
- **MLflow** model serving
- Cloud deployment (AWS SageMaker, Azure ML, GCP AI Platform)
- **CI/CD** pipeline with GitHub Actions
- Monitoring with Prometheus and Grafana
- A/B testing framework
- Complete deployment checklist

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml.git
cd bank-marketing-term-deposit-ml
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebooks
```bash
jupyter notebook
```

Navigate to `notebooks/` and execute notebooks in order (01 through 08).

### 5. Train Models (Alternative: Using Scripts)
```bash
# Run all preprocessing and training
python scripts/train.py --config config/model_xgboost.yaml
```

### 6. View MLflow Experiments
```bash
mlflow ui --backend-store-uri experiments/mlruns
# Open http://localhost:5000 in browser
```

## 📊 Key Results

### Dataset Statistics
- **Total Samples**: 86,399 (merged dataset)
- **Features**: 20 input features + 1 target variable
- **Class Distribution**: ~88% No, ~12% Yes (imbalanced)
- **Data Sources**: 
  - bank-full.csv: 45,211 rows (16 features)
  - bank-additional-full.csv: 41,188 rows (20 features)

### Model Performance
All models evaluated with:
- Cross-validation
- Class imbalance handling
- Threshold optimization
- Multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

Best performing models tracked in MLflow for reproducibility.

### Feature Insights
Top influential features (based on SHAP analysis):
1. Call duration (strongest predictor, but only available post-call)
2. Previous campaign outcome
3. Economic indicators (employment rate, euribor3m)
4. Contact timing (month, day)
5. Client demographics (age, job, education)

## 🛠️ Technology Stack

### Machine Learning
- **scikit-learn**: Traditional ML algorithms
- **XGBoost, LightGBM, CatBoost**: Gradient boosting
- **PyTorch**: Neural networks
- **imbalanced-learn**: SMOTE for class imbalance

### Experiment Tracking
- **MLflow**: Experiment tracking, model registry, serving

### Explainability
- **SHAP**: Global and local model interpretability
- **LIME**: Local interpretable explanations

### Deployment
- **FastAPI**: REST API development
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Prometheus + Grafana**: Monitoring

### Data & Visualization
- **pandas, NumPy**: Data manipulation
- **matplotlib, seaborn, plotly**: Visualization

## 📈 Project Deliverables

✅ **8 Comprehensive Jupyter Notebooks** covering all coursework tasks  
✅ **Literature Review** with 5+ peer-reviewed references  
✅ **Merged Dataset** with proper column alignment  
✅ **Feature Engineering** with 5 new features  
✅ **6 ML Models** from different families  
✅ **MLflow Tracking** for reproducibility  
✅ **Model Interpretability** with SHAP and LIME  
✅ **Critical Analysis** of limitations and ethics  
✅ **Production Deployment Strategy** with Docker, K8s, CI/CD  

## 🔬 Experiment Tracking

All experiments are tracked in MLflow:
```bash
# View experiments
mlflow ui

# Access at http://localhost:5000
```

Tracked information:
- Model parameters and hyperparameters
- Training metrics (Accuracy, F1, ROC-AUC, etc.)
- Model artifacts (saved models, preprocessors)
- Visualizations (confusion matrices, ROC curves)

## 🚢 Deployment

### Local Deployment
```bash
cd deployment
docker-compose up
```

Access API at: `http://localhost:8000/docs` (FastAPI Swagger UI)

### Cloud Deployment

**AWS SageMaker:**
```python
# See notebook 08 for complete example
from sagemaker.sklearn import SKLearnModel
model.deploy(instance_type='ml.t2.medium')
```

**Azure ML:**
```python
# See notebook 08 for complete example
from azureml.core import Model
Model.deploy(workspace=ws, name='bank-marketing-service')
```

**GCP AI Platform:**
```bash
# See notebook 08 for complete commands
gcloud ai-platform versions create v1 --model=bank_marketing
```

## 📊 Monitoring & Observability

- **Prometheus**: Metrics collection (prediction latency, throughput, model confidence)
- **Grafana**: Dashboards for visualization
- **Logging**: Structured JSON logs for all predictions
- **Alerting**: Automated alerts for model degradation

Access Grafana: `http://localhost:3000` (admin/admin)

## 🤝 Contributing

This is an academic project. For suggestions or issues:
1. Open an issue describing the problem
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Bank Marketing dataset
- **Moro et al. (2011, 2014)** for original research and dataset creation
- **Portuguese Banking Institution** for data collection

## 📧 Contact

**Author**: Lahiru Manulanka Munasinghe  
**GitHub**: [@lahirumanulanka](https://github.com/lahirumanulanka)

---

**Note**: This project demonstrates end-to-end ML pipeline development for academic purposes. For production deployment, ensure compliance with GDPR, fair lending regulations, and ethical AI guidelines.
