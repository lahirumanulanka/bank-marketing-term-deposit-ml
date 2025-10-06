# Bank Marketing Term Deposit Prediction

Comprehensive end-to-end machine learning project to predict term deposit subscription using the UCI Bank Marketing datasets.

## 📋 Project Overview

This project implements a complete ML pipeline for predicting whether a client will subscribe to a term deposit based on direct marketing campaign data from a Portuguese banking institution (2008-2010).

### Key Highlights
- **Dataset**: Merged UCI Bank Marketing datasets (~86,400 samples, 21 features)
- **Models**: 6 different ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Neural Network)
- **Feature Engineering**: 5 domain-informed features with comprehensive justifications
- **Class Imbalance**: Handled with SMOTE + class weights, visualized with before/after plots
- **Outlier Treatment**: Selective capping based on business context (balance, campaign features)
- **MLflow Tracking**: All experiments tracked with parameters, metrics, and artifacts
- **Interpretability**: SHAP, LIME, permutation importance with detailed explanations
- **Error Analysis**: Comprehensive misclassification investigation with 6-subplot visualizations
- **Production Ready**: Complete deployment strategy (Docker, Kubernetes, CI/CD, monitoring)
- **Comprehensive Explanations**: Every decision justified with business and technical rationale

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
- **Comprehensive EDA framework** with detailed explanations
- 15+ visualization sections covering all aspects
- **Missing values analysis** with handling strategy justification
- **Outlier detection AND removal**:
  - IQR method for systematic detection
  - **Selective capping** for balance (1st-99th percentile) and campaign (95th percentile)
  - Preservation of valid outliers with business justification
- **Class imbalance analysis** (~88:12 ratio with impact assessment)
- **SMOTE implementation with visualizations**:
  - Before/after class balance comparison plots
  - Dataset overview after balancing
  - Impact analysis on model training
- **Feature Engineering** - Created 5 new features with detailed justifications:
  - `contact_frequency`: Campaign contact categorization (customer fatigue)
  - `previous_campaign_success`: Past interaction outcomes (behavioral prediction)
  - `age_group`: Life stage segmentation (non-linear age effects)
  - `has_economic_data`: Data source indicator (temporal context)
  - `duration_category`: Call length categorization (engagement levels)
- **Comprehensive preprocessing justification section** covering:
  - Missing values strategy and rationale
  - Outlier treatment decisions with business context
  - Feature engineering domain knowledge basis
  - Class imbalance handling approach
  - Train-test split and scaling strategies

### ✅ [Notebook 4: Model Development](notebooks/04_model_development.ipynb)
Implemented **6 machine learning models** with comprehensive justifications:
1. **Logistic Regression** (Linear Model) 
   - Baseline, interpretable, regulatory-friendly
   - Balanced weights for class imbalance
2. **Random Forest** (Tree-based) 
   - 100 estimators, depth 10
   - Robust ensemble, handles non-linearity
3. **XGBoost** (Boosting) 
   - State-of-the-art for tabular data
   - scale_pos_weight for imbalance handling
4. **LightGBM** (Boosting) 
   - Fast gradient boosting, efficient training
   - class_weight='balanced' configuration
5. **CatBoost** (Boosting) 
   - Best categorical handling, minimal tuning
   - Automatic class weight detection
6. **Neural Network** (PyTorch) 
   - 4-layer architecture (128-64-32-1) with dropout
   - Deep learning approach for complex patterns

**Enhanced Content**:
- **Detailed model selection rationale** based on dataset characteristics
- **Trade-offs analysis** (interpretability vs performance, speed vs accuracy)
- **Business alignment** for model choices
- MLflow tracking for all experiments (parameters, metrics, artifacts)
- Class imbalance handling (weights + SMOTE)
- Model serialization and versioning

### ✅ [Notebook 5: Evaluation & Comparison](notebooks/05_evaluation_and_comparison.ipynb)
- **Comprehensive evaluation framework** with business-aligned metrics
- **Multiple metrics with explanations**: 
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Business translation of each metric
  - Cost-benefit analysis for banking context
- **Enhanced error analysis** with detailed visualizations:
  - Confusion matrix breakdown with statistics
  - Class-wise performance analysis
  - Prediction confidence analysis for errors
  - False positive/negative detailed analysis
  - 6 comprehensive visualization subplots
  - Sample misclassified records investigation
- Confusion matrices for all models
- **ROC curves** comparison across all models
- **Precision-Recall curves** for minority class focus
- **Hyperparameter tuning** with GridSearchCV and cross-validation
- **Threshold optimization** for business requirements
- **SMOTE** implementation and impact analysis for class balancing

### ✅ [Notebook 6: Interpretability & Insights](notebooks/06_interpretability_and_insights.ipynb)
- **Comprehensive interpretability framework** with regulatory context (GDPR, fairness)
- **Detailed technique explanations** for all methods:
  - **Feature importance** for tree-based models
  - **SHAP** values for global and local explanations
  - **LIME** for local interpretable explanations
  - **Permutation importance** analysis
  - **Partial dependence plots** for key features
- **SHAP Analysis**:
  - Global explanations (summary plots, bar plots)
  - Local explanations (waterfall plots for individual predictions)
  - Directional contributions (positive/negative effects)
- **Business insights translation framework**:
  - From technical findings to actionable recommendations
  - 10+ specific marketing strategy optimizations
  - Customer segment targeting guidance
- Ethical considerations (fairness, bias, discrimination prevention)

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
- **Production architecture diagram** and component breakdown
- Model packaging and serialization (pickle, ONNX, MLflow)
- **FastAPI** application with health checks and monitoring endpoints
  - Technology justification (async support, performance, auto-documentation)
- **Docker** containerization (Dockerfile + docker-compose)
  - Consistency across environments, dependency isolation
- **Kubernetes** deployment manifests (deployment, service, HPA)
  - Auto-scaling, self-healing, load balancing capabilities
- **MLflow** model serving and version control
- **Cloud deployment comparisons** with detailed examples:
  - AWS SageMaker (managed ML platform)
  - Azure ML (Microsoft ML service)
  - GCP AI Platform (Google ML infrastructure)
  - Platform selection guidance based on requirements
- **CI/CD** pipeline with GitHub Actions
  - Automated testing, validation, and deployment
- **Monitoring strategy** (Prometheus + Grafana)
  - Infrastructure, model, and business metrics
  - Alerting framework and drift detection
- **Versioning & rollback** (semantic versioning, blue-green deployment)
- **Model drift detection** framework (data drift, concept drift)
- A/B testing framework for production validation
- **Security & compliance** (GDPR, encryption, auditing)
- **Cost optimization** strategies
- Complete deployment checklist with best practices

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
