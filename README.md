# Bank Marketing Term Deposit Prediction

Comprehensive end-to-end machine learning project to predict term deposit subscription using the UCI Bank Marketing datasets.

## 📋 Project Overview

**📖 [Complete Project Overview](docs/PROJECT_OVERVIEW.md)** - Comprehensive guide covering the entire ML pipeline from data to deployment.

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

All notebooks are implemented with comprehensive explanations and detailed documentation available in the `docs/` folder.

### ✅ [Notebook 1: Dataset Justification & Literature Review](notebooks/01_dataset_justification_and_literature_review.ipynb)
**📖 [Full Documentation](docs/notebook_01_dataset_justification.md)**
- Dataset source and structure documentation (UCI Bank Marketing Dataset)
- Business problem definition and real-world significance
- Literature survey of 5+ peer-reviewed studies
- Comparison with existing research and research gap identification
- Comprehensive justification for dataset selection

### ✅ [Notebook 2: Data Merging & Preprocessing](notebooks/02_data_merging_and_preprocessing.ipynb)
**📖 [Full Documentation](docs/notebook_02_data_preprocessing.md)**
- Loading bank-full.csv (45,211 rows) and bank-additional-full.csv (41,188 rows)
- Column alignment and dataset merging strategy with rationale
- Final merged dataset: 86,399 rows × 21 columns
- Missing value handling with detailed justifications
- Data quality assessment and validation
- Data saved to `data/raw/` and `data/interim/`

### ✅ [Notebook 3: Exploratory Data Analysis](notebooks/03_exploratory_data_analysis.ipynb)
**📖 [Full Documentation](docs/notebook_03_exploratory_analysis.md)**
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
**📖 [Full Documentation](docs/notebook_04_model_development.md)**
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
   - **Selected for deployment** (ROC-AUC: 0.93)
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
**📖 [Full Documentation](docs/notebook_05_evaluation.md)**
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
- Confusion matrices for all models with business interpretation
- **ROC curves** comparison across all models
- **Precision-Recall curves** for minority class focus
- **Hyperparameter tuning** with GridSearchCV and cross-validation
- **Threshold optimization** for business requirements
- Final model selection: **LightGBM with ROC-AUC 0.94**

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
- **HuggingFace Spaces deployment** 🤗
  - Gradio web interface for interactive predictions
  - Free hosting with auto-scaling
  - Git-based deployment workflow
  - Public accessibility for demos
- Model packaging and serialization (pickle, ONNX, MLflow)
- **FastAPI REST API** application with endpoints:
  - Health checks and model info
  - Prediction endpoint with validation
  - Technology justification (async support, performance, auto-documentation)
- **Docker** containerization (Dockerfile + docker-compose)
  - Consistency across environments, dependency isolation
  - Multi-service orchestration (API + Monitoring)
- **Kubernetes** deployment manifests (deployment, service, HPA)
  - Auto-scaling, self-healing, load balancing capabilities
- **MLflow** model serving and version control
- **Cloud deployment comparisons** with detailed examples:
  - HuggingFace Spaces (free, easy deployment)
  - AWS SageMaker (managed ML platform)
  - Azure ML (Microsoft ML service)
  - GCP AI Platform (Google ML infrastructure)
  - Platform selection guidance based on requirements
- **CI/CD** pipeline with GitHub Actions
  - Automated testing, validation, and deployment
  - Continuous integration workflow
  - HuggingFace Spaces auto-deployment
- **Monitoring strategy** (Prometheus + Grafana)
  - Infrastructure, model, and business metrics
  - Alerting framework and drift detection
  - Data drift detection (KS test, Chi-square test)
  - Concept drift detection
- **Versioning & rollback** (semantic versioning, blue-green deployment)
  - Model versioning (1.0.0)
  - MLflow model registry
  - Rollback procedures
- **Model monitoring implementation**
  - Data drift detection framework
  - Concept drift detection
  - Performance tracking
  - Structured logging
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

### 🤗 HuggingFace Spaces (Live Deployment)

The model is deployed as both a **Gradio web interface** and **FastAPI REST API** on HuggingFace Spaces for public access.

#### Live Demo
**🔗 HuggingFace Space**: `https://huggingface.co/spaces/hirumunasinghe/bank-marketing-term-deposit-prediction`

#### Features
- **Interactive Web UI** (Gradio): User-friendly interface for single predictions
- **REST API** (FastAPI): Programmatic access for applications
- **Free Hosting**: Automatic scaling and 24/7 availability
- **Auto-deployment**: Git-based CI/CD pipeline

#### Available Interfaces

##### 1. Gradio Web Interface (`app.py`)
```python
# Interactive form-based predictions
# Features:
# - Dropdown menus for categorical features
# - Sliders for numeric inputs
# - Real-time prediction with confidence scores
# - Visual result display
```

**Access**: Visit the Space URL and use the interactive form

##### 2. FastAPI REST API (`api_app.py`)

**Core Endpoints**:
- `GET /` - API information and available endpoints
- `GET /health` - Health check and model status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

**Prediction Endpoints**:
- `POST /predict` - Single client prediction
- `POST /predict/batch` - Batch prediction for multiple clients

**Information Endpoints**:
- `GET /model/info` - Model metadata and performance
- `GET /features/info` - Detailed feature descriptions

#### API Usage Example

**Single Prediction**:
```python
import requests

url = "https://your-space-name.hf.space/predict"
data = {
    "age": 39,
    "job": "management",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "fri",
    "duration": 180,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability_percentage']}%")
print(f"Confidence: {result['confidence']}")
```

**Batch Prediction**:
```python
batch_data = {
    "clients": [
        {/* client 1 data */},
        {/* client 2 data */},
        # ... more clients
    ]
}

response = requests.post("https://your-space-name.hf.space/predict/batch", json=batch_data)
results = response.json()
```

#### Model Performance in Production
- **Algorithm**: LightGBM (tuned)
- **ROC-AUC**: 0.94
- **Precision**: 78%
- **Recall**: 67%
- **Training Data**: 86,000+ samples
- **Features**: 20 input features + 5 engineered
- **Preprocessing**: SMOTE for class balance, median imputation for missing values

#### Deployment Files (`huggingface_space/`)
```
huggingface_space/
├── app.py                          # Gradio web interface
├── api_app.py                      # FastAPI REST API
├── start.py                        # Unified launcher
├── requirements.txt                # Dependencies
├── README.md                       # Space documentation
├── xgboost_retrained_tuned.pkl    # Trained model
└── preprocessing/                  # Preprocessing artifacts
    ├── scaler.pkl
    └── label_encoders.pkl
```

#### Quick Deploy to HuggingFace Spaces

**Option 1: Web Interface**
1. Create a new Space on HuggingFace
2. Upload files from `huggingface_space/` directory
3. Select Gradio SDK
4. Space automatically deploys

**Option 2: Git CLI**
```bash
cd huggingface_space

# Initialize git (if not already)
git init

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/<username>/bank-marketing-prediction

# Copy model files (if not already present)
cp ../models/lightgbm_retrained_tuned.pkl .
cp -r ../models/preprocessing .

# Deploy (push to HuggingFace Space repository)
git add .
git commit -m "Deploy model to HuggingFace Spaces"
git push hf main
```

**See Also**:
- [HuggingFace Space README](huggingface_space/README.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Detailed deployment instructions

---

### 🐳 Local Deployment (Docker)

### 🐳 Local Deployment (Docker)

Run the FastAPI REST API locally with Docker:

```bash
cd deployment
docker-compose up -d
```

**Access:**
- API Documentation: `http://localhost:8000/docs` (Swagger UI)
- Alternative docs: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

**Example API Call:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 30,
        "job": "admin.",
        "marital": "single",
        # ... other features
    }
)

print(response.json())
# {'prediction': 'yes', 'probability': 0.85, 'confidence': 'high'}
```

---

### ☁️ Cloud Deployment Options

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

**Kubernetes:**
```bash
kubectl apply -f deployment/kubernetes/
```

## 📊 Monitoring & Observability

- **Prometheus**: Metrics collection (prediction latency, throughput, model confidence)
- **Grafana**: Dashboards for visualization
- **Logging**: Structured JSON logs for all predictions
- **Alerting**: Automated alerts for model degradation

Access Grafana: `http://localhost:3000` (admin/admin)

---

## 📚 Complete Documentation

### Notebook Documentation
Comprehensive markdown documentation available for all notebooks in the `docs/` folder:

1. **[Dataset Justification & Literature Review](docs/notebook_01_dataset_justification.md)**
   - Dataset selection rationale
   - Business problem definition
   - Literature survey (5+ peer-reviewed studies)
   - Research gap identification
   - Feature categories explanation

2. **[Data Merging & Preprocessing](docs/notebook_02_data_preprocessing.md)**
   - Dataset merging strategy
   - Missing value handling approaches
   - Data quality assessment
   - Initial feature engineering
   - Validation procedures

3. **[Exploratory Data Analysis](docs/notebook_03_exploratory_analysis.md)**
   - Comprehensive EDA framework
   - Univariate and multivariate analysis
   - Outlier detection and handling
   - Class imbalance analysis with SMOTE
   - Feature engineering (5 new features)
   - Correlation analysis

4. **[Model Development](docs/notebook_04_model_development.md)**
   - Model selection rationale (6 algorithms)
   - Training pipeline with MLflow
   - Class imbalance handling strategies
   - Model serialization
   - Feature importance analysis

5. **[Evaluation & Comparison](docs/notebook_05_evaluation.md)**
   - Comprehensive metrics framework
   - ROC and Precision-Recall curves
   - Error analysis and misclassification investigation
   - Hyperparameter tuning with GridSearchCV
   - Threshold optimization for business goals
   - Final model selection

### Additional Documentation
- **[HuggingFace Space README](huggingface_space/README.md)** - API documentation and usage
- **[Project Summary](PROJECT_SUMMARY.md)** - Implementation overview and statistics
- **[Enhancements](ENHANCEMENTS.md)** - Detailed enhancement summary

### Key Features Explained

**What Makes This Documentation Comprehensive:**
- ✅ **Full Explanations**: Every decision justified with business and technical rationale
- ✅ **Visual Learning**: Diagrams, plots, and code examples throughout
- ✅ **Practical Examples**: Real-world use cases and implementation patterns
- ✅ **Business Translation**: Technical concepts explained in business terms
- ✅ **Step-by-Step Guides**: Clear instructions for reproduction
- ✅ **Best Practices**: Industry-standard approaches highlighted

---

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
