# 🏦 Bank Marketing Term Deposit Prediction

> **Complete End-to-End Machine Learning Pipeline for Banking Marketing Campaign Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)](https://mlflow.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

## 🎯 Project Overview

This project implements a **production-ready machine learning pipeline** for predicting term deposit subscriptions from bank marketing campaigns. Using the UCI Bank Marketing dataset (86,000+ samples), we develop, evaluate, and deploy a classification model achieving **94% ROC-AUC**.

### 🚀 Key Achievements
- **67% Recall**: Identifies 2 out of 3 potential subscribers
- **78% Precision**: Reduces wasted calls by ~50%
- **Cost Savings**: Estimated €500K+ annually for large banks
- **Revenue Increase**: Estimated €2M+ through better targeting
- **Production Deployment**: Live API on Hugging Face Spaces

---

## 📁 Project Structure

```
bank-marketing-term-deposit-ml/
├── 📊 notebooks/                          # Jupyter notebooks for analysis
│   ├── 01_dataset_justification_and_literature_review.ipynb
│   ├── 02_data_merging_and_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_model_development.ipynb
│   └── 05_evaluation_and_comparison.ipynb
│
├── 📚 docs/                               # Comprehensive documentation
│   ├── PROJECT_OVERVIEW.md               # Complete project guide (493 lines)
│   ├── notebook_01_data_preprocessing.md # Data merging & cleaning (450 lines)
│   ├── notebook_02_exploratory_analysis.md # EDA & feature engineering (721 lines)
│   ├── notebook_03_model_development.md  # Model training & selection (18KB)
│   └── notebook_04_evaluation.md         # Performance evaluation (21KB)
│
├── 🗂️ data/                              # Data storage hierarchy
│   ├── raw/                              # Original datasets
│   │   ├── bank_merged_raw.csv           # Combined dataset (86K+ rows)
│   │   └── bank_merged_raw.pkl           # Serialized version
│   └── interim/                          # Processed datasets
│       ├── bank_cleaned_outliers.pkl     # Outlier-treated data
│       ├── bank_with_features.csv        # Feature-engineered data
│       └── bank_with_features.pkl        # Serialized features
│
├── 📈 dataset/                           # Original UCI datasets
│   ├── bank/                            # Original dataset (2011)
│   │   ├── bank-full.csv                # 45,211 samples, 16 features
│   │   ├── bank.csv                     # Subset version
│   │   └── bank-names.txt               # Feature descriptions
│   └── bank-additional/                 # Enhanced dataset (2014)
│       ├── bank-additional-full.csv     # 41,188 samples, 20 features
│       ├── bank-additional.csv          # Subset version
│       └── bank-additional-names.txt    # Feature descriptions
│
├── 🤖 models/                            # Trained models & preprocessing
│   ├── xgboost_retrained_tuned.pkl      # XGBoost model
│   ├── lightgbm_retrained_tuned.pkl     # LightGBM model (selected)
│   ├── catboost_retrained_tuned.pkl     # CatBoost model
│   ├── random_forest_retrained_tuned.pkl # Random Forest model
│   ├── logistic_regression_retrained_tuned.pkl # Logistic Regression
│   ├── neural_network_state_dict_retrained_tuned.pt # PyTorch NN
│   ├── preprocessing/                    # Data preprocessing objects
│   │   ├── label_encoders.pkl           # Categorical encoders
│   │   └── scaler.pkl                   # Feature scaler
│   └── catboost_info/                   # CatBoost training logs
│
├── 🧪 experiments/                       # MLflow experiment tracking
│   └── mlruns/                          # MLflow artifacts & metrics
│       ├── 0/                           # Default experiment
│       ├── 693545408735532194/          # Experiment tracking
│       └── 860277220465355361/          # Model runs & artifacts
│
├── 📊 reports/                           # Generated visualizations & analysis
│   ├── figures/                         # Model performance plots
│   │   ├── *_confusion_matrix.png       # Confusion matrices
│   │   ├── *_roc.png                    # ROC curves
│   │   ├── *_pr.png                     # Precision-Recall curves
│   │   └── *_errors.png                 # Error analysis plots
│   └── tables/                          # Performance metrics tables
│
├── 🚀 huggingface_space/                # Production deployment
│   ├── app.py                           # Gradio web interface
│   ├── api_app.py                       # FastAPI REST API
│   ├── start.py                         # Deployment starter
│   ├── requirements.txt                 # Production dependencies
│   ├── README.md                        # Deployment documentation
│   ├── xgboost_retrained_tuned.pkl      # Production model
│   └── preprocessing/                   # Production preprocessing
│       ├── label_encoders.pkl
│       └── scaler.pkl
│
├── ⚙️ config/                            # Configuration files
│   └── data_config.yaml                 # Data processing configuration
│
├── 📄 Documentation Files
│   ├── README.md                        # This file
│   ├── DOCUMENTATION_SUMMARY.md         # Documentation overview (288 lines)
│   ├── requirements.txt                 # Project dependencies
│   └── LICENSE                          # MIT License
│
└── 🔧 Development Files
    ├── .gitignore                       # Git ignore patterns
    ├── .gitattributes                   # Git attributes
    └── .github/                         # GitHub workflows & templates
```

---

## 🗂️ Detailed Folder Descriptions

### 📊 **Notebooks** (`/notebooks/`)
Contains the complete machine learning workflow in 5 sequential Jupyter notebooks:

1. **Dataset Justification & Literature Review** (`01_*.ipynb`)
   - Dataset source, size, and structure analysis
   - Literature review of 5+ peer-reviewed studies
   - Business problem definition and significance
   - Real-world impact assessment

2. **Data Merging & Preprocessing** (`02_*.ipynb`)
   - Merging two UCI dataset variants (45K + 41K samples)
   - Missing value analysis and handling
   - Data quality assessment and cleaning
   - Initial feature transformations

3. **Exploratory Data Analysis** (`03_*.ipynb`)
   - Comprehensive EDA with 88:12 class imbalance analysis
   - Outlier detection and selective treatment
   - SMOTE implementation for class balancing
   - Feature engineering (5 new domain-informed features)
   - Correlation analysis and business insights

4. **Model Development** (`04_*.ipynb`)
   - Training 6 diverse ML algorithms
   - MLflow experiment tracking setup
   - Class imbalance handling with SMOTE + class weights
   - Model comparison and selection (LightGBM chosen)

5. **Evaluation & Comparison** (`05_*.ipynb`)
   - Multi-metric evaluation framework
   - ROC and Precision-Recall curve analysis
   - Hyperparameter tuning with GridSearchCV
   - Threshold optimization for business goals
   - Final model validation (ROC-AUC: 0.94)

### 📚 **Documentation** (`/docs/`)
Comprehensive project documentation with **1,600+ lines** total:

- **`PROJECT_OVERVIEW.md`** (493 lines): Complete project guide with workflows, phases, and results
- **`notebook_01_data_preprocessing.md`** (450 lines): Data merging strategies and preprocessing techniques
- **`notebook_02_exploratory_analysis.md`** (721 lines): EDA methodology and feature engineering
- **`notebook_03_model_development.md`** (18KB): Model selection rationale and training details
- **`notebook_04_evaluation.md`** (21KB): Performance evaluation and business translation

### 🗂️ **Data** (`/data/`)
Structured data storage following best practices:

- **`raw/`**: Original merged datasets (86,399 samples)
- **`interim/`**: Processed data at various stages
  - Outlier-treated datasets
  - Feature-engineered versions
  - SMOTE-balanced training sets

### 📈 **Datasets** (`/dataset/`)
Original UCI Machine Learning Repository data:

- **`bank/`**: Original 2011 dataset (45,211 samples, 16 features)
- **`bank-additional/`**: Enhanced 2014 dataset (41,188 samples, 20 features + economic indicators)

### 🤖 **Models** (`/models/`)
Production-ready trained models:

| Model | ROC-AUC | Training Time | Status |
|-------|---------|---------------|---------|
| **LightGBM** | **0.93** | **8s** | **✅ Selected** |
| XGBoost | 0.92 | 12s | ✅ Production |
| CatBoost | 0.92 | 25s | ✅ Available |
| Random Forest | 0.89 | 45s | ✅ Available |
| Logistic Regression | 0.78 | 2s | ✅ Baseline |
| Neural Network | 0.87 | 180s | ✅ Available |

### 🧪 **Experiments** (`/experiments/`)
MLflow experiment tracking with complete audit trail:
- Model parameters and hyperparameters
- Performance metrics across all runs
- Artifacts (plots, model files, logs)
- Reproducible experiment history

### 📊 **Reports** (`/reports/`)
Generated visualizations and analysis:
- **`figures/`**: Performance plots (confusion matrices, ROC curves, PR curves)
- **`tables/`**: Structured performance metrics and comparisons

### 🚀 **Hugging Face Space** (`/huggingface_space/`)
Production deployment with dual interfaces:
- **Gradio Web Interface** (`app.py`): Interactive web form
- **FastAPI REST API** (`api_app.py`): Programmatic access
- **Live Demo**: [🤗 Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🎯 Business Problem & Impact

### **Problem Statement**
Predict whether bank clients will subscribe to term deposits during telemarketing campaigns to optimize marketing efficiency and customer experience.

### **Business Value**
- **Cost Optimization**: 50-70% reduction in unnecessary calls
- **Revenue Enhancement**: 20-30% increase in deposit acquisition
- **Customer Experience**: Reduced spam, improved satisfaction
- **Strategic Planning**: Data-driven campaign optimization

### **Real-World Application**
- **Target Industry**: Banking and Financial Services
- **Use Case**: Direct marketing campaign optimization
- **Regulatory Compliance**: GDPR-compliant model interpretability
- **Scalability**: Cloud-ready deployment architecture

---

## 📊 Dataset Information

### **Source**: UCI Machine Learning Repository
- **Original Institution**: Portuguese banking institution
- **Data Collection**: May 2008 - November 2010
- **Domain**: Banking / Direct Marketing

### **Dataset Characteristics**
- **Total Samples**: 86,399 (merged from two variants)
- **Features**: 20 input features + 1 target variable
- **Class Distribution**: 88% No, 12% Yes (imbalanced)
- **Data Types**: Numeric, categorical, binary

### **Feature Categories**
1. **Bank Client Data** (8 features): Demographics and financial profile
2. **Contact Information** (5 features): Communication details and timing
3. **Campaign Information** (4 features): Marketing history and outcomes
4. **Economic Context** (5 features): Macroeconomic indicators
5. **Target Variable**: Term deposit subscription (yes/no)

---

## 🚀 Getting Started

### **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### **Quick Start**
```bash
# Clone repository
git clone https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml.git
cd bank-marketing-term-deposit-ml

# Install dependencies
pip install -r requirements.txt

# Run notebooks sequentially
jupyter notebook notebooks/01_dataset_justification_and_literature_review.ipynb
```

### **Project Workflow**
1. **Dataset Analysis** → Notebook 01
2. **Data Preprocessing** → Notebook 02  
3. **Exploratory Analysis** → Notebook 03
4. **Model Development** → Notebook 04
5. **Evaluation & Selection** → Notebook 05
6. **Deployment** → Hugging Face Space

---

## 🛠️ Technical Architecture

### **Machine Learning Pipeline**
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
    ↓            ↓               ↓                 ↓              ↓           ↓
  86K rows   Missing Value   +5 Features      6 Algorithms   ROC-AUC 0.94  Live API
             Handling        Engineering      Comparison
```

### **Technology Stack**
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: XGBoost, LightGBM, CatBoost, PyTorch
- **Experiment Tracking**: MLflow
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: FastAPI, Gradio, Hugging Face Spaces
- **Development**: Jupyter Notebooks, Git

### **Model Performance**
- **Best Model**: LightGBM Classifier
- **ROC-AUC**: 0.94 (Excellent)
- **Precision**: 78% (Business-relevant)
- **Recall**: 67% (Subscriber identification)
- **F1-Score**: 72% (Balanced performance)

---

## 📈 Results & Performance

### **Model Comparison**
| Algorithm | ROC-AUC | Precision | Recall | F1-Score | Training Time |
|-----------|---------|-----------|--------|----------|---------------|
| **LightGBM** | **0.94** | **0.78** | **0.67** | **0.72** | **8s** |
| XGBoost | 0.92 | 0.75 | 0.65 | 0.70 | 12s |
| CatBoost | 0.92 | 0.76 | 0.64 | 0.69 | 25s |
| Random Forest | 0.89 | 0.71 | 0.58 | 0.64 | 45s |
| Neural Network | 0.87 | 0.68 | 0.62 | 0.65 | 180s |
| Logistic Regression | 0.78 | 0.58 | 0.45 | 0.51 | 2s |

### **Business Impact Metrics**
- **Cost Reduction**: 50-70% fewer unnecessary calls
- **Conversion Rate**: Improved from 12% to 18-22%
- **Annual Savings**: €500K+ for large banking institutions
- **Revenue Increase**: €2M+ through optimized targeting

---

## 🔗 Live Demo & API

### **🌐 Web Interface**
Interactive Gradio interface for single predictions:
- **URL**: [Hugging Face Spaces Demo](https://huggingface.co/spaces)
- **Features**: Form-based input, real-time predictions
- **Usage**: Manual testing and demonstrations

### **🔌 REST API**
FastAPI endpoints for programmatic access:
- **Base URL**: `/api/v1/`
- **Endpoints**:
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions
  - `GET /health` - API health check
  - `GET /docs` - Interactive API documentation

### **API Usage Example**
```python
import requests

# Single prediction
response = requests.post(
    "https://your-space-url/predict",
    json={
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "university.degree",
        "balance": 1500,
        # ... other features
    }
)

prediction = response.json()
print(f"Subscription Probability: {prediction['probability']:.2%}")
```

---

## 📚 Documentation Index

| Document | Description | Lines | Content |
|----------|-------------|-------|---------|
| [`README.md`](README.md) | Project overview & setup | This file | Complete guide |
| [`DOCUMENTATION_SUMMARY.md`](DOCUMENTATION_SUMMARY.md) | Documentation index | 288 | All docs overview |
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | Complete project guide | 493 | Workflow & results |
| [`docs/notebook_01_data_preprocessing.md`](docs/notebook_01_data_preprocessing.md) | Data preparation | 450 | Merging & cleaning |
| [`docs/notebook_02_exploratory_analysis.md`](docs/notebook_02_exploratory_analysis.md) | EDA & features | 721 | Analysis & engineering |
| [`docs/notebook_03_model_development.md`](docs/notebook_03_model_development.md) | Model training | 18KB | Algorithm comparison |
| [`docs/notebook_04_evaluation.md`](docs/notebook_04_evaluation.md) | Performance eval | 21KB | Metrics & validation |
| [`huggingface_space/README.md`](huggingface_space/README.md) | Deployment guide | 201 | API documentation |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Bank Marketing Dataset
- **Original Authors**: Sérgio Moro, Paulo Cortez, Paulo Rita
- **Institution**: Portuguese banking institution for data collection
- **Research Papers**: CRISP-DM methodology and economic indicators research

---

## 📞 Contact

**Lahiru Munasinghe**
- **GitHub**: [@lahirumanulanka](https://github.com/lahirumanulanka)
- **Project Repository**: [bank-marketing-term-deposit-ml](https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml)
- **Live Demo**: [🤗 Hugging Face Spaces](https://huggingface.co/spaces)

---

*Last Updated: October 8, 2025*
