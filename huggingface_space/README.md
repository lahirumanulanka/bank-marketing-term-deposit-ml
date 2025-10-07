---
title: Bank Marketing Term Deposit Prediction API
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: mit
---

# 🏦 Bank Marketing Term Deposit Prediction

This Hugging Face Space provides both a **Gradio web interface** and a **FastAPI REST API** for predicting whether a client will subscribe to a term deposit based on marketing campaign data.

## 🚀 Quick Start

### Option 1: Gradio Web Interface
- **File**: `app.py`
- **Access**: Direct web interface
- **Usage**: Interactive form-based prediction

### Option 2: FastAPI REST API
- **File**: `api_app.py`  
- **Access**: RESTful API endpoints
- **Usage**: Programmatic access for applications

## 📊 Model Information

- **Algorithm**: LightGBM Classifier
- **Dataset**: UCI Bank Marketing Dataset (Portuguese banking institution, 2008-2010)
- **Features**: 20 input features including demographic, financial, campaign, and economic indicators
- **Performance**: ROC-AUC ~0.93
- **Training**: 86,000+ samples with SMOTE for class imbalance handling

## 🔗 API Endpoints

When using the FastAPI version (`api_app.py`):

### Core Endpoints
- `GET /` - API information and available endpoints
- `GET /health` - Health check and model status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Prediction Endpoints
- `POST /predict` - Single client prediction
- `POST /predict/batch` - Batch prediction for multiple clients

### Information Endpoints
- `GET /model/info` - Model metadata and performance
- `GET /features/info` - Detailed feature descriptions

## 📝 API Usage Examples

### Single Prediction
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
print(f"Probability: {result['probability_percentage']}")
```

### Batch Prediction
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

## 🎯 Input Features

### Demographic (4 features)
- **age**: Client age (18-95)
- **job**: Job type (admin, blue-collar, entrepreneur, etc.)
- **marital**: Marital status (divorced, married, single, unknown)
- **education**: Education level (basic.4y, basic.6y, basic.9y, etc.)

### Financial (3 features)
- **default**: Has credit in default? (yes, no, unknown)
- **housing**: Has housing loan? (yes, no, unknown)
- **loan**: Has personal loan? (yes, no, unknown)

### Campaign (8 features)
- **contact**: Contact type (cellular, telephone)
- **month**: Last contact month (jan-dec)
- **day_of_week**: Last contact day (mon-fri)
- **duration**: Last contact duration in seconds
- **campaign**: Number of contacts during this campaign
- **pdays**: Days since last contact (999 if never contacted)
- **previous**: Number of contacts before this campaign
- **poutcome**: Previous campaign outcome (failure, nonexistent, success)

### Economic (5 features)
- **emp_var_rate**: Employment variation rate
- **cons_price_idx**: Consumer price index
- **cons_conf_idx**: Consumer confidence index
- **euribor3m**: Euribor 3 month rate
- **nr_employed**: Number of employees

## 📁 Files

- `app.py` - Gradio web interface
- `api_app.py` - FastAPI REST API server
- `test_api.html` - HTML test interface for API
- `test_api_client.py` - Python client example
- `requirements.txt` - Dependencies
- Model files:
  - `lightgbm_retrained_tuned.pkl` - Trained model
  - `preprocessing/scaler.pkl` - Feature scaler
  - `preprocessing/label_encoders.pkl` - Categorical encoders

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Gradio interface
python app.py

# Run FastAPI server
python api_app.py

# Test API
python test_api_client.py
```

## 📈 Model Performance

- **ROC-AUC**: ~0.93
- **Training Data**: 86,000+ samples
- **Class Balance**: SMOTE applied for imbalanced dataset
- **Validation**: Cross-validation and holdout testing
- **Optimization**: Hyperparameter tuning with Optuna

## 🔍 Confidence Levels

The API provides confidence interpretations:
- **Very High**: ≥80% probability
- **High**: 70-79% probability  
- **Moderate**: 60-69% probability
- **Low**: 50-59% probability
- **Very Low**: <50% probability

## 📄 License

MIT License - see repository for details.

## 👨‍💻 Author

**Lahiru Manulanka Munasinghe**
- GitHub: [@lahirumanulanka](https://github.com/lahirumanulanka)
- Repository: [bank-marketing-term-deposit-ml](https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml)

## 🎓 Citation

If you use this model in your research, please cite:
```
@software{munasinghe2024bank,
  author = {Lahiru Manulanka Munasinghe},
  title = {Bank Marketing Term Deposit Prediction},
  year = {2024},
  url = {https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml}
}
```
