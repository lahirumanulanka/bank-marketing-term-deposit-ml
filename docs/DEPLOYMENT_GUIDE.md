# HuggingFace Spaces Deployment Guide

This guide provides comprehensive instructions for deploying the Bank Marketing Term Deposit Prediction model to HuggingFace Spaces with API access.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Steps](#deployment-steps)
- [Versioning Strategy](#versioning-strategy)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Monitoring](#model-monitoring)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)

## Overview

The deployment solution includes:

1. **HuggingFace Spaces** - Gradio web interface for interactive predictions
2. **FastAPI REST API** - Programmatic access to the model
3. **Docker Containerization** - Consistent deployment environment
4. **GitHub Actions CI/CD** - Automated testing and deployment
5. **Monitoring & Drift Detection** - Track model performance over time

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Users                                    │
└───────────┬─────────────────────────────────┬──────────────────┘
            │                                 │
            ▼                                 ▼
    ┌───────────────┐                ┌────────────────┐
    │  HuggingFace  │                │   FastAPI      │
    │    Spaces     │                │   REST API     │
    │   (Gradio)    │                │                │
    └───────┬───────┘                └────────┬───────┘
            │                                 │
            │       ┌─────────────────────┐   │
            └───────►  LightGBM Model     ◄───┘
                    │  + Preprocessors    │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Monitoring        │
                    │   - Drift Detection │
                    │   - Performance     │
                    │   - Logging         │
                    └─────────────────────┘
```

## Prerequisites

### Required Accounts
- GitHub account (for repository and CI/CD)
- HuggingFace account (for Spaces deployment)

### Required Tools
- Git (with Git LFS for large files)
- Python 3.10+
- Docker (optional, for local testing)

### Model Files
Ensure you have the following trained model files:
- `models/lightgbm_retrained_tuned.pkl`
- `models/preprocessing/scaler.pkl`
- `models/preprocessing/label_encoders.pkl`

## Deployment Steps

### 1. HuggingFace Spaces Deployment

#### Manual Deployment

1. **Create a new Space on HuggingFace:**
   ```bash
   # Install HuggingFace CLI
   pip install huggingface_hub
   
   # Login to HuggingFace
   huggingface-cli login
   ```

2. **Clone your Space repository:**
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/bank-marketing-prediction
   cd bank-marketing-prediction
   ```

3. **Copy files from this repository:**
   ```bash
   cp ../huggingface_space/* .
   cp ../models/lightgbm_retrained_tuned.pkl .
   mkdir preprocessing
   cp ../models/preprocessing/*.pkl preprocessing/
   ```

4. **Initialize Git LFS (for large model files):**
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

6. **Your Space will be available at:**
   ```
   https://huggingface.co/spaces/<your-username>/bank-marketing-prediction
   ```

#### Automated Deployment with GitHub Actions

1. **Create HuggingFace Token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with write permissions
   - Copy the token

2. **Add token to GitHub Secrets:**
   - Go to your GitHub repository settings
   - Navigate to Secrets and Variables → Actions
   - Create new secret named `HF_TOKEN`
   - Paste your HuggingFace token

3. **Update workflow file:**
   - Edit `.github/workflows/deploy-huggingface.yml`
   - Replace `<your-username>` with your HuggingFace username

4. **Trigger deployment:**
   - Push changes to `main` branch
   - Or manually trigger from GitHub Actions tab

### 2. FastAPI Deployment

#### Local Deployment

1. **Install dependencies:**
   ```bash
   cd deployment
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```bash
   python api.py
   ```

3. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

#### Docker Deployment

1. **Build the Docker image:**
   ```bash
   cd deployment
   docker build -t bank-marketing-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 bank-marketing-api
   ```

#### Docker Compose with Monitoring

1. **Start all services:**
   ```bash
   cd deployment
   docker-compose up -d
   ```

2. **Access services:**
   - API: http://localhost:8000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## Versioning Strategy

### Model Versioning

We use **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes in model API or significant accuracy changes
- **MINOR**: New features, improved performance, backward compatible
- **PATCH**: Bug fixes, minor improvements

**Current Version: 1.0.0**

#### Version Tracking

1. **Model Metadata:**
   ```python
   model_metadata = {
       "version": "1.0.0",
       "trained_date": "2024-01-01",
       "dataset": "UCI Bank Marketing",
       "algorithm": "LightGBM",
       "performance": {"roc_auc": 0.93}
   }
   ```

2. **Version Tags:**
   ```bash
   # Create version tag
   git tag -a v1.0.0 -m "Initial model release"
   git push origin v1.0.0
   ```

3. **Model Registry:**
   - Use MLflow Model Registry for versioning
   - Track experiments and model versions
   - Enable easy rollback

### Data Versioning

- Use DVC (Data Version Control) for dataset versioning
- Tag dataset versions with model versions
- Document data schema changes

### Code Versioning

- Git for source code
- Branch strategy: main, develop, feature/*
- Pull requests for all changes
- Code review required before merge

## CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Continuous Integration (`ci-cd.yml`)

**Triggers:**
- Push to main or develop
- Pull requests to main

**Jobs:**
- **Test**: Run unit and integration tests
- **Validate Model**: Check model files and loading
- **Lint**: Code quality checks (black, isort, flake8)

#### 2. HuggingFace Deployment (`deploy-huggingface.yml`)

**Triggers:**
- Push to main (changes in huggingface_space/)
- Manual workflow dispatch

**Jobs:**
- Validate model files
- Copy files to Space directory
- Deploy to HuggingFace Spaces
- Verify deployment

### Pipeline Best Practices

1. **Automated Testing:**
   - Unit tests for all functions
   - Integration tests for API endpoints
   - Model validation tests

2. **Quality Gates:**
   - Code coverage > 80%
   - All tests must pass
   - Linting checks pass

3. **Deployment Checks:**
   - Model file integrity
   - API health check
   - Performance benchmarks

## Model Monitoring

### Monitoring Components

#### 1. Performance Monitoring

Track key metrics:
- **Prediction Latency**: Response time for predictions
- **Throughput**: Requests per second
- **Error Rate**: Failed predictions
- **Model Confidence**: Distribution of prediction probabilities

#### 2. Data Drift Detection

Detect changes in input data distribution:

```python
from deployment.monitoring.model_monitor import ModelMonitor

# Initialize monitor with reference data
monitor = ModelMonitor(reference_data=training_data)

# Detect drift
drift_report = monitor.detect_data_drift(new_data)

if drift_report['alert']:
    print("⚠️ Data drift detected!")
    print(f"Drifted features: {drift_report['drifted_features']}")
```

**Drift Detection Methods:**
- **Numerical features**: Kolmogorov-Smirnov test
- **Categorical features**: Chi-square test
- **Alert threshold**: >30% features show drift

#### 3. Concept Drift Detection

Monitor changes in model predictions:

```python
# Analyze prediction patterns
concept_drift = monitor.detect_concept_drift(window_size=100)

if concept_drift['alert']:
    print("⚠️ Concept drift detected!")
```

**Concept Drift Indicators:**
- Prediction probability distribution changes
- Positive prediction rate changes
- Performance degradation

#### 4. Logging

Structured JSON logging for all predictions:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "input": {...},
  "prediction": 1,
  "probability": 0.85,
  "model_version": "1.0.0"
}
```

### Monitoring Dashboard (Grafana)

1. **Setup Grafana:**
   ```bash
   docker-compose up grafana
   ```

2. **Access dashboard:** http://localhost:3000

3. **Key Metrics:**
   - Prediction rate over time
   - Probability distribution
   - Response time histogram
   - Error rate
   - Drift alerts

### Alerting Rules

Configure alerts for:
- Data drift detected (>30% features)
- Concept drift detected
- Performance degradation (>10% drop)
- High error rate (>5%)
- High latency (>1s p95)

## API Usage

### REST API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00",
  "version": "1.0.0"
}
```

#### 2. Model Information

```bash
curl http://localhost:8000/model/info
```

#### 3. Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "job": "admin.",
    "marital": "single",
    "education": "university.degree",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "duration": 180,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
  }'
```

Response:
```json
{
  "prediction": "yes",
  "probability": 0.85,
  "confidence": "high",
  "timestamp": "2024-01-01T12:00:00",
  "model_version": "1.0.0"
}
```

### Python Client Example

```python
import requests

# API endpoint
url = "http://localhost:8000/predict"

# Client data
data = {
    "age": 30,
    "job": "admin.",
    "marital": "single",
    # ... other features
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
```

## Troubleshooting

### Common Issues

#### 1. Model Not Loading

**Error:** `Model file not found`

**Solution:**
- Check model file paths
- Ensure Git LFS is installed for large files
- Verify file permissions

#### 2. Dependency Conflicts

**Error:** `ImportError: No module named 'lightgbm'`

**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

#### 3. HuggingFace Deployment Fails

**Error:** `Authentication failed`

**Solution:**
- Verify HF_TOKEN is correct
- Check token permissions (write access)
- Re-login: `huggingface-cli login`

#### 4. High Memory Usage

**Solution:**
- Use smaller model (or quantization)
- Increase container memory limits
- Optimize preprocessing

### Performance Optimization

1. **Caching:**
   - Cache model in memory
   - Use Redis for frequent predictions

2. **Batch Predictions:**
   - Process multiple predictions together
   - Reduce overhead

3. **Async Processing:**
   - Use FastAPI async endpoints
   - Non-blocking I/O

## Security Considerations

### API Security

1. **Authentication:**
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/predict")
   async def predict(data: ClientData, credentials: HTTPBearer = Depends(security)):
       # Verify token
       pass
   ```

2. **Rate Limiting:**
   - Implement request rate limits
   - Prevent abuse

3. **Input Validation:**
   - Pydantic models for validation
   - Sanitize inputs

### Data Privacy

- **GDPR Compliance:**
  - Don't log sensitive personal data
  - Implement data retention policies
  - Allow data deletion requests

- **Encryption:**
  - HTTPS for API
  - Encrypt sensitive data at rest

## Maintenance

### Regular Tasks

1. **Weekly:**
   - Review monitoring dashboards
   - Check for drift alerts
   - Analyze prediction logs

2. **Monthly:**
   - Review model performance
   - Update documentation
   - Security updates

3. **Quarterly:**
   - Retrain model with new data
   - Performance analysis
   - A/B testing new versions

### Rollback Procedure

If issues occur:

1. **Identify version to rollback to:**
   ```bash
   git tag -l
   ```

2. **Checkout previous version:**
   ```bash
   git checkout v0.9.0
   ```

3. **Redeploy:**
   ```bash
   git push origin main --force
   ```

4. **Verify rollback:**
   - Check API health
   - Test predictions
   - Monitor logs

## Support

For issues or questions:
- GitHub Issues: https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml/issues
- Email: [Your email]

## References

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Docker Documentation](https://docs.docker.com/)
