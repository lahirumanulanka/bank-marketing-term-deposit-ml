# 🚀 Deployment Quick Reference

## Overview

This directory contains everything needed to deploy the Bank Marketing Term Deposit Prediction model to production.

## 📁 What's Included

### 1. HuggingFace Spaces Deployment
- **Location**: `../huggingface_space/`
- **Purpose**: Interactive web UI with Gradio
- **Deployment Time**: <10 minutes
- **Cost**: Free (basic tier)

### 2. FastAPI REST API
- **Location**: `./api.py`
- **Purpose**: Programmatic model access
- **Deployment**: Docker or local
- **Documentation**: Auto-generated at `/docs`

### 3. Docker Configuration
- **Files**: `Dockerfile`, `docker-compose.yml`
- **Purpose**: Containerized deployment
- **Includes**: API + Prometheus + Grafana

### 4. Monitoring Tools
- **Location**: `./monitoring/`
- **Features**: Drift detection, performance tracking
- **Technology**: Custom Python + Prometheus

### 5. CI/CD Pipelines
- **Location**: `../.github/workflows/`
- **Workflows**: Testing, deployment automation
- **Platform**: GitHub Actions

## 🎯 Quick Start Options

### Option 1: Deploy to HuggingFace Spaces (Recommended for Demos)

```bash
# 1. Navigate to HuggingFace Space directory
cd ../huggingface_space/

# 2. Copy model files (adjust paths as needed)
cp ../models/lightgbm_retrained_tuned.pkl .
mkdir preprocessing
cp ../models/preprocessing/*.pkl preprocessing/

# 3. Create a Space on HuggingFace.co

# 4. Clone your Space repository
git clone https://huggingface.co/spaces/<username>/bank-marketing-prediction
cd bank-marketing-prediction

# 5. Copy files
cp /path/to/huggingface_space/* .

# 6. Initialize Git LFS and push
git lfs install
git lfs track "*.pkl"
git add .
git commit -m "Initial deployment"
git push
```

**Result**: Live web app at `https://huggingface.co/spaces/<username>/bank-marketing-prediction`

### Option 2: Run FastAPI Locally

```bash
# 1. Install dependencies
cd deployment
pip install -r requirements.txt

# 2. Run the API
python api.py

# 3. Access the API
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - Health check: http://localhost:8000/health
```

### Option 3: Docker Deployment

```bash
# Start all services (API + Monitoring)
cd deployment
docker-compose up -d

# Access services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Option 4: Automated CI/CD

```bash
# 1. Add HuggingFace token to GitHub Secrets
# - Go to repository Settings → Secrets
# - Add secret: HF_TOKEN = <your-token>

# 2. Push changes to trigger deployment
git push origin main
```

## 📚 Documentation

- **Comprehensive Guide**: `../docs/DEPLOYMENT_GUIDE.md` (400+ lines)
- **Quick Start**: `../docs/HUGGINGFACE_QUICKSTART.md` (<10 min)
- **Implementation Summary**: `../docs/IMPLEMENTATION_SUMMARY.md`
- **Validation Checklist**: `../docs/VALIDATION_CHECKLIST.md`
- **Deployment Notebook**: `../notebooks/08_deployment_strategy.ipynb`

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
MODEL_VERSION=1.0.0
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

### Model Files Required

```
models/
├── lightgbm_retrained_tuned.pkl    # Main model
└── preprocessing/
    ├── scaler.pkl                   # Feature scaler
    └── label_encoders.pkl           # Categorical encoders
```

## 🧪 Testing

### Test API Locally

```bash
# Run example script
cd deployment
python examples/api_usage.py
```

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

### Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

## 📊 Monitoring

### View Metrics

```bash
# Start monitoring stack
docker-compose up -d

# Access Grafana
open http://localhost:3000
# Login: admin/admin
```

### Check Drift

```python
from monitoring.model_monitor import ModelMonitor

monitor = ModelMonitor(reference_data=train_data)
drift_report = monitor.detect_data_drift(new_data)
print(drift_report)
```

## 🔒 Security

### API Authentication (Optional)

Add to `api.py`:

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(data: ClientData, credentials: HTTPBearer = Depends(security)):
    # Verify token
    pass
```

### Rate Limiting

Install `slowapi`:

```bash
pip install slowapi
```

Add to `api.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("5/minute")
async def predict(...):
    pass
```

## 🐛 Troubleshooting

### Model Not Loading

```bash
# Check model file exists
ls -lh ../models/*.pkl

# Test loading manually
python -c "import pickle; pickle.load(open('../models/lightgbm_retrained_tuned.pkl', 'rb'))"
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Docker Issues

```bash
# View logs
docker-compose logs -f

# Rebuild images
docker-compose build --no-cache

# Reset everything
docker-compose down -v
docker-compose up --build
```

## 📈 Performance

### Expected Metrics

- **Latency**: <50ms (p95)
- **Throughput**: 100+ req/s (single instance)
- **Memory**: ~500MB
- **CPU**: <10% (idle)

### Optimization Tips

1. **Use Gunicorn with multiple workers**:
   ```bash
   gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Enable caching**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1)
   def load_model():
       # Load model once
       pass
   ```

3. **Batch predictions**:
   ```python
   @app.post("/predict/batch")
   async def predict_batch(data: List[ClientData]):
       # Process multiple predictions
       pass
   ```

## 🚀 Production Checklist

- [ ] Environment variables configured
- [ ] Model files validated
- [ ] API health check passing
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Security measures applied
- [ ] Performance tested
- [ ] Documentation reviewed
- [ ] Backup strategy in place
- [ ] Rollback plan documented

## 📞 Support

- **Issues**: Open a GitHub issue
- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` directory

## 🎓 Learning Resources

- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Docker Documentation](https://docs.docker.com/)
- [HuggingFace Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://www.gradio.app/docs/)

## 📝 Version History

- **v1.0.0** (Current): Initial deployment implementation
  - HuggingFace Spaces support
  - FastAPI REST API
  - Docker containerization
  - CI/CD pipelines
  - Monitoring and drift detection

## 🌟 Features

✅ Multiple deployment options (HF Spaces, Docker, Cloud)
✅ Automatic API documentation
✅ Real-time monitoring
✅ Drift detection
✅ CI/CD automation
✅ Comprehensive logging
✅ Security best practices
✅ Production-ready code

---

**Happy Deploying!** 🚀

For detailed instructions, see: `../docs/DEPLOYMENT_GUIDE.md`
